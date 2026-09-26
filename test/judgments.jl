# Judgments (MAP-14): DataPart, the schema convention, wire rewrites, and the vLLM
# token-trie driver, all offline. The trie scenario mirrors the reference's own
# test (lm15-python tests/test_judgments.py) so the two implementations agree.
const WINE = judgments(
    "quality" => score("How good?", ["bad" => "Bad wine", "ok" => "Fine wine", "great" => "Great wine"]),
    "style" => choice("Which style?", ["fruit" => "Fruit-forward", "oak" => "Oak-driven", "other" => nothing]),
    "ageing" => yes_no("Will it age?"),
)
const NOTE = user("Blackberry, oak, decades ahead.")
const TRIE_PREFIX = [1, 2, 3]
const TRIE_PATHS = Dict("fruit" => [10, 99], "oak" => [11, 99], "other" => [12, 99], "true" => [20, 99], "false" => [21, 99])

function trie_transport(logprobs; drop_ids=false, chat_reply=nothing, seen=Ref{Any}(nothing))
    return function (request)
        payload = JSON.parse(String(copy(request.body)))
        if endswith(request.url, "/tokenize")
            answer = payload["messages"][end]["content"]
            key = strip(replace(answer, "Answer:" => ""))
            tokens = vcat(TRIE_PREFIX, isempty(key) ? Int[] : TRIE_PATHS[key][1:1])
            !isempty(key) && !payload["continue_final_message"] && append!(tokens, TRIE_PATHS[key][2:end], [7])
            return HttpResponse(; status=200, body=Vector{UInt8}(codeunits(JSON.serialize(Dict("tokens" => tokens)))))
        elseif endswith(request.url, "/chat/completions")
            return HttpResponse(; status=200, body=Vector{UInt8}(codeunits(JSON.serialize(chat_reply))))
        end
        seen[] = payload["prompt"]
        choices = [Dict("index" => i - 1, "text" => "x", "finish_reason" => "length", "logprobs" => Dict("top_logprobs" =>
            [drop_ids ? Dict() : Dict("token_id:$t" => get(logprobs, t, -9.0) for t in payload["logprob_token_ids"])]))
            for i in eachindex(payload["prompt"])]
        return HttpResponse(; status=200, body=Vector{UInt8}(codeunits(JSON.serialize(Dict("model" => "m", "choices" => choices,
            "usage" => Dict("prompt_tokens" => 40, "completion_tokens" => length(choices)))))))
    end
end

@testset "judgments: data parts, convention, wires" begin
    @test data(Dict("a" => 1)) isa DataPart
    @test to_dict(data(nothing)) == Dict("type" => "data", "value" => nothing)
    @test_throws ArgumentError DataPart(; value=1, probabilities=Dict("x" => Dict("a" => 0.5)))  # method missing
    @test_throws ArgumentError user(DataPart(; value=1, probabilities=Dict("x" => Dict("a" => 1.0)), method="provider_classification"))
    found = judgments_in_schema(WINE["schema"])
    @test collect(keys(found)) == ["quality", "style", "ageing"]
    @test found["quality"].kind == "ordered" && found["style"].kind == "choice" && found["ageing"].kind == "boolean"
    @test expected_level(Dict("0" => 0.2, "1" => 0.3, "2" => 0.5)) ≈ 1.3
    g = build_request(GeminiLM(; api_key="k"), Request("gemini-2.5-flash", NOTE; config=Config(; response_format=WINE)))
    body = JSON.parse(String(copy(g.body)))
    @test body["generationConfig"]["responseJsonSchema"]["properties"]["style"]["enum"] == ["fruit", "oak", "other"]
    o = JSON.parse(String(copy(build_request(OpenAILM(; api_key="k"), Request("gpt-5-mini", NOTE; config=Config(; response_format=WINE))).body)))
    @test o["text"]["format"]["schema"] == WINE["schema"]  # verbatim
    # A wire that measures nothing: if_available records, required refuses.
    r = Request("claude-haiku-4-5", NOTE; config=Config(; response_format=WINE, probabilities="if_available"))
    @test [(a.field, a.action) for a in plan(AnthropicLM(; api_key="k"), r)] ⊇ [("config.probabilities", "dropped")]
    @test_throws UnsupportedFeatureError plan(AnthropicLM(; api_key="k"), Request(r; config=Config(r.config; probabilities="required")))
end

@testset "judgments: the token trie on a server that scores named tokens" begin
    fmt = judgments("style" => choice("Which style?", ["fruit", "oak", "other"]), "ageing" => yes_no("Will it age?"))
    seen = Ref{Any}(nothing)
    lm = OpenAIChatLM(; api_key="k", base_url="http://vllm:8001/v1", compat="vllm",
        transport=trie_transport(Dict(10 => -0.1, 11 => -2.0, 12 => -3.0, 20 => -0.05, 21 => -3.0, 99 => -0.01); seen))
    r = complete(lm, Request("m", NOTE; config=Config(; response_format=fmt, probabilities="required")))
    @test data_part(r).method == "candidate_sequence_likelihood"
    @test data(r) == Dict("style" => "fruit", "ageing" => true)
    @test isapprox(probabilities(r)["style"]["fruit"], 0.830; atol=0.005)
    @test r.provider_data["judgments"] == Dict("nodes" => 7, "tokenize_calls" => 12, "method" => "candidate_sequence_likelihood")
    @test Set(keys(r.provider_data["coverage"])) == Set(["style", "ageing"])
    @test length(seen[]) == 7 && seen[][1] == TRIE_PREFIX  # the root node is the prefill alone
    @test isempty(r.adaptations)  # a requested measurement is not itself an adaptation

    # A server that drops logprob_token_ids: if_available answers by structured output
    # and records it; required refuses.
    ageing = judgments("ageing" => yes_no("Will it age?"))
    reply = Dict("id" => "c", "model" => "m", "choices" => [Dict("index" => 0, "finish_reason" => "stop",
        "message" => Dict("role" => "assistant", "content" => "{\"ageing\": true}"))], "usage" => Dict("prompt_tokens" => 1, "completion_tokens" => 1))
    lm = OpenAIChatLM(; api_key="k", base_url="http://vllm:8000/v1", compat="vllm", transport=trie_transport(Dict(); drop_ids=true, chat_reply=reply))
    r = complete(lm, Request("m", NOTE; config=Config(; response_format=ageing, probabilities="if_available")))
    @test data(r) == Dict("ageing" => true) && probabilities(r) === nothing
    @test ("config.probabilities", "dropped") in [(a.field, a.action) for a in r.adaptations]
    @test_throws UnsupportedFeatureError complete(lm, Request("m", NOTE; config=Config(; response_format=ageing, probabilities="required")))

    # Off: an ordinary structured call, folded into a DataPart, nothing recorded.
    calls = Ref(0)
    plain = OpenAIChatLM(; api_key="k", base_url="http://vllm:8001/v1", compat="vllm", transport=request -> begin
        calls[] += 1
        HttpResponse(; status=200, body=Vector{UInt8}(codeunits(JSON.serialize(Dict(reply..., "choices" => [Dict("index" => 0,
            "finish_reason" => "stop", "message" => Dict("role" => "assistant", "content" => "{\"ageing\": false}"))])))))
    end)
    r = complete(plain, Request("m", NOTE; config=Config(; response_format=ageing)))
    @test data(r) == Dict("ageing" => false) && probabilities(r) === nothing && isempty(r.adaptations) && calls[] == 1
    @test resolve(LMRouter(), "jev-latest").provider == "typesafe"
end
