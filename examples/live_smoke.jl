# Live smoke: real requests through LM15.jl, with receipts. It needs keys (or saved
# sign-ins) and costs a little money, so it is not a test and no CI runs it; the contract
# harness is the gate. The same bindings and checks as lm15-go examples/live_smoke.
#
#   set -a; . ../.env; set +a
#   julia --project=. examples/live_smoke.jl receipts/DATE-live-smoke            # API keys
#   julia --project=. examples/live_smoke.jl receipts/DATE-live-smoke --managed  # plus saved sign-ins
#   julia --project=. examples/live_smoke.jl OUT --only openai,groq              # a subset
#
# Checks, through LMRouter as a program uses it:
#   hello   one complete and one stream of the same request: equal text and finish
#           reason, and the text chunks concatenate to the assembled text;
#   order   a json_schema listing "reasoning" then "answer"; the body sent keeps that
#           order, and the model's JSON is reported with its key order;
#   tools   a function tool; the model calls it, the result goes back, it answers;
#   models  the credential's catalog lists something;
#   judgments (typesafe only) three declared judgments, probabilities "required".
# Receipts: one JSON file per binding and check with each request as sent (credential
# headers and query keys redacted), the reply status, and what LM15 made of it. Run
# lm15-contract/tools/check_secrecy.py --root on the directory before keeping it.

using LM15
using LM15: JSON, JSONObject, obj

const BINDINGS = [
    (name="openai", key="OPENAI_API_KEY", model="gpt-5-mini", tokens=4000),
    (name="anthropic", key="ANTHROPIC_API_KEY", model="anthropic:claude-haiku-4-5", tokens=1000),
    (name="gemini", key="GEMINI_API_KEY", model="gemini:gemini-2.5-flash", tokens=4000),
    (name="groq", key="GROQ_API_KEY", model="groq:openai/gpt-oss-20b", tokens=2000),
    (name="openrouter", key="OPENROUTER_API_KEY", model="openrouter:openai/gpt-4.1-nano", tokens=1000),
    (name="deepseek", key="DEEPSEEK_API_KEY", model="deepseek:deepseek-v4-flash", tokens=2000),
    (name="zai", key="ZAI_API_KEY", model="zai:glm-5.3-flash", tokens=2000),
    (name="meta", key="META_API_KEY", model="meta:muse-spark-1.3", tokens=2000),
    (name="moonshotai", key="MOONSHOTAI_API_KEY", model="moonshotai:kimi-k3", tokens=2000),
    (name="deepinfra", key="DEEPINFRA_API_KEY", model="deepinfra:deepseek-ai/DeepSeek-V4.1-Flash", tokens=2000),
    (name="together", key="TOGETHER_API_KEY", model="together:meta-llama/Llama-3.3-70B-Instruct-Turbo", tokens=1000),
    (name="fireworks", key="FIREWORKS_API_KEY", model="fireworks:accounts/fireworks/models/deepseek-v4p1-flash", tokens=2000),
    (name="parasail", key="PARASAIL_API_KEY", model="parasail:meta-llama/Llama-3.3-70B-Instruct", tokens=1000),
    (name="typesafe", key="TYPESAFE_API_KEY", model="typesafe:jev-latest", tokens=0),
]
const MANAGED = [
    (name="xai-account", key="", model="xai:grok-4.7", tokens=2000),
    (name="claude-code-account", key="", model="claude-code:claude-haiku-4-5", tokens=1000),
    (name="openrouter-account", key="", model="openrouter:openai/gpt-4.1-nano", tokens=1000),
    (name="github-copilot-account", key="", model="github-copilot:gpt-4.1", tokens=1000),
]
# The provider answers 400 to these, as receipted; LM15 sends what the reference sends.
const PROVIDER_REFUSES = Dict(("deepseek", "order") => "DeepSeek has no json_schema mode and answers 400")

# ─── a transport that records what was sent (redacted) and the reply's head ──
struct Recorder <: LM15.AbstractTransport
    inner::LM15.HTTPTransport
    log::Vector{Any}
end
const SECRET_HEADERS = ("authorization", "x-api-key", "api-key", "x-goog-api-key")
function LM15.open_response(f, t::Recorder, request::LM15.WireRequest)
    uri = LM15.HTTP.URI(request.url)
    url = isempty(uri.query) ? request.url :
        replace(request.url, r"([?&]key=)[^&]*" => s"\1<redacted>")
    sent = obj("method" => request.method, "url" => url,
        "headers" => obj((lowercase(k) => (lowercase(k) in SECRET_HEADERS ? "<redacted>" : v) for (k, v) in request.headers)...),
        "body" => try
            JSON.parse(String(copy(request.body)))
        catch
            isempty(request.body) ? nothing : "<$(length(request.body)) bytes>"
        end)
    entry = obj("sent" => sent)
    push!(t.log, entry)
    return LM15.open_response(t.inner, request) do head, io
        entry["status"] = head.status
        f(head, io)
    end
end

short(s, n=100) = length(s) > n ? first(s, n) * "…" : s
err_text(e) = e isa LM15Error ? "$(LM15.class_name(e)) ($(e.code)): $(e.message)" : sprint(showerror, e)
response_value(r) = r === nothing ? nothing : to_dict(r)

function hello(router, b, problems, notes)
    req = Request(b.model, user("Reply with exactly the two words: hello world"); config=Config(; max_tokens=b.tokens))
    done = complete(router, req)
    chunks = String[]
    streamed = stream(router, req) do rs
        foreach(c -> push!(chunks, c), text_chunks(rs))
        response(rs)
    end
    ct, st = something(text(done), ""), something(text(streamed), "")
    push!(notes, "complete $(repr(ct)) finish=$(done.finish_reason); stream $(repr(st)) finish=$(streamed.finish_reason) in $(length(chunks)) chunks")
    occursin("hello world", lowercase(ct)) && occursin("hello world", lowercase(st)) || push!(problems, "the reply is not the two words asked for")
    done.finish_reason == streamed.finish_reason || push!(problems, "finish_reason differs between complete and stream")
    join(chunks) == st || push!(problems, "the text chunks do not concatenate to the assembled text")
    streamed.usage.output_tokens === nothing && push!(notes, "the stream reported no output token count")
    return [response_value(done), response_value(streamed)]
end
const ORDER_SCHEMA = JSON.parse("""{"type": "object", "properties": {
    "reasoning": {"type": "string", "description": "Work the problem out step by step before answering."},
    "answer": {"type": "string", "description": "The final answer only."}},
    "required": ["reasoning", "answer"], "additionalProperties": false}""")
function order(router, b, problems, notes, log)
    format = obj("type" => "json_schema", "name" => "worked_answer", "strict" => true, "schema" => ORDER_SCHEMA)
    req = Request(b.model, user("A bat and a ball cost \$1.10 in total. The bat costs \$1.00 more than the ball. How much does the ball cost? Answer in JSON.");
        config=Config(; max_tokens=b.tokens, response_format=format))
    before = length(log)
    r = complete(router, req)
    for entry in log[(before + 1):end]
        body = JSON.serialize(entry["sent"]["body"])
        i, j = findfirst("\"reasoning\":{", body), findfirst("\"answer\":{", body)
        i !== nothing && j !== nothing && first(i) > first(j) && push!(problems, "the body sent lists answer before reasoning")
    end
    for a in r.adaptations
        push!(notes, "adapted $(a.field): $(a.action)")
        if a.field == "config.response_format" && a.action == "dropped"
            push!(notes, "the reply, free-form: $(repr(short(something(text(r), ""), 80)))")
            return [response_value(r)], "adapted"
        end
    end
    t = strip(something(text(r), ""))
    t = strip(replace(t, r"^```(json)?" => "", r"```$" => ""))
    value = try
        JSON.parse(t)
    catch
        nothing
    end
    if value isa AbstractDict
        push!(notes, "the model's JSON keys, in the order written: $(collect(keys(value))); answer $(repr(get(value, "answer", nothing)))")
        occursin("0.05", string(get(value, "answer", ""))) || occursin("5 cents", string(get(value, "answer", ""))) ||
            push!(notes, "the answer is not \$0.05")
    else
        push!(problems, "the reply is not a JSON object (finish=$(r.finish_reason)): $(repr(short(t, 120)))")
    end
    return [response_value(r)], nothing
end
function tools(router, b, problems, notes)
    params = JSON.parse("""{"type": "object", "properties": {
        "location": {"type": "string", "description": "City name"},
        "date": {"type": "string", "description": "YYYY-MM-DD"}},
        "required": ["location", "date"], "additionalProperties": false}""")
    tool = FunctionTool(; name="get_forecast", description="Weather forecast for a city on a date.", parameters=params)
    req = Request(b.model, user("Use the tool: what is the forecast for Montreal on 2026-10-01? Then answer in one sentence.");
        tools=(tool,), config=Config(; max_tokens=b.tokens))
    first_reply = complete(router, req)
    calls = tool_calls(first_reply)
    if isempty(calls)
        push!(problems, "no tool call (finish=$(first_reply.finish_reason), text $(repr(short(something(text(first_reply), ""), 80))))")
        return [response_value(first_reply)]
    end
    call = first(calls)
    push!(notes, "call $(call.name)($(JSON.serialize(call.input)))")
    call.name == "get_forecast" && occursin("montr", lowercase(string(get(call.input, "location", "")))) ||
        push!(problems, "the call is not get_forecast for Montreal")
    second = complete(router, Request(req; messages=(req.messages..., first_reply.message,
        tool_message(call, "{\"forecast\": \"sunny\", \"high_c\": 14}"))))
    final = something(text(second), "")
    push!(notes, "final $(repr(short(final)))")
    occursin("sunny", lowercase(final)) || occursin("14", final) || push!(problems, "the final answer does not use the tool result")
    return [response_value(first_reply), response_value(second)]
end
function models(router, b, problems, notes)
    list = list_models(lm(router, b.model))
    push!(notes, "$(length(list)) models")
    isempty(list) && push!(problems, "an empty catalog")
    return [obj("ids" => [m.id for m in list])]
end
function judgment_check(router, b, problems, notes)
    format = judgments("style" => choice("Dominant style?", ["fruit", "oak", "mineral"]),
        "quality" => score("How good is this wine?", ["poor" => nothing, "fair" => nothing, "great" => nothing]),
        "ageing" => yes_no("Will it improve with age?"); name="wine")
    r = complete(router, Request(b.model, user("Tasting note: deep ruby, blackcurrant and cedar, firm tannins, long finish, 2019 Pauillac.");
        config=Config(; response_format=format, probabilities="required")))
    probs = something(probabilities(r), Dict())
    for name in ("style", "quality", "ageing")
        dist = get(probs, name, Dict())
        total = isempty(dist) ? 0.0 : sum(values(dist))
        isempty(dist) || abs(total - 1) > 0.02 ? push!(problems, "$name: no distribution summing to 1") :
            push!(notes, "$name: $(data(r)[name]) $(Dict(k => round(v; digits=3) for (k, v) in dist))")
    end
    data_part(r).method == "provider_classification" || push!(problems, "the method is not provider_classification")
    return [response_value(r)]
end

function main(args)
    out = isempty(args) || startswith(args[1], "--") ? "receipts/live-smoke" : args[1]
    only = let i = findfirst(==("--only"), args)
        i === nothing ? nothing : split(args[i + 1], ',')
    end
    mkpath(out)
    log = Any[]
    recorder = Recorder(LM15.HTTPTransport(), log)
    plain = LMRouter(RouterConfig(; transport=recorder))
    managed = "--managed" in args ? LMRouter(RouterConfig(; transport=recorder, auth=local_auth(), env=Dict{String,String}())) : nothing
    bindings = vcat(BINDINGS, managed === nothing ? [] : MANAGED)
    results = Any[]
    for b in bindings
        only === nothing || b.name in only || continue
        isempty(b.key) || !isempty(get(ENV, b.key, "")) || (println("skip $(b.name): \$$(b.key) not set"); continue)
        router = isempty(b.key) ? managed : plain
        checks = b.name == "typesafe" ? ("judgments",) : ("hello", "order", "tools", "models")
        for check in checks
            problems, notes = String[], String[]
            before = length(log)
            verdict = nothing
            values = Any[]
            try
                if check == "hello"
                    values = hello(router, b, problems, notes)
                elseif check == "order"
                    values, verdict = order(router, b, problems, notes, log)
                elseif check == "tools"
                    values = tools(router, b, problems, notes)
                elseif check == "models"
                    values = models(router, b, problems, notes)
                else
                    values = judgment_check(router, b, problems, notes)
                end
            catch e
                e isa InterruptException && rethrow()
                reason = get(PROVIDER_REFUSES, (b.name, check), nothing)
                if reason !== nothing && e isa LM15Error
                    verdict = "provider-refuses"
                    push!(notes, "$reason: $(err_text(e))")
                elseif e isa UnsupportedFeatureError && length(log) == before
                    verdict = "refused"
                    push!(notes, "refused before the wire: $(err_text(e))")
                elseif e isa AuthOperationError
                    verdict = "signed-out"
                    push!(notes, err_text(e))
                else
                    push!(problems, err_text(e))
                end
            end
            verdict = something(verdict, isempty(problems) ? "ok" : "problem")
            result = obj("binding" => b.name, "model" => b.model, "check" => check, "verdict" => verdict,
                "notes" => notes, "problems" => problems)
            push!(results, result)
            write(joinpath(out, "$(b.name)-$check.json"),
                JSON.serialize(obj("result" => result, "exchanges" => log[(before + 1):end], "lm15" => values)))
            println(rpad("$(b.name) $check", 34), verdict, isempty(problems) ? "" : "  " * join(problems, "; "))
        end
    end
    counts = Dict{String,Int}()
    foreach(r -> counts[r["verdict"]] = get(counts, r["verdict"], 0) + 1, results)
    open(joinpath(out, "SUMMARY.md"), "w") do io
        println(io, "# Live smoke, LM15.jl $(pkgversion(LM15))\n")
        println(io, "$(length(results)) checks: ", join(("$v $k" for (k, v) in sort!(collect(counts))), ", "), ".\n")
        println(io, "| Binding | Model | Check | Verdict | Notes |\n|---|---|---|---|---|")
        for r in results
            detail = join(vcat(r["problems"], r["notes"]), "; ")
            println(io, "| $(r["binding"]) | `$(r["model"])` | $(r["check"]) | $(r["verdict"]) | $(replace(short(detail, 240), "|" => "\\|", "\n" => " ")) |")
        end
    end
    println("\n", join(("$v $k" for (k, v) in sort!(collect(counts))), ", "))
end

main(ARGS)
