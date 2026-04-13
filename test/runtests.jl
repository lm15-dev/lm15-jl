using Test
using LM15

@testset "LM15 Tests" begin

@testset "JSON parser" begin
    @test LM15.JSON.parse("""{"name":"Alice","age":30}""") == Dict("name"=>"Alice","age"=>30)
    @test LM15.JSON.parse("[1,2,3]") == [1, 2, 3]
    @test LM15.JSON.parse("\"hello\"") == "hello"
    @test LM15.JSON.parse("true") == true
    @test LM15.JSON.parse("null") === nothing
    @test LM15.JSON.parse("3.14") ≈ 3.14

    # Round-trip
    d = Dict("a" => 1, "b" => [2, 3], "c" => nothing)
    @test LM15.JSON.parse(LM15.JSON.serialize(d))["a"] == 1
end

@testset "Types" begin
    p = TextPart("hello")
    @test p.type == "text"
    @test p.text == "hello"

    p = ThinkingPart("hmm")
    @test p.type == "thinking"
    @test p.text == "hmm"

    p = ToolCallPart("c1", "weather", Dict{String,Any}("city" => "Paris"))
    @test p.type == "tool_call"
    @test p.id == "c1"
    @test p.name == "weather"
    @test p.input["city"] == "Paris"

    p = ImageURL("https://example.com/img.png")
    @test p.type == "image"
    @test p.source.url == "https://example.com/img.png"

    m = UserMessage("hello")
    @test m.role == "user"
    @test length(m.parts) == 1
    @test m.parts[1].text == "hello"
end

@testset "DataSource" begin
    ds = DataSource(type="base64", data="AQID", media_type="application/octet-stream")
    @test decode_bytes(ds) == UInt8[1, 2, 3]

    ds2 = DataSource(type="url", url="https://example.com")
    @test_throws ErrorException decode_bytes(ds2)
end

@testset "LMResponse accessors" begin
    resp = LMResponse("r1", "test",
        Message(role="assistant", parts=[TextPart("Hello"), TextPart("World")]),
        "stop", Usage(), nothing)
    @test text(resp) == "Hello\nWorld"

    resp2 = LMResponse("r2", "test",
        Message(role="assistant", parts=[
            TextPart("thinking..."),
            ToolCallPart("c1", "weather", Dict{String,Any}("city"=>"Paris")),
        ]),
        "tool_call", Usage(), nothing)
    @test length(tool_calls(resp2)) == 1
    @test tool_calls(resp2)[1].name == "weather"

    resp3 = LMResponse("r3", "test",
        Message(role="assistant", parts=[ThinkingPart("let me think"), TextPart("answer")]),
        "stop", Usage(), nothing)
    @test thinking(resp3) == "let me think"
end

@testset "Capabilities" begin
    @test resolve_provider("claude-sonnet-4-5") == "anthropic"
    @test resolve_provider("gpt-4.1-mini") == "openai"
    @test resolve_provider("gemini-2.5-flash") == "gemini"
    @test resolve_provider("o1-preview") == "openai"
    @test resolve_provider("o3-mini") == "openai"
    @test_throws UnsupportedModelError resolve_provider("llama-3")
end

@testset "Errors" begin
    @test LM15.map_http_error(401, "bad") isa AuthError
    @test LM15.map_http_error(402, "no") isa BillingError
    @test LM15.map_http_error(429, "slow") isa RateLimitError
    @test LM15.map_http_error(500, "oops") isa ServerError
    @test LM15.map_http_error(400, "bad") isa InvalidRequestError
    @test LM15.map_http_error(408, "timeout") isa TimeoutError

    @test LM15.canonical_error_code(AuthError("")) == "auth"
    @test LM15.canonical_error_code(RateLimitError("")) == "rate_limit"
    @test LM15.canonical_error_code(ContextLengthError("")) == "context_length"
    @test LM15.canonical_error_code(ServerError("")) == "server"

    @test LM15.is_transient(RateLimitError(""))
    @test LM15.is_transient(ServerError(""))
    @test !LM15.is_transient(AuthError(""))
end

@testset "Conversation" begin
    conv = Conversation(system="test")
    user!(conv, "hello")
    user!(conv, "world")
    @test length(conv.messages) == 2
    @test conv.system == "test"

    clear!(conv)
    @test length(conv.messages) == 0
end

@testset "Cost estimation" begin
    u = Usage(input_tokens=1000, output_tokens=500, total_tokens=1500)
    rates = Dict("input"=>3.0, "output"=>15.0)
    cost = estimate_cost(u, rates, "openai")
    @test cost.total > 0
    @test cost.input ≈ 1000 * 3.0 / 1_000_000
    @test cost.output ≈ 500 * 15.0 / 1_000_000
end

@testset "Providers" begin
    p = providers()
    @test haskey(p, "openai")
    @test haskey(p, "anthropic")
    @test haskey(p, "gemini")
    @test "OPENAI_API_KEY" in p["openai"]
end

@testset "Result" begin
    function make_stream(text)
        (req) -> [
            StreamEvent(type="start", id="r1", model="test"),
            StreamEvent(type="delta", part_index=0, delta=PartDelta("text", text, nothing, nothing)),
            StreamEvent(type="end", finish_reason="stop", usage=Usage()),
        ]
    end

    r = LMResult(
        request=LMRequest(model="test", messages=[UserMessage("hi")]),
        start_stream=make_stream("Hello!"),
    )
    @test text(r) == "Hello!"
    @test finish_reason(r) == "stop"

    # Tool auto-execution
    call_count = Ref(0)
    tool_stream = (req) -> begin
        call_count[] += 1
        if call_count[] == 1
            [StreamEvent(type="start", id="r1", model="test"),
             StreamEvent(type="delta", part_index=0,
                delta_raw=Dict{String,Any}("type"=>"tool_call","id"=>"c1","name"=>"greet","input"=>"{}")),
             StreamEvent(type="end", finish_reason="tool_call", usage=Usage())]
        else
            [StreamEvent(type="start", id="r2", model="test"),
             StreamEvent(type="delta", part_index=0, delta=PartDelta("text", "done", nothing, nothing)),
             StreamEvent(type="end", finish_reason="stop", usage=Usage())]
        end
    end

    r2 = LMResult(
        request=LMRequest(model="test", messages=[UserMessage("hi")],
            tools=[FunctionTool("greet", "Say hi")]),
        start_stream=tool_stream,
        callable_registry=Dict{String,Function}("greet" => args -> "Hello!"),
        max_tool_rounds=2,
    )
    @test text(r2) == "done"
    @test call_count[] == 2
end

@testset "Client" begin
    lm = UniversalLM()
    @test_throws LM15Error LM15.complete(lm, LMRequest(model="test", messages=[UserMessage("hi")]))
end

@testset "Additional accessors" begin
    resp = LMResponse("r1", "test",
        Message(role="assistant", parts=[TextPart("{\"name\":\"Alice\",\"age\":30}")]),
        "stop", Usage(), nothing)
    j = json(resp)
    @test j["name"] == "Alice"
    @test j["age"] == 30

    resp2 = LMResponse("r2", "test",
        Message(role="assistant", parts=[
            TextPart("text"),
            CitationPart(text="src", url="https://x.com", title="X"),
        ]),
        "stop", Usage(), nothing)
    @test length(citations(resp2)) == 1
    @test citations(resp2)[1].url == "https://x.com"
end

@testset "Model copy" begin
    lm = UniversalLM()
    m = Model(lm, "test", system="original")
    m2 = LM15.copy(m, system="override")
    @test m2.system == "override"
    @test m.system == "original"
    @test m2.model_name == "test"
end

@testset "Providers info" begin
    p = providers()
    @test haskey(p, "openai")
    @test "OPENAI_API_KEY" in p["openai"]
end

end # main testset
