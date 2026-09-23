# Regression specifications for the implementation review. These have not been
# executed in the implementation-only phase.

@testset "canonical constructors and Julia conveniences" begin
    message = user("Look at this", image(url="https://example.org/picture.png"))
    req = Request("openai:model", message)
    @test req.messages == (message,)
    @test length(req.messages[1].parts) == 2
    @test req.messages[1].parts[2] isa ImagePart

    changed = Request(req; config=Config(req.config; max_tokens=100))
    @test req.config.max_tokens === nothing
    @test changed.config.max_tokens == 100
    @test changed.messages === req.messages
    @test_throws ArgumentError Request(req; misspelled_setting=true)
    @test_throws ArgumentError Config(req.config; max_tokens=true)

    call = tool_call("call-1", "weather", Dict("city" => "Paris"))
    result = tool_message(call, "Sunny")
    @test only(result.parts).id == call.id
    @test only(result.parts).name == call.name
    @test only(only(result.parts).content).text == "Sunny"
    @test ToolCallInfo(call).input === call.input

    # Factory construction must not read this nonexistent local file.
    media = image(path=joinpath("does-not-exist", "photo.JPEG"))
    @test media.media_type == "image/jpeg"
    @test_throws ArgumentError image(url="x", data="AA==")
end

@testset "numeric logprobs and strict caller-authored nests" begin
    config = from_json(Config, "{\"temperature\":1,\"logprobs\":0}")
    @test config.temperature === 1.0
    @test config.logprobs === 0
    detail = ErrorDetail(code="unsupported_model", message="no such model")
    @test from_json(ErrorDetail, to_json(detail)).message == detail.message
    event = StreamErrorEvent(error=detail)
    @test from_json(StreamEvent, to_json(event)).error.message == detail.message
    cached = CacheConfig(resource="cachedContents/example", prefix_until_index=0)
    @test from_json(CacheConfig, to_json(cached)).resource == cached.resource
    @test from_json(Config, to_json(Config(cache=cached))).cache.resource == cached.resource
    @test occursin("1.0", to_json(config))
    @test from_json(Config, to_json(config)).logprobs === 0
    @test_throws ArgumentError from_json(Config, "{\"logprobs\":true}")
    @test_throws ArgumentError from_json(Config, "{\"tool_choice\":3}")
    @test_throws ArgumentError from_json(
        Request,
        """
{"model":"m","messages":[{"role":"user","parts":[{"type":"text","text":"hi"}]}],"config":null}
""",
    )
    answer = from_json(
        Response,
        """
{"model":"m","message":{"role":"assistant","parts":[{"type":"text","text":"hi"}]},"finish_reason":"stop","usage":"unreported"}
""",
    )
    @test answer.usage.input_tokens === nothing
end

struct RotatingCredential
    count::Base.RefValue{Int}
end
(c::RotatingCredential)() = (c.count[] += 1; BearerToken("token-$(c.count[])"))

@testset "credential callbacks run once and preserve error families" begin
    provider = RotatingCredential(Ref(0))
    client = OpenAILM(api_key=provider)
    @test provider.count[] == 0
    build_request(client, Request("m", user("hi")))
    @test provider.count[] == 1
    @test_throws ArgumentError resolve_credential(() -> (() -> "not-a-credential-value"))
    @test ContextLengthError("long") isa InvalidRequestError
    @test InvalidRequestError("bad").code == "invalid_request"
    @test DeviceCodeExpiredError("xai") isa AuthError
    @test DeviceCodeExpiredError("xai") isa LM15Error
    @test DeviceCodeExpiredError("xai").code == "auth"
end

@testset "access policies and named presets select the right door" begin
    code = AnthropicLM(access=LM15.provider_definition("claude-code").access, api_key="explicit")
    @test code.provider == "claude-code"
    @test code.access.backend == "claude-code"
    @test_throws ArgumentError OpenAILM(access=code.access, api_key="explicit")
    local_client = OpenAIChatLM(compat="lm-studio", api_key="local")
    @test local_client.base_url == "http://localhost:1234/v1"
    @test_throws NotConfiguredError OpenAIChatLM(compat="qwen", api_key="explicit")
    @test_throws NotConfiguredError OpenAIChatLM(compat="ollama")

    custom = GeminiLM(api_key="explicit", base_url="https://gateway.example/v1beta")
    @test custom.upload_base_url === nothing
    @test_throws NotConfiguredError build_file_request(
        custom, :upload; request=FileUploadRequest(filename="x", bytes_data=UInt8[1])
    )

    invalid = OpenAIChatCompat(model_overrides=(("m", Dict("forced_tool_choice" => "typo")),))
    @test_throws ArgumentError OpenAIChatLM(api_key="explicit", compat=invalid)
end

@testset "implicit caching never discards a stored-resource or affinity-key intent" begin
    for (provider, model) in (
        ("meta-anthropic", "muse-spark"),
        ("deepseek-anthropic", "deepseek-v4-flash"),
        ("moonshotai-anthropic", "kimi-k3"),
    )
        client = ProviderLM(provider; api_key="explicit")
        for cache in (CacheConfig(resource="cachedContents/opaque"), CacheConfig(key="affinity"))
            req = Request(model, user("suffix"); config=Config(; cache))
            @test_throws UnsupportedFeatureError build_request(client, req)
        end
    end
end

@testset "displays keep raw payloads out of the notebook summary" begin
    part = image(data=UInt8[0, 1, 2, 3])
    @test !occursin(part.data, sprint(show, part))
    answer = Response(
        model="m",
        message=assistant("hello"),
        finish_reason="stop",
        provider_data=Dict("private" => SENTINEL),
    )
    @test occursin("hello", sprint(show, MIME"text/plain"(), answer))
    @test !occursin(SENTINEL, sprint(show, MIME"text/plain"(), answer))
end

@testset "Codex refuses an output cap or store=true, never strips it (MAP-13 rule 4)" begin
    lm = OpenAICodexLM(api_key="tok", account_id="acct", env=Dict{String,String}())
    @test !occursin("max_output_tokens", String(copy(build_request(lm, Request("gpt-5.5", user("hi"))).body)))
    for (config, field) in ((Config(max_tokens=5), "config.max_tokens"), (Config(store=true), "config.store"))
        err = try
            build_request(lm, Request("gpt-5.5", user("hi"); config))
            nothing
        catch e
            e
        end
        @test err isa UnsupportedFeatureError
        @test occursin("openai-codex: $field", sprint(showerror, err))
    end
end
