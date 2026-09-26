# Reply faults, diagnostics and routing details the contract harness cannot drive:
# INV-053/054/055, error diagnostics (docs/error-diagnostics.md), the prefix strip at
# the codec boundary, and managed sign-in basics on a memory store.
using CodecZlib: transcode, GzipCompressor, ZlibCompressor, DeflateCompressor

reply(body; status=200, headers=Pair{String,String}[]) =
    HttpResponse(; status, headers, body=Vector{UInt8}(codeunits(body)))
const CHAT_OK = """{"id":"c","model":"m","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"hi"}}]}"""

@testset "INV-053: compressed replies are inflated or refused by name" begin
    req = Request("m", user("x"))
    for (coding, codec) in (("gzip", GzipCompressor), ("x-gzip", GzipCompressor), ("deflate", ZlibCompressor), ("deflate", DeflateCompressor))
        packed = transcode(codec, Vector{UInt8}(codeunits(CHAT_OK)))
        lm = OpenAIChatLM(; api_key="k", transport=_ -> HttpResponse(; status=200,
            headers=["Content-Encoding" => coding], body=packed))
        @test text(complete(lm, req)) == "hi"
    end
    for coding in ("br", "zstd")
        lm = OpenAIChatLM(; api_key="k", transport=_ -> HttpResponse(; status=200,
            headers=["Content-Encoding" => coding], body=UInt8[1, 2, 3]))
        err = try
            complete(lm, req); nothing
        catch e
            e
        end
        @test err isa TransportError && occursin(coding, err.message)
    end
end

@testset "INV-054: a non-JSON success is a provider fault" begin
    lm = OpenAIChatLM(; api_key="k", transport=_ -> reply("<html>gateway</html>";
        headers=["Content-Type" => "text/html", "x-request-id" => "r-1"]))
    err = try
        complete(lm, Request("m", user("x"))); nothing
    catch e
        e
    end
    @test err isa ProviderError && !(err isa ServerError) && !retryable(err)
    @test err.status == 200 && err.request_id == "r-1" && occursin("text/html", err.message)
end

@testset "INV-055: text with no UTF-8 form is refused before the wire" begin
    sent = Ref(false)
    lm = OpenAIChatLM(; api_key="k", transport=_ -> (sent[] = true; reply(CHAT_OK)))
    lone = String(UInt8[0x61, 0xed, 0xa0, 0xbd])  # a lone high surrogate
    @test_throws ArgumentError complete(lm, Request("m", user(lone)))
    @test !sent[]
end

@testset "error diagnostics: rate-limit headers, retry hints, request ids" begin
    headers = ["x-ratelimit-remaining-requests" => "0", "x-ratelimit-remaining-requests" => "-1",
        "retry-after-ms" => "1500", "apim-request-id" => "apim-1", "authorization" => "Bearer SECRET"]
    lm = OpenAIChatLM(; api_key="k", transport=_ -> reply("""{"error":{"message":"slow down","code":"rate_limit_exceeded"}}""";
        status=429, headers))
    err = try
        complete(lm, Request("m", user("x"))); nothing
    catch e
        e
    end
    @test err isa RateLimitError
    @test err.retry_after == 1.5
    @test err.request_id == "apim-1"
    @test err.rate_limit_headers["x-ratelimit-remaining-requests"] == ("0", "-1")
    @test !haskey(err.rate_limit_headers, "authorization")
    shown = sprint(showerror, err)
    @test occursin("Retry advice: 1.5 seconds", shown) && !occursin("SECRET", shown)
end

@testset "a client strips exactly its own provider prefix" begin
    lm = GeminiLM(; api_key="k")
    @test occursin("/models/gemini-2.5-flash:", build_request(lm, Request("gemini:gemini-2.5-flash", user("x"))).url)
    local_lm = OpenAIChatLM(; compat="ollama", api_key="local")
    body = JSON.parse(String(copy(build_request(local_lm, Request("qwen3.5:0.8b", user("x"))).body)))
    @test body["model"] == "qwen3.5:0.8b"
end

@testset "managed sign-in on a memory store" begin
    auth = memory_auth()
    @test isempty(connections(auth))
    c = set_api_key(auth, "openai", SENTINEL)
    @test c.kind == "api_key" && c.identity_generation == "1"
    @test !occursin(SENTINEL, sprint(show, c))
    @test request_auth(auth, "openai").credential.value == SENTINEL
    err = try
        set_api_key(auth, "openai", "other"); nothing
    catch e
        e
    end
    @test err isa AuthOperationError && err.reason == "connection_exists"
    selection = ModelSelection("openai", "gpt-5-mini", c.id, c.identity_generation)
    bound = BoundClient(auth, selection)
    @test request(bound, "hi").model == "openai:gpt-5-mini"
    replaced = set_api_key(auth, "openai", "second"; replace=c.id)
    @test replaced.id != c.id && replaced.identity_generation == "2"
    pinned = try
        request_auth(auth, "openai"; pinned=(c.id, c.identity_generation)); nothing
    catch e
        e
    end
    @test pinned isa AuthOperationError && pinned.reason == "connection_changed"
    forgotten = logout(auth, "openai")
    @test forgotten.forgot && status(auth, "openai").logged_out
    # Logout blocks the ambient key: a managed router does not fall back to it.
    router = LMRouter(RouterConfig(; auth, env=Dict("OPENAI_API_KEY" => SENTINEL)))
    blocked = try
        lm(router, "openai:gpt-5-mini"); nothing
    catch e
        e
    end
    @test blocked isa AuthOperationError && blocked.reason == "login_required"
    @test logout(auth, c.id).forgot == false  # an old id never removes anything
    @test "radius" in [d.id for d in login_providers()]
    @test all(m -> m.availability == "unavailable", login_methods("radius"))
end

@testset "managed store refuses what it cannot read, and never overwrites it" begin
    mktempdir() do dir
        path = joinpath(dir, "credentials.json")
        write(path, """{"xai": {"type": "oauth"}, "xai": {"type": "oauth"}}""")
        auth = local_auth(path)
        err = try
            connections(auth); nothing
        catch e
            e
        end
        @test err isa AuthOperationError && err.reason == "storage_unavailable"
        write(path, """{"_lm15": {"version": 2, "slots": {}}}""")
        err = try
            set_api_key(auth, "openai", "k"); nothing
        catch e
            e
        end
        @test err isa AuthOperationError && err.reason == "unsupported_store_version"
        @test read(path, String) == """{"_lm15": {"version": 2, "slots": {}}}"""
    end
end
