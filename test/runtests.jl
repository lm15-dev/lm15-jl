# Unit tests plus the lm15-contract auth-resolution fixture runner
# (auth/resolution.json, spec/auth.md AUTH-1/AUTH-7, ratified 2026-08-31).
# Divergence between this port and the fixtures is a port bug, never a
# reason to edit the fixture (AUTHORITY.md).

using Test
using LM15
using LM15: JSON

const SENTINEL = "SECRET-SENTINEL-DO-NOT-PRINT"

fake_codex_jwt(account_id, exp_seconds) = begin
    encode(value) = replace(
        Base64.base64encode(JSON.serialize(value)),
        '+' => '-', '/' => '_', '=' => "",
    )
    header = encode(Dict("alg" => "none", "typ" => "JWT"))
    payload = encode(Dict(
        "exp" => exp_seconds,
        "https://api.openai.com/auth" => Dict("chatgpt_account_id" => account_id),
    ))
    "$header.$payload.signature"
end

using Base64

@testset "credential providers (AUTH-2)" begin
    calls = Ref(0)
    provider = () -> "token-$(calls[] += 1)"
    @test resolve_credential(provider) != resolve_credential(provider)
    @test resolve_credential("static") == "static"
    credential = StaticCredential(SENTINEL)
    @test !occursin(SENTINEL, sprint(show, credential))
    @test !occursin(SENTINEL, sprint(show, MIME("text/plain"), credential))
    @test resolve_credential(credential) == SENTINEL
end

@testset "aliases and errors" begin
    @test explain_auth("openai_chat"; env = Dict{String,String}()).provider == "openai-chat"
    error_text = try
        explain_auth("not-a-provider")
        ""
    catch error
        @test error isa UnknownProviderError
        sprint(showerror, error)
    end
    @test occursin("anthropic", error_text)
end

@testset "gemini env key order: first declared wins" begin
    report = explain_auth("gemini"; env = Dict("GEMINI_API_KEY" => "a", "GOOGLE_API_KEY" => "b"))
    @test selected_step(report).kind == "env:GEMINI_API_KEY"
    @test report.steps[3].kind == "env:GOOGLE_API_KEY"
    @test report.steps[3].state === :shadowed
end

@testset "codex reader: JWT expiry with skew, account id" begin
    dir = mktempdir()
    path = joinpath(dir, "auth.json")
    fresh = fake_codex_jwt("acct_test", round(Int, time()) + 3600)
    write(path, JSON.serialize(Dict(
        "auth_mode" => "chatgpt",
        "tokens" => Dict("access_token" => fresh, "refresh_token" => "rt"),
    )))
    credential = read_codex_cli_credential(path)
    @test !is_expired(credential)
    @test account_id(credential) == "acct_test"
    @test has_refresh_token(credential)

    near_expiry = fake_codex_jwt("acct", round(Int, time()) + 120)  # inside 5min skew
    write(path, JSON.serialize(Dict("tokens" => Dict("access_token" => near_expiry))))
    @test is_expired(read_codex_cli_credential(path))
end

@testset "missing files are typed NotConfiguredError with a re-login hint" begin
    missing_path = joinpath(mktempdir(), "nope.json")
    for reader in (read_claude_code_credential, read_codex_cli_credential)
        thrown = try
            reader(missing_path)
            nothing
        catch error
            error
        end
        @test thrown isa NotConfiguredError
        @test occursin("Log in again", sprint(showerror, thrown))
    end
end

@testset "credential renderings never contain token material (AUTH-5)" begin
    dir = mktempdir()
    path = joinpath(dir, "credentials.json")
    write(path, JSON.serialize(Dict("claudeAiOauth" => Dict(
        "accessToken" => SENTINEL,
        "refreshToken" => SENTINEL,
        "expiresAt" => round(Int64, time() * 1000) + 60_000,
    ))))
    credential = read_claude_code_credential(path)
    @test !occursin(SENTINEL, sprint(show, credential))
    @test !occursin(SENTINEL, sprint(show, MIME("text/plain"), credential))
    @test access_token(credential) == SENTINEL  # accessor returns the real token
end

# ─── contract fixture runner ─────────────────────────────────────────

function materialize_borrowed_file(state, sentinel)
    dir = mktempdir()
    state == "missing" && return joinpath(dir, "does-not-exist.json")
    oauth = Dict{String,Any}("accessToken" => sentinel)
    if state == "fresh"
        oauth["expiresAt"] = round(Int64, time() * 1000) + 3_600_000
        oauth["refreshToken"] = sentinel
    elseif state == "expired-with-refresh"
        oauth["expiresAt"] = 1
        oauth["refreshToken"] = sentinel
    elseif state == "expired-no-refresh"
        oauth["expiresAt"] = 1
    else
        error("unknown borrowed_file state $state")
    end
    path = joinpath(dir, "credentials.json")
    write(path, JSON.serialize(Dict("claudeAiOauth" => oauth)))
    return path
end

@testset "auth resolution contract" begin
    fixture = JSON.parse(read(joinpath(@__DIR__, "..", "conformance", "auth_resolution.json"), String))
    sentinel = fixture["sentinel"]
    cases = fixture["cases"]
    @test !isempty(cases)

    for fixture_case in cases
        @testset "$(fixture_case["id"])" begin
            kwargs = Dict{Symbol,Any}(:env => Dict{String,String}())
            for (key, value) in get(fixture_case, "env", Dict())
                kwargs[:env][key] = value
            end
            api_key_providers = get(fixture_case, "api_keys_providers", nothing)
            api_key_providers !== nothing && (kwargs[:api_key_providers] = String.(api_key_providers))
            borrowed = get(fixture_case, "borrowed_file", nothing)
            if borrowed !== nothing
                @test fixture_case["provider"] == "claude-code"
                kwargs[:claude_credentials_path] =
                    materialize_borrowed_file(borrowed["state"], sentinel)
            end

            report = explain_auth(fixture_case["provider"]; kwargs...)

            expect = fixture_case["expect"]
            @test report.configured == expect["configured"]
            @test length(report.steps) == length(expect["steps"])
            for (index, expected) in enumerate(expect["steps"])
                @test report.steps[index].kind == expected["kind"]
                @test String(report.steps[index].state) == expected["state"]
            end

            # AUTH-5: no rendering may carry the planted sentinel.
            for rendering in (describe(report), sprint(show, report), sprint(show, MIME("text/plain"), report))
                @test !occursin(sentinel, rendering)
            end
        end
    end
end
