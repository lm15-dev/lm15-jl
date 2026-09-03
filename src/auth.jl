# lm15-contract auth surface (spec/auth.md, ratified 2026-08-31):
# credential providers (AUTH-2), the resolution chain (AUTH-1), the explain
# report (AUTH-7), and read-side borrowed CLI credentials (AUTH-8).
#
# Secrecy invariant (AUTH-5): no secret value is stored on an AuthReport,
# rendered by describe, or printed by any show method in this file.
#
# Not yet implemented in this port (stated, not absorbed): the AUTH-3/4
# write side (locked double-checked refresh, atomic 0600 writes) and the
# AUTH-9 login primitives. This port currently reads credentials only.

using Base64: base64decode

# A token inside this window counts as expired (AUTH-3 skew).
const REFRESH_SKEW_MS = 5 * 60 * 1000

# ─── credential providers (AUTH-2) ───────────────────────────────────

"""
A credential is a static key string or a zero-argument provider callable.
Adapters resolve it once per request at request-build time and never cache
the returned value; caching belongs to the provider itself.
"""
const Credential = Union{AbstractString,Function}

"""Resolve a credential value, invoking a provider callable."""
resolve_credential(credential::AbstractString) = String(credential)
resolve_credential(credential::Function) = String(credential())

"""A fixed credential whose `show` is redacted (AUTH-5)."""
struct StaticCredential
    value::String
end

resolve_credential(credential::StaticCredential) = credential.value
Base.show(io::IO, ::StaticCredential) = print(io, "StaticCredential(redacted)")
Base.show(io::IO, ::MIME"text/plain", ::StaticCredential) = print(io, "StaticCredential(redacted)")

# ─── errors (AUTH-6) ─────────────────────────────────────────────────

"""No usable credential source. The hint names the fix."""
struct NotConfiguredError <: Exception
    provider::String
    message::String
    credential_hint::String
end

function Base.showerror(io::IO, error::NotConfiguredError)
    print(io, error.provider, ": ", error.message, "\n\n  To fix:\n    - ", error.credential_hint, "\n")
end

struct UnknownProviderError <: Exception
    provider::String
end

function Base.showerror(io::IO, error::UnknownProviderError)
    print(io, "Unknown provider ", repr(error.provider),
        ". Known providers: ", join(known_providers(), ", "))
end

# ─── provider table (mirrors the reference router) ───────────────────

struct ProviderSpec
    env_keys::Vector{String}
    default_key::Union{String,Nothing}
    oauth_file::Union{String,Nothing}
end

spec(; env_keys = String[], default_key = nothing, oauth_file = nothing) =
    ProviderSpec(env_keys, default_key, oauth_file)

const PROVIDERS = Dict{String,ProviderSpec}(
    "openai" => spec(env_keys = ["OPENAI_API_KEY"]),
    "openai-chat" => spec(env_keys = ["OPENAI_API_KEY"]),
    "anthropic" => spec(env_keys = ["ANTHROPIC_API_KEY"]),
    "gemini" => spec(env_keys = ["GEMINI_API_KEY", "GOOGLE_API_KEY"]),
    "groq" => spec(env_keys = ["GROQ_API_KEY"]),
    "openrouter" => spec(env_keys = ["OPENROUTER_API_KEY"]),
    "deepseek" => spec(env_keys = ["DEEPSEEK_API_KEY"]),
    "zai" => spec(env_keys = ["ZAI_API_KEY"]),
    "ollama" => spec(default_key = "ollama"),
    "vllm" => spec(default_key = "EMPTY"),
    "sglang" => spec(default_key = "EMPTY"),
    "claude-code" => spec(oauth_file = "claude-code"),
    "openai-codex" => spec(oauth_file = "openai-codex"),
)

"""Map the permanent underscore alias to the hyphenated provider string."""
canonical_provider(name::AbstractString) = replace(String(name), "_" => "-")

"""Every provider in the built-in table, sorted."""
known_providers() = sort!(collect(keys(PROVIDERS)))

# ─── explain report (AUTH-7) ─────────────────────────────────────────

# One rung of the chain. `kind` uses the contract vocabulary ("api_keys",
# "env:<KEY>", "placeholder", "oauth-file"); `state` is one of :selected
# (this rung supplies the credential), :shadowed (usable, but an earlier
# rung wins), :absent. `detail` carries no secret material by construction.
struct AuthStep
    kind::String
    detail::String
    state::Symbol
end

struct AuthReport
    provider::String
    steps::Vector{AuthStep}
    configured::Bool
end

"""The winning step, or `nothing`."""
selected_step(report::AuthReport) =
    findfirst(step -> step.state === :selected, report.steps) isa Int ?
    report.steps[findfirst(step -> step.state === :selected, report.steps)] : nothing

const STEP_MARKERS = Dict(:selected => "=> ", :shadowed => " ~ ", :absent => " - ")

"""Render the rung-by-rung report. Never contains secret values."""
function describe(report::AuthReport)
    lines = ["auth for provider $(repr(report.provider)):"]
    for step in report.steps
        push!(lines, "  $(STEP_MARKERS[step.state])$(step.kind): $(step.detail)")
    end
    winner = selected_step(report)
    push!(lines, winner === nothing ? "  configured: no" : "  configured: yes — $(winner.kind)")
    return join(lines, "\n")
end

Base.show(io::IO, report::AuthReport) = print(io, describe(report))
Base.show(io::IO, ::MIME"text/plain", report::AuthReport) = print(io, describe(report))

"""
    explain_auth(provider; env=nothing, api_key_providers=String[],
                 claude_credentials_path=nothing, codex_auth_path=nothing)

Walk the AUTH-1 chain and report every rung (AUTH-7). No network I/O.
`env === nothing` means the process environment; pass a Dict for hermetic
tests. `api_key_providers` lists providers with explicit credentials —
presence only, values are never consulted. Env values are tested for
presence only and never retained — that presence check is the one stated
purity trade-off.
"""
function explain_auth(
    provider::AbstractString;
    env::Union{Nothing,AbstractDict} = nothing,
    api_key_providers::AbstractVector{<:AbstractString} = String[],
    claude_credentials_path::Union{Nothing,AbstractString} = nothing,
    codex_auth_path::Union{Nothing,AbstractString} = nothing,
)
    canonical = canonical_provider(provider)
    haskey(PROVIDERS, canonical) || throw(UnknownProviderError(String(provider)))
    provider_spec = PROVIDERS[canonical]

    if provider_spec.oauth_file !== nothing
        step = oauth_file_step(provider_spec.oauth_file, claude_credentials_path, codex_auth_path)
        return AuthReport(canonical, [step], step.state === :selected)
    end

    environment = env === nothing ? ENV : env
    steps = AuthStep[]
    selected = false

    if canonical in api_key_providers
        push!(steps, AuthStep("api_keys", "provided (value never shown)", :selected))
        selected = true
    else
        push!(steps, AuthStep("api_keys", "not provided", :absent))
    end

    for key in provider_spec.env_keys
        kind = "env:$key"
        if !isempty(get(environment, key, ""))
            push!(steps, AuthStep(kind, "set (value never shown)", selected ? :shadowed : :selected))
            selected = true
        else
            push!(steps, AuthStep(kind, "not set", :absent))
        end
    end

    if provider_spec.default_key !== nothing
        push!(steps, AuthStep(
            "placeholder",
            "preset default for keyless $canonical servers",
            selected ? :shadowed : :selected,
        ))
        selected = true
    end

    return AuthReport(canonical, steps, selected)
end

function oauth_file_step(oauth_provider, claude_path, codex_path)
    credential = if oauth_provider == "claude-code"
        path = claude_path === nothing ? default_claude_credentials_path() : String(claude_path)
        try_read(read_claude_code_credential, path)
    else
        path = codex_path === nothing ? default_codex_auth_path() : String(codex_path)
        try_read(read_codex_cli_credential, path)
    end
    credential === nothing && return AuthStep("oauth-file", "missing or unreadable", :absent)
    if is_expired(credential)
        has_refresh_token(credential) &&
            return AuthStep("oauth-file", "expired, refresh token present", :selected)
        return AuthStep("oauth-file", "expired, NO refresh token", :absent)
    end
    return AuthStep("oauth-file", "fresh", :selected)
end

function try_read(reader, path)
    try
        return reader(path)
    catch error
        error isa NotConfiguredError && return nothing
        rethrow()
    end
end

# ─── borrowed CLI credentials, read side (AUTH-8) ────────────────────

"""
A locally stored OAuth credential. `show` is redacted (AUTH-5); read the
token through `access_token(credential)`.
"""
struct LocalOAuthCredential
    access_token::String
    refresh_token::Union{String,Nothing}
    expires_at_ms::Union{Int64,Nothing}
    account_id::Union{String,Nothing}
end

access_token(credential::LocalOAuthCredential) = credential.access_token
has_refresh_token(credential::LocalOAuthCredential) = credential.refresh_token !== nothing
account_id(credential::LocalOAuthCredential) = credential.account_id
is_expired(credential::LocalOAuthCredential) =
    credential.expires_at_ms !== nothing &&
    round(Int64, time() * 1000) >= credential.expires_at_ms

function _redacted_show(io::IO, credential::LocalOAuthCredential)
    print(io, "LocalOAuthCredential(expires_at_ms=", credential.expires_at_ms,
        ", refresh=", has_refresh_token(credential), ", redacted)")
end
Base.show(io::IO, credential::LocalOAuthCredential) = _redacted_show(io, credential)
Base.show(io::IO, ::MIME"text/plain", credential::LocalOAuthCredential) = _redacted_show(io, credential)

"""`~/.claude/.credentials.json` (AUTH-8)."""
default_claude_credentials_path() = joinpath(homedir(), ".claude", ".credentials.json")

"""`~/.codex/auth.json` (AUTH-8)."""
default_codex_auth_path() = joinpath(homedir(), ".codex", "auth.json")

const CLAUDE_HINT = "Log in again: run `claude` and use /login (Claude subscription auth)"
const CODEX_HINT = "Log in again: run `codex login` (ChatGPT subscription auth)"

function read_json_object(path, provider, hint)
    text = try
        read(path, String)
    catch
        throw(NotConfiguredError(provider, "No readable credentials file at $path.", hint))
    end
    data = try
        JSON.parse(text)
    catch
        throw(NotConfiguredError(provider, "Credentials file at $path is not valid JSON.", hint))
    end
    data isa AbstractDict ||
        throw(NotConfiguredError(provider, "Credentials file at $path has an unexpected shape.", hint))
    return data
end

string_field(object, key) =
    (value = get(object, key, nothing); value isa AbstractString && !isempty(value) ? String(value) : nothing)

dict_field(object, key) =
    (value = get(object, key, nothing); value isa AbstractDict ? value : nothing)

int_field(object, key) =
    (value = get(object, key, nothing); value isa Real ? round(Int64, value) : nothing)

"""Read-only loader for the Claude Code CLI credential file."""
function read_claude_code_credential(path::AbstractString = default_claude_credentials_path())
    data = read_json_object(path, "claude-code", CLAUDE_HINT)
    oauth = dict_field(data, "claudeAiOauth")
    oauth === nothing &&
        throw(NotConfiguredError("claude-code", "Credentials file at $path has no claudeAiOauth section.", CLAUDE_HINT))
    access = string_field(oauth, "accessToken")
    access === nothing &&
        throw(NotConfiguredError("claude-code", "Credentials file at $path has no access token.", CLAUDE_HINT))
    return LocalOAuthCredential(
        access,
        string_field(oauth, "refreshToken"),
        int_field(oauth, "expiresAt"),
        nothing,
    )
end

"""
Read-only loader for the OpenAI Codex CLI auth file; expiry comes from the
access token's JWT `exp` claim minus the AUTH-3 skew.
"""
function read_codex_cli_credential(path::AbstractString = default_codex_auth_path())
    data = read_json_object(path, "openai-codex", CODEX_HINT)
    tokens = dict_field(data, "tokens")
    tokens === nothing &&
        throw(NotConfiguredError("openai-codex", "Credentials file at $path has no tokens section.", CODEX_HINT))
    access = string_field(tokens, "access_token")
    access === nothing &&
        throw(NotConfiguredError("openai-codex", "Credentials file at $path has no access token.", CODEX_HINT))
    payload = jwt_payload(access)
    exp_seconds = payload === nothing ? nothing : int_field(payload, "exp")
    auth_claim = payload === nothing ? nothing : dict_field(payload, "https://api.openai.com/auth")
    stored_account = string_field(tokens, "account_id")
    claim_account = auth_claim === nothing ? nothing : string_field(auth_claim, "chatgpt_account_id")
    return LocalOAuthCredential(
        access,
        string_field(tokens, "refresh_token"),
        exp_seconds === nothing ? nothing : exp_seconds * 1000 - REFRESH_SKEW_MS,
        stored_account === nothing ? claim_account : stored_account,
    )
end

function jwt_payload(token::AbstractString)
    parts = split(token, '.')
    length(parts) == 3 || return nothing
    segment = String(parts[2])
    padded = replace(segment, '-' => '+', '_' => '/')
    padded *= "="^mod(-length(padded), 4)
    decoded = try
        String(base64decode(padded))
    catch
        return nothing
    end
    payload = try
        JSON.parse(decoded)
    catch
        return nothing
    end
    return payload isa AbstractDict ? payload : nothing
end
