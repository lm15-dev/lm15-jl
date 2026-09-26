# Provider login flows (spec/auth-managed.md AUTH-13, AUTH-18, AUTH-26). A flow answers
# four questions: `login` (run a method, return a LoginResult), `renew` (fresh material,
# or LoginDenied: permanent), `request_auth` (what a request sends), `expiry`. Every
# client id, URL, scope and parameter comes from lm15-contract auth/managed/profiles.json,
# copied verbatim into src/data/login_profiles.json: no SDK is the source of those values.
# Material is the provider-private entry saved in the store (store-layout.md).

const LOGIN_PROFILES = JSON.parse(read(joinpath(@__DIR__, "..", "data", "login_profiles.json"), String))["providers"]

"""What a login produced: private `material`, a display `label`, an untrusted
`account_label`, the renewal kind, and non-secret `settings`."""
struct LoginResult
    material::JSONObject
    label::String
    account_label::Maybe{String}
    renewal::String
    settings::Dict{String,String}
end
LoginResult(material, label; account_label=nothing, renewal="refresh_token", settings=Dict{String,String}()) =
    LoginResult(material, label, account_label, renewal, settings)
Base.show(io::IO, r::LoginResult) = print(io, "LoginResult(", repr(r.label), ", <material withheld>)")
"""
    RequestAuth

What a request on a managed connection sends: the `credential` (an AUTH-2 value), extra
`headers`, a credential-dependent `base_url`, an `account_id`, or a `named` cloud identity
(a saved cloud recipe; the chain rung runs).
"""
struct RequestAuth
    credential::Any
    headers::Dict{String,String}
    base_url::Maybe{String}
    account_id::Maybe{String}
    named::Maybe{String}
end
RequestAuth(credential; headers=Dict{String,String}(), base_url=nothing, account_id=nothing, named=nothing) =
    RequestAuth(credential, Dict{String,String}(headers), base_url, account_id, named)
Base.show(io::IO, r::RequestAuth) = print(io, "RequestAuth(<credential withheld>)")

# A request template from the profile, with `<placeholder>` values filled in.
function profile_request(provider, method_id, name, values=Dict{String,String}())
    methods = LOGIN_PROFILES[provider]["methods"]
    request = methods[method_id]["requests"][name]
    if haskey(request, "same_as")
        target = String(request["same_as"])
        request = if occursin('.', target)
            m, r = split(target, '.'; limit=2)
            methods[m]["requests"][r]
        else
            methods[method_id]["requests"][target]
        end
    end
    fill(v) = begin
        out = String(v)
        for (k, x) in values
            out = replace(out, "<$k>" => x)
        end
        out
    end
    params = JSONObject(k => fill(v) for (k, v) in get(request, "params", JSONObject()))
    headers = Pair{String,String}[k => fill(v) for (k, v) in get(request, "headers", JSONObject())]
    return fill(request["url"]), params, headers
end
profile_values(provider) = Dict{String,String}(
    k => String(v) for (k, v) in (("client_id", get(LOGIN_PROFILES[provider], "client_id", nothing)),
        ("scope", get(LOGIN_PROFILES[provider], "scope", nothing))) if v isa AbstractString)
token_urlsafe(n) = String(base64url(rand(RandomDevice(), UInt8, n)))
token_hex(n) = bytes2hex(rand(RandomDevice(), UInt8, n))
now_ms(ctx::LoginContext) = round(Int64, ctx.wall_clock() * 1000)
"""OAuth material with the actual expiry and the numbers the renewal lead needs;
`expires_in_s === nothing` records an unknown expiry."""
function oauth_material(; access, refresh, expires_in_s, now_ms, extra=JSONObject())
    m = JSONObject("type" => "oauth", "access" => access)
    refresh === nothing || isempty(refresh) || (m["refresh"] = refresh)
    m["issued_at"] = now_ms
    if expires_in_s !== nothing && expires_in_s > 0
        m["lifetime_s"] = Float64(expires_in_s)
        m["expires"] = round(Int64, now_ms + expires_in_s * 1000)
    end
    merge!(m, extra)
    return m
end
function material_expiry_ms(m)
    v = get(m, "expires", nothing)
    v isa Real && !(v isa Bool) || return nothing
    return round(Int64, v)
end
function https_url(v; http_ok=false)
    v isa AbstractString && !isempty(v) || return nothing
    u = try
        HTTP.URI(v)
    catch
        return nothing
    end
    (u.scheme == "https" || (http_ok && u.scheme == "http")) && !isempty(u.host) || return nothing
    return String(v)
end

abstract type ProviderFlow end
function flow_expiry(::ProviderFlow, m)
    get(m, "type", nothing) == "api_key" && return "never"
    return material_expiry_ms(m)
end
function flow_lifetime_s(::ProviderFlow, m)
    v = positive_number(get(m, "lifetime_s", nothing))
    v === nothing || return v
    issued, expires = get(m, "issued_at", nothing), material_expiry_ms(m)
    issued isa Real && !(issued isa Bool) && expires !== nothing && return max((expires - issued) / 1000.0, 0.0)
    return nothing
end
function string_body(body, key)
    v = get(body, key, nothing)
    return v isa AbstractString && !isempty(v) ? String(v) : nothing
end

# ─── xAI (RFC 8628 device code) ───────────────────────────────────────
struct XaiFlow <: ProviderFlow end
const XAI_DEFAULT_LIFETIME_S = 3600.0
function xai_material(body, now; previous_refresh=nothing)
    access = string_body(body, "access_token")
    access === nothing && throw(LoginDenied("xAI token response carried no access token"))
    refresh = something(string_body(body, "refresh_token"), previous_refresh, Some(nothing))
    lifetime = something(positive_number(get(body, "expires_in", nothing)), XAI_DEFAULT_LIFETIME_S)
    return oauth_material(; access, refresh, expires_in_s=lifetime, now_ms=now)
end
function flow_login(::XaiFlow, ctx, m::LoginMethod, settings, answers)
    m.id == "device" || throw(ArgumentError(m.id))
    values = profile_values("xai")
    url, params, _ = profile_request("xai", "device", "device_authorization", values)
    reply = http_form(ctx, url, params)
    reply.ok || throw(LoginDenied("xAI refused to start a device authorization (HTTP $(reply.status))"))
    body = reply.body
    device_code, user_code = string_body(body, "device_code"), string_body(body, "user_code")
    (device_code === nothing || user_code === nothing) &&
        throw(LoginDenied("xAI device authorization response is missing required fields"))
    verification = https_url(get(body, "verification_uri", nothing))
    verification === nothing && throw(LoginDenied("xAI returned an untrusted verification URL"))
    complete = get(body, "verification_uri_complete", nothing)
    target = complete isa AbstractString && !isempty(complete) ?
        something(https_url(complete), Some(nothing)) : verification
    target === nothing && throw(LoginDenied("xAI returned an untrusted verification URL"))
    interval, expires_in = positive_number(get(body, "interval", nothing)), positive_number(get(body, "expires_in", nothing))
    notify(ctx, DeviceCodeNotice(user_code, target, something(expires_in, 900.0), something(interval, 5.0)))
    token_url, token_params, _ = profile_request("xai", "device", "device_token", merge(values, Dict("device_code" => device_code)))
    poll = function ()
        r = http_form(ctx, token_url, token_params)
        r.ok && return device_step("complete", xai_material(r.body, now_ms(ctx)))
        error = get(r.body, "error", nothing)
        error == "authorization_pending" && return device_step("pending")
        error == "slow_down" && return device_step("slow_down", nothing, positive_number(get(r.body, "interval", nothing)))
        error in ("access_denied", "authorization_denied") && return device_step("denied")
        error == "expired_token" && return device_step("expired")
        throw(LoginDenied("xAI device token polling failed (HTTP $(r.status))"))
    end
    material = run_device_flow(ctx, poll; interval_s=interval, expires_in_s=expires_in)
    return LoginResult(material, "xAI subscription"; renewal="refresh_token")
end
function flow_renew(::XaiFlow, ctx, material, settings)
    refresh = string_body(material, "refresh")
    refresh === nothing && throw(LoginDenied("xAI credential has no refresh token"))
    url, params, _ = profile_request("xai", "device", "renewal", merge(profile_values("xai"), Dict("refresh" => refresh)))
    reply = http_form(ctx, url, params)
    reply.ok || throw(LoginDenied(reply.status in (400, 401, 403) ?
        "xAI rejected the refresh token (HTTP $(reply.status))" : "xAI refresh failed (HTTP $(reply.status))";
        status=reply.status, provider_code=reply.oauth_error, stage="renewal"))
    return LoginResult(xai_material(reply.body, now_ms(ctx); previous_refresh=refresh), "xAI subscription"; renewal="refresh_token")
end
flow_request_auth(::XaiFlow, material, settings) = RequestAuth(BearerToken(; value=material["access"]))

# ─── Claude (authorization code + PKCE; hosted return page or loopback) ─────
struct ClaudeFlow <: ProviderFlow end
function claude_tokens(body, now)
    access, refresh = string_body(body, "access_token"), string_body(body, "refresh_token")
    (access === nothing || refresh === nothing) && throw(LoginDenied("Claude token response is missing required fields"))
    return oauth_material(; access, refresh, expires_in_s=positive_number(get(body, "expires_in", nothing)), now_ms=now)
end
function flow_login(::ClaudeFlow, ctx, m::LoginMethod, settings, answers)
    m.id in ("browser", "loopback") || throw(auth_operation_error("Unknown Claude login method";
        reason="method_unavailable", stage="discovery", recovery="choose_method", provider="claude-code"))
    check!(ctx)
    profile = LOGIN_PROFILES["claude-code"]["methods"][m.id]
    hosted = m.id == "browser"
    verifier = token_urlsafe(Int(profile["pkce"]["verifier_bytes"]))
    pkce = PKCEPair(verifier, pkce_challenge(verifier))
    state = token_urlsafe(Int(profile["state"]["state_bytes"]))
    redirect = profile["return"]["registered_uri"]
    listener = nothing
    try
        if !hosted
            bind = split(profile["return"]["bind"], ':')
            try
                listener = CallbackListener(; path=HTTP.URI(redirect).path, expected_state=state,
                    port=parse(Int, last(bind)), redirect_host="localhost")
            catch e
                e isa AuthOperationError && e.reason == "method_unavailable" || rethrow()
                notify(ctx, InfoNotice("Could not listen on port $(last(bind)); paste the full redirect URL when the browser finishes."))
            end
        end
        values = merge(profile_values("claude-code"), Dict("challenge" => pkce.challenge, "state" => state))
        authorize, query, _ = profile_request("claude-code", m.id, "authorize", values)
        instructions = hosted ?
            "Sign in to Claude in your browser. On the Authentication code page, copy the whole displayed code (including #state) and paste it here. The full return URL also works. Your browser may be on another machine; no localhost connection is needed." :
            "Sign in to Claude in your browser. If the local callback cannot be reached, paste the full redirect URL (or code#state) here."
        notify(ctx, AuthUrlNotice("$authorize?$(form_encode(query))", instructions))
        p = ManualCodePrompt("return", "Paste the full code#state or return URL here",
            "the full return URL, or code#state (a bare code without state is not accepted)")
        returned = await_return(ctx, listener, p, pasted -> parse_manual_return(pasted; expected_state=state,
            allow_bare_code=false, registered_path=HTTP.URI(redirect).path, registered_uri=redirect))
        check!(ctx)
        notify(ctx, ProgressNotice("exchange", "Exchanging the authorization code…"))
        token_url, token_params, _ = profile_request("claude-code", m.id, "token",
            merge(values, Dict("code" => returned.code, "verifier" => pkce.verifier)))
        reply = http_json(ctx, token_url, token_params)
        reply.ok || throw(LoginDenied(
            "Claude authorization-code exchange failed: $(failure_summary(reply)). The authorization code will not be retried automatically.";
            status=reply.status, provider_code=reply.oauth_error, stage="exchange"))
        check!(ctx)
        return LoginResult(claude_tokens(reply.body, now_ms(ctx)), "Claude subscription"; renewal="refresh_token")
    finally
        listener === nothing || stop!(listener)
    end
end
function flow_renew(::ClaudeFlow, ctx, material, settings)
    refresh = string_body(material, "refresh")
    refresh === nothing && throw(LoginDenied("Claude credential has no refresh token"))
    url, params, _ = profile_request("claude-code", "browser", "renewal", merge(profile_values("claude-code"), Dict("refresh" => refresh)))
    reply = http_json(ctx, url, params)
    reply.ok || throw(LoginDenied("Claude token renewal failed: $(failure_summary(reply))";
        status=reply.status, provider_code=reply.oauth_error, stage="renewal"))
    return LoginResult(claude_tokens(reply.body, now_ms(ctx)), "Claude subscription"; renewal="refresh_token")
end
flow_request_auth(::ClaudeFlow, material, settings) = RequestAuth(BearerToken(; value=material["access"]))

# ─── ChatGPT / Codex (authorization code + PKCE, or device code) ────────────
struct CodexFlow <: ProviderFlow end
const CODEX_DEVICE_TIMEOUT_S = 15 * 60.0
function codex_tokens(body, now)
    access, refresh = string_body(body, "access_token"), string_body(body, "refresh_token")
    (access === nothing || refresh === nothing) && throw(LoginDenied("ChatGPT token response is missing required fields"))
    lifetime = positive_number(get(body, "expires_in", nothing))
    if lifetime === nothing
        exp = int_field(something(jwt_payload(access), obj()), "exp")
        exp === nothing || (lifetime = max((exp * 1000 + 5 * 60 * 1000 - now) / 1000.0, 0.0); lifetime == 0 && (lifetime = nothing))
    end
    claim = something(dict_field(something(jwt_payload(access), obj()), "https://api.openai.com/auth"), obj())
    account = string_field(claim, "chatgpt_account_id")
    account === nothing && throw(LoginDenied("ChatGPT token carries no account id"))
    extra = JSONObject("accountId" => account)
    id_token = string_body(body, "id_token")
    id_token === nothing || (extra["id_token"] = id_token)
    return oauth_material(; access, refresh, expires_in_s=lifetime, now_ms=now, extra)
end
function codex_exchange(ctx, code, verifier, method_id)
    notify(ctx, ProgressNotice("exchange", "Exchanging the authorization code…"))
    url, params, _ = profile_request("openai-codex", method_id, "token",
        merge(profile_values("openai-codex"), Dict("code" => code, "verifier" => verifier)))
    reply = http_form(ctx, url, params)
    reply.ok || throw(LoginDenied("ChatGPT rejected the authorization code (HTTP $(reply.status))";
        status=reply.status, provider_code=reply.oauth_error, stage="exchange"))
    material = codex_tokens(reply.body, now_ms(ctx))
    return LoginResult(material, "ChatGPT subscription"; renewal="refresh_token", account_label=material["accountId"])
end
function flow_login(f::CodexFlow, ctx, m::LoginMethod, settings, answers)
    m.id == "device" && return codex_device(f, ctx)
    m.id == "browser" || throw(ArgumentError(m.id))
    profile = LOGIN_PROFILES["openai-codex"]["methods"]["browser"]
    verifier = token_urlsafe(Int(profile["pkce"]["verifier_bytes"]))
    pkce = PKCEPair(verifier, pkce_challenge(verifier))
    state = token_hex(Int(profile["state"]["state_bytes"]))
    redirect = profile["return"]["registered_uri"]
    port = parse(Int, last(split(profile["return"]["bind"], ':')))
    listener = try
        CallbackListener(; path=HTTP.URI(redirect).path, expected_state=state, port, redirect_host="localhost")
    catch e
        e isa AuthOperationError || rethrow()
        notify(ctx, InfoNotice("Could not listen on port $port; paste the redirect URL when the browser finishes."))
        nothing
    end
    try
        values = merge(profile_values("openai-codex"), Dict("challenge" => pkce.challenge, "state" => state))
        authorize, query, _ = profile_request("openai-codex", "browser", "authorize", values)
        notify(ctx, AuthUrlNotice("$authorize?$(form_encode(query))",
            "Sign in to ChatGPT in your browser. If the browser is on another machine, paste the final redirect URL back here."))
        p = ManualCodePrompt("return", "Paste the redirect URL here (or wait for the browser)")
        returned = await_return(ctx, listener, p, pasted -> parse_manual_return(pasted; expected_state=state,
            allow_bare_code=false, registered_path=HTTP.URI(redirect).path))
        return codex_exchange(ctx, returned.code, pkce.verifier, "browser")
    finally
        listener === nothing || stop!(listener)
    end
end
function codex_device(::CodexFlow, ctx)
    values = profile_values("openai-codex")
    url, params, _ = profile_request("openai-codex", "device", "device_authorization", values)
    reply = http_json(ctx, url, params)
    if !reply.ok
        reply.status == 404 && throw(LoginDenied("ChatGPT device-code login is not enabled for this server; use the browser method"))
        throw(LoginDenied("ChatGPT refused to start a device authorization (HTTP $(reply.status))"))
    end
    body = reply.body
    device_id, user_code = string_body(body, "device_auth_id"), string_body(body, "user_code")
    interval = get(body, "interval", nothing)
    interval isa AbstractString && (interval = tryparse(Float64, strip(interval)))
    (device_id === nothing || user_code === nothing) &&
        throw(LoginDenied("ChatGPT device authorization response is missing required fields"))
    interval_s = interval isa Real && !(interval isa Bool) && isfinite(interval) && interval >= 0 ? Float64(interval) : nothing
    # The verification page is fixed, never taken from the response (profiles.json).
    verification = "https://auth.openai.com/codex/device"
    notify(ctx, DeviceCodeNotice(user_code, verification, CODEX_DEVICE_TIMEOUT_S, something(interval_s == 0 ? nothing : interval_s, 5.0)))
    token_url, token_params, _ = profile_request("openai-codex", "device", "device_token",
        merge(values, Dict("device_auth_id" => device_id, "user_code" => user_code)))
    poll = function ()
        r = http_json(ctx, token_url, token_params)
        if r.ok
            code, verifier = string_body(r.body, "authorization_code"), string_body(r.body, "code_verifier")
            (code === nothing || verifier === nothing) && throw(LoginDenied("ChatGPT device token response is missing required fields"))
            return device_step("complete", (code, verifier))
        end
        r.status in (403, 404) && return device_step("pending")
        error = get(r.body, "error", nothing)
        code = error isa AbstractDict ? get(error, "code", nothing) : error
        code == "deviceauth_authorization_pending" && return device_step("pending")
        code == "slow_down" && return device_step("slow_down")
        throw(LoginDenied("ChatGPT device authorization failed (HTTP $(r.status))"))
    end
    code, verifier = run_device_flow(ctx, poll; interval_s, expires_in_s=CODEX_DEVICE_TIMEOUT_S)
    return codex_exchange(ctx, code, verifier, "device")
end
function flow_renew(::CodexFlow, ctx, material, settings)
    refresh = string_body(material, "refresh")
    refresh === nothing && throw(LoginDenied("ChatGPT credential has no refresh token"))
    url, params, _ = profile_request("openai-codex", "browser", "renewal", merge(profile_values("openai-codex"), Dict("refresh" => refresh)))
    reply = http_form(ctx, url, params)
    reply.ok || throw(LoginDenied("ChatGPT rejected the refresh token (HTTP $(reply.status))";
        status=reply.status, provider_code=reply.oauth_error, stage="renewal"))
    body = copy(reply.body)
    string_body(body, "refresh_token") === nothing && (body["refresh_token"] = refresh)
    m = codex_tokens(body, now_ms(ctx))
    return LoginResult(m, "ChatGPT subscription"; renewal="refresh_token", account_label=m["accountId"])
end
function flow_request_auth(::CodexFlow, material, settings)
    account = something(string_body(material, "accountId"),
        string_field(something(dict_field(something(jwt_payload(material["access"]), obj()), "https://api.openai.com/auth"), obj()), "chatgpt_account_id"),
        Some(nothing))
    return RequestAuth(BearerToken(; value=material["access"]);
        headers=account === nothing ? Dict{String,String}() : Dict("chatgpt-account-id" => account), account_id=account)
end

# ─── GitHub Copilot (device code, then a Copilot token) ───────────────────
struct CopilotFlow <: ProviderFlow end
const COPILOT_HEADERS = [k => v for (k, v) in COPILOT_HEADER_PAIRS]
const COPILOT_DEFAULT_DOMAIN = "github.com"
const COPILOT_DEFAULT_API_BASE = "https://api.individual.githubcopilot.com"
function copilot_domain(settings)
    raw = strip(get(settings, "enterprise_domain", ""))
    isempty(raw) && return COPILOT_DEFAULT_DOMAIN
    host = try
        lowercase(HTTP.URI(occursin("://", raw) ? raw : "https://$raw").host)
    catch
        ""
    end
    !isempty(host) && occursin(r"^[a-z0-9.-]+$", host) || throw(LoginDenied("invalid GitHub Enterprise domain"))
    return host
end
"""The account's API host from the Copilot token, validated against GitHub's domains;
never an arbitrary host a token string names (AUTH-20.9)."""
function copilot_base_url(material, settings)
    token = get(material, "access", "")
    domain = copilot_domain(settings)
    m = token isa AbstractString ? match(r"proxy-ep=([^;]+)", token) : nothing
    if m !== nothing
        api = replace(lowercase(strip(m.captures[1])), r"^proxy\." => "api.")
        allowed = domain == COPILOT_DEFAULT_DOMAIN ? (".githubcopilot.com",) : (".$domain", ".githubcopilot.com")
        occursin(r"^[a-z0-9.-]+$", api) && any(s -> endswith(api, s), allowed) && return "https://$api"
    end
    domain == COPILOT_DEFAULT_DOMAIN || return "https://copilot-api.$domain"
    return COPILOT_DEFAULT_API_BASE
end
function copilot_exchange(ctx, github_token, settings)
    url, _, headers = profile_request("github-copilot", "device", "copilot_token",
        Dict("domain" => copilot_domain(settings), "github_token" => github_token))
    reply = http_get(ctx, url; headers)
    reply.status in (401, 403) && throw(LoginDenied("GitHub rejected the token for Copilot; sign in again"; status=reply.status, stage="renewal"))
    reply.ok || throw(LoginDenied("Copilot token exchange failed (HTTP $(reply.status))"; status=reply.status))
    token, expires_at = string_body(reply.body, "token"), get(reply.body, "expires_at", nothing)
    (token === nothing || !(expires_at isa Real) || expires_at isa Bool) &&
        throw(LoginDenied("Copilot token response is missing required fields"))
    now = now_ms(ctx)
    expires = round(Int64, expires_at * 1000)
    return JSONObject("type" => "oauth", "access" => token, "refresh" => github_token, "issued_at" => now,
        "lifetime_s" => max((expires - now) / 1000.0, 1.0), "expires" => expires)
end
function flow_login(::CopilotFlow, ctx, m::LoginMethod, settings, answers)
    merged = Dict{String,String}(settings)
    isempty(get(answers, "enterprise_domain", "")) || (merged["enterprise_domain"] = answers["enterprise_domain"])
    domain = copilot_domain(merged)
    values = merge(profile_values("github-copilot"), Dict("domain" => domain))
    url, params, headers = profile_request("github-copilot", "device", "device_authorization", values)
    reply = http_form(ctx, url, params; headers)
    reply.ok || throw(LoginDenied("GitHub refused to start a device authorization (HTTP $(reply.status))"))
    body = reply.body
    device_code, user_code, verification = string_body(body, "device_code"), string_body(body, "user_code"), string_body(body, "verification_uri")
    (device_code === nothing || user_code === nothing || verification === nothing) &&
        throw(LoginDenied("GitHub device authorization response is missing required fields"))
    https_url(verification; http_ok=true) === nothing && throw(LoginDenied("GitHub returned an untrusted verification URL"))
    interval_s, expires_s = positive_number(get(body, "interval", nothing)), positive_number(get(body, "expires_in", nothing))
    notify(ctx, DeviceCodeNotice(user_code, verification, something(expires_s, 900.0), something(interval_s, 5.0)))
    token_url, token_params, token_headers = profile_request("github-copilot", "device", "device_token",
        merge(values, Dict("device_code" => device_code)))
    poll = function ()
        r = http_form(ctx, token_url, token_params; headers=token_headers)
        token = string_body(r.body, "access_token")
        token === nothing || return device_step("complete", token)
        error = get(r.body, "error", nothing)
        error == "authorization_pending" && return device_step("pending")
        error == "slow_down" && return device_step("slow_down", nothing, positive_number(get(r.body, "interval", nothing)))
        error == "expired_token" && return device_step("expired")
        error == "access_denied" && return device_step("denied")
        throw(LoginDenied("GitHub device authorization failed (HTTP $(r.status))"))
    end
    github_token = run_device_flow(ctx, poll; interval_s, expires_in_s=expires_s)
    notify(ctx, ProgressNotice("exchange", "Exchanging the GitHub token for a Copilot token…"))
    material = copilot_exchange(ctx, github_token, merged)
    label = domain == COPILOT_DEFAULT_DOMAIN ? "GitHub Copilot" : "GitHub Copilot ($domain)"
    return LoginResult(material, label; renewal="remint",
        settings=domain == COPILOT_DEFAULT_DOMAIN ? Dict{String,String}() : Dict("enterprise_domain" => domain))
end
function flow_renew(::CopilotFlow, ctx, material, settings)
    github_token = string_body(material, "refresh")
    github_token === nothing && throw(LoginDenied("Copilot credential has no GitHub token to renew with"))
    return LoginResult(copilot_exchange(ctx, github_token, settings), "GitHub Copilot"; renewal="remint")
end
flow_request_auth(::CopilotFlow, material, settings) = RequestAuth(BearerToken(; value=material["access"]);
    headers=Dict(COPILOT_HEADERS), base_url=copilot_base_url(material, settings))

# ─── Kimi Code (device code against auth.kimi.com) ───────────────────────
struct KimiFlow <: ProviderFlow end
kimi_host(settings) = rstrip(get(settings, "oauth_host", LOGIN_PROFILES["kimi-code"]["host"]), '/')
function kimi_tokens(body, now)
    access, refresh = string_body(body, "access_token"), string_body(body, "refresh_token")
    (access === nothing || refresh === nothing) && throw(LoginDenied("Kimi Code token response is missing required fields"))
    return oauth_material(; access, refresh, expires_in_s=positive_number(get(body, "expires_in", nothing)), now_ms=now)
end
function flow_login(::KimiFlow, ctx, m::LoginMethod, settings, answers)
    host = kimi_host(settings)
    values = merge(profile_values("kimi-code"), Dict("host" => host))
    url, params, _ = profile_request("kimi-code", "device", "device_authorization", values)
    reply = http_form(ctx, url, params)
    reply.ok || throw(LoginDenied("Kimi Code refused to start a device authorization (HTTP $(reply.status))"))
    body = reply.body
    device_code, user_code = string_body(body, "device_code"), string_body(body, "user_code")
    verification = something(https_url(get(body, "verification_uri_complete", nothing); http_ok=true),
        https_url(get(body, "verification_uri", nothing); http_ok=true), Some(nothing))
    (device_code === nothing || user_code === nothing || verification === nothing) &&
        throw(LoginDenied("Kimi Code device authorization response is missing required fields"))
    interval = positive_number(get(body, "interval", nothing))
    expires_in = something(positive_number(get(body, "expires_in", nothing)), CODEX_DEVICE_TIMEOUT_S)
    notify(ctx, DeviceCodeNotice(user_code, verification, expires_in, something(interval, 5.0)))
    token_url, token_params, _ = profile_request("kimi-code", "device", "device_token", merge(values, Dict("device_code" => device_code)))
    poll = function ()
        r = http_form(ctx, token_url, token_params)
        r.ok && get(r.body, "access_token", nothing) isa AbstractString && return device_step("complete", kimi_tokens(r.body, now_ms(ctx)))
        error = get(r.body, "error", nothing)
        error == "authorization_pending" && return device_step("pending")
        error == "slow_down" && return device_step("slow_down", nothing, positive_number(get(r.body, "interval", nothing)))
        error == "expired_token" && return device_step("expired")
        error == "access_denied" && return device_step("denied")
        throw(LoginDenied("Kimi Code device token request failed (HTTP $(r.status))"))
    end
    material = run_device_flow(ctx, poll; interval_s=interval, expires_in_s=expires_in)
    default_host = rstrip(LOGIN_PROFILES["kimi-code"]["host"], '/')
    return LoginResult(material, "Kimi Code subscription"; renewal="refresh_token",
        settings=host == default_host ? Dict{String,String}() : Dict("oauth_host" => host))
end
function flow_renew(::KimiFlow, ctx, material, settings)
    refresh = string_body(material, "refresh")
    refresh === nothing && throw(LoginDenied("Kimi Code credential has no refresh token"))
    url, params, _ = profile_request("kimi-code", "device", "renewal",
        merge(profile_values("kimi-code"), Dict("host" => kimi_host(settings), "refresh" => refresh)))
    reply = http_form(ctx, url, params)
    (reply.status in (401, 403) || get(reply.body, "error", nothing) == "invalid_grant") &&
        throw(LoginDenied("Kimi Code rejected the refresh token (HTTP $(reply.status))"; status=reply.status, stage="renewal"))
    # 429 is transient: not a denial of the credential (the manager marks
    # needs_login only on LoginDenied); 5xx was already a ServerError.
    reply.ok || throw(RateLimitError("Kimi Code rate-limited the token refresh (HTTP $(reply.status))";
        provider="kimi-code", status=reply.status))
    return LoginResult(kimi_tokens(reply.body, now_ms(ctx)), "Kimi Code subscription"; renewal="refresh_token")
end
flow_request_auth(::KimiFlow, material, settings) = RequestAuth(BearerToken(; value=material["access"]))

# ─── Meta (device code, then a minted Model API key) ─────────────────────
struct MetaFlow <: ProviderFlow end
function meta_mint(ctx, identity)
    notify(ctx, ProgressNotice("exchange", "Enabling Meta Model API access…"))
    url, params, headers = profile_request("meta", "device", "key_mint", Dict("identity_token" => identity))
    reply = http_json(ctx, url, params; headers)
    reply.status in (401, 403) && throw(LoginDenied("Meta session is no longer valid; sign in again"; status=reply.status, stage="renewal"))
    reply.ok || throw(LoginDenied("Meta API key mint failed (HTTP $(reply.status))"; status=reply.status))
    key = string_body(reply.body, "api_key")
    if key === nothing
        action = https_url(get(reply.body, "action_url", nothing); http_ok=true)
        throw(LoginDenied("Meta did not issue an API key" * (action === nothing ? "" : "; complete setup at $action")))
    end
    lifetime = Float64(LOGIN_PROFILES["meta"]["methods"]["device"]["tokens"]["lifetime_s"])
    now = now_ms(ctx)
    return JSONObject("type" => "oauth", "access" => key, "refresh" => identity, "issued_at" => now,
        "lifetime_s" => lifetime, "expires" => round(Int64, now + lifetime * 1000))
end
function flow_login(::MetaFlow, ctx, m::LoginMethod, settings, answers)
    m.id == "device" || throw(ArgumentError(m.id))
    values = profile_values("meta")
    url, params, _ = profile_request("meta", "device", "device_authorization", values)
    reply = http_form(ctx, url, params)
    reply.ok || throw(LoginDenied("Meta refused to start a device authorization (HTTP $(reply.status))"))
    body = reply.body
    device_code, user_code = string_body(body, "device_code"), string_body(body, "user_code")
    verification = something(https_url(get(body, "verification_uri_complete", nothing); http_ok=true),
        https_url(get(body, "verification_uri", nothing); http_ok=true), Some(nothing))
    (device_code === nothing || user_code === nothing || verification === nothing) &&
        throw(LoginDenied("Meta device authorization response is missing required fields"))
    interval, expires_in = positive_number(get(body, "interval", nothing)), positive_number(get(body, "expires_in", nothing))
    notify(ctx, DeviceCodeNotice(user_code, verification, something(expires_in, 900.0), something(interval, 5.0)))
    token_url, token_params, _ = profile_request("meta", "device", "device_token", merge(values, Dict("device_code" => device_code)))
    poll = function ()
        r = http_form(ctx, token_url, token_params)
        token = string_body(r.body, "access_token")
        r.ok && token !== nothing && return device_step("complete", token)
        error = get(r.body, "error", nothing)
        error == "authorization_pending" && return device_step("pending")
        error == "slow_down" && return device_step("slow_down", nothing, positive_number(get(r.body, "interval", nothing)))
        error == "access_denied" && return device_step("denied")
        error == "expired_token" && return device_step("expired")
        throw(LoginDenied("Meta device token request failed (HTTP $(r.status))"))
    end
    identity = run_device_flow(ctx, poll; interval_s=interval, expires_in_s=expires_in)
    return LoginResult(meta_mint(ctx, identity), "Meta (Muse subscription)"; renewal="remint")
end
function flow_renew(::MetaFlow, ctx, material, settings)
    identity = string_body(material, "refresh")
    identity === nothing && throw(LoginDenied("Meta credential has no identity token to re-mint with"))
    return LoginResult(meta_mint(ctx, identity), "Meta (Muse subscription)"; renewal="remint")
end
flow_request_auth(::MetaFlow, material, settings) = RequestAuth(ApiKey(; value=material["access"]))

# ─── OpenRouter (PKCE; the code is exchanged for a minted API key) ─────────
struct OpenRouterFlow <: ProviderFlow end
function flow_login(::OpenRouterFlow, ctx, m::LoginMethod, settings, answers)
    m.id == "browser" || throw(ArgumentError(m.id))
    verifier = token_urlsafe(Int(LOGIN_PROFILES["openrouter"]["methods"]["browser"]["pkce"]["verifier_bytes"]))
    pkce = PKCEPair(verifier, pkce_challenge(verifier))
    path = "/oauth/callback/$(token_urlsafe(24))"
    # No state in OpenRouter's protocol: the one-time random callback path plus
    # PKCE is the evidenced equivalent binding (AUTH-18).
    listener = CallbackListener(; path, expected_state=nothing, port=0)
    try
        authorize, query, _ = profile_request("openrouter", "browser", "authorize",
            Dict("callback_url" => redirect_uri(listener), "challenge" => pkce.challenge))
        notify(ctx, AuthUrlNotice("$authorize?$(form_encode(query))",
            "Sign in to OpenRouter in your browser and approve the key. If the browser is on another machine, paste the final redirect URL back here."))
        p = ManualCodePrompt("return", "Paste the redirect URL or code here (or wait for the browser)", "the full redirect URL, or the code")
        returned = await_return(ctx, listener, p, pasted -> parse_manual_return(pasted; expected_state=nothing,
            allow_bare_code=true, registered_path=path))
        notify(ctx, ProgressNotice("exchange", "Exchanging the code for an API key…"))
        url, params, _ = profile_request("openrouter", "browser", "token", Dict("code" => returned.code, "verifier" => pkce.verifier))
        reply = http_json(ctx, url, params)
        reply.ok || throw(LoginDenied("OpenRouter rejected the authorization code (HTTP $(reply.status))";
            status=reply.status, provider_code=reply.oauth_error, stage="exchange"))
        key = string_body(reply.body, "key")
        key === nothing && throw(LoginDenied("OpenRouter returned no key"))
        return LoginResult(JSONObject("type" => "api_key", "key" => key, "minted" => true), "OpenRouter (minted key)"; renewal="none")
    finally
        stop!(listener)
    end
end
flow_renew(::OpenRouterFlow, ctx, material, settings) = LoginResult(copy(material), "OpenRouter (minted key)"; renewal="none")
flow_request_auth(::OpenRouterFlow, material, settings) = RequestAuth(ApiKey(; value=material["key"]))
flow_expiry(::OpenRouterFlow, material) = "never"

# ─── Recipes: connections that are recipes, not tokens ──────────────────
# api_key (a literal key), env (the variable's NAME, read at request time), external
# (another tool's login, read and renewed in place, nothing copied), cloud (a named
# cloud identity), local (a keyless local server).
const EXTERNAL_SOURCES = OrderedDict(
    "claude-code-cli" => ("claude-code", "your Claude Code login (~/.claude/.credentials.json)"),
    "codex-cli" => ("openai-codex", "your Codex CLI login (~/.codex/auth.json)"),
    "pi-xai" => ("xai", "your Pi agent xAI login (~/.pi/agent/auth.json)"),
)
api_key_method(console_url) = LoginMethod("api_key", "Paste an API key", "api_key", "form";
    fields=(MethodField("key", "API key"; type="secret"),),
    guidance=console_url === nothing ? nothing : "Create one at $console_url",
    billing_note="Metered per token by the provider.")
env_method(keys) = LoginMethod("env", "Use the key in \$$(first(keys)) from the environment", "api_key", "source_recipe";
    fields=(MethodField("name", "Environment variable"; type="select", options=Tuple(SelectOption(k, "\$$k") for k in keys)),),
    billing_note="Metered per token by the provider; the variable's value is read at request time, never saved.")
function external_method(source)
    route, label = EXTERNAL_SOURCES[source]
    return LoginMethod("external:$source", "Use $label", "account", "source_recipe"; subscription=true,
        billing_note="Whatever that tool's login is entitled to; LM15 reads and renews it in place and copies nothing.",
        guidance="Sign in with that tool first if it says no credential is present.")
end
struct RecipeFlow <: ProviderFlow
    provider::String
end
function flow_login(f::RecipeFlow, ctx, m::LoginMethod, settings, answers)
    if m.id == "api_key"
        key = strip(get(answers, "key", ""))
        isempty(key) && throw(LoginDenied("no API key was entered"))
        return LoginResult(JSONObject("type" => "api_key", "key" => String(key)), "$(f.provider) API key"; renewal="none")
    elseif m.id == "env"
        name = get(answers, "name", "")
        isempty(name) && throw(LoginDenied("no environment variable was chosen"))
        return LoginResult(JSONObject("type" => "env", "name" => name), "$(f.provider) key from \$$name"; renewal="recipe")
    elseif startswith(m.id, "external:")
        source = m.id[10:end]
        haskey(EXTERNAL_SOURCES, source) || throw(LoginDenied("unknown external source $(repr(source))"))
        probe_external(source)  # fail now, typed, if that tool has no login
        return LoginResult(JSONObject("type" => "external", "source" => source),
            "$(f.provider) via $(EXTERNAL_SOURCES[source][2])"; renewal="external")
    elseif m.id == "cloud"
        named = get(answers, "named", "")
        named in NAMED_CREDENTIALS || throw(LoginDenied("choose one of $(join(NAMED_CREDENTIALS, ", "))"))
        return LoginResult(JSONObject("type" => "cloud", "named" => named), "$(f.provider) via $named identity"; renewal="recipe")
    elseif m.id == "local"
        base = something(nonempty_string(get(answers, "base_url", nothing)), nonempty_string(get(settings, "base_url", nothing)), "")
        return LoginResult(JSONObject("type" => "local", "base_url" => base, "key" => something(nonempty_string(get(answers, "key", nothing)), "local")),
            "$(f.provider) local server"; renewal="none",
            settings=isempty(base) ? Dict{String,String}() : Dict("base_url" => base))
    end
    throw(ArgumentError(m.id))
end
flow_renew(f::RecipeFlow, ctx, material, settings) =
    LoginResult(copy(material), f.provider; renewal=string(get(material, "type", "none")))
function flow_request_auth(::RecipeFlow, material, settings)
    kind = get(material, "type", nothing)
    kind == "api_key" && return RequestAuth(ApiKey(; value=material["key"]))
    if kind == "env"
        name = material["name"]
        value = get(ENV, name, "")
        isempty(value) && throw(LoginDenied("\$$name is not set in this process's environment"))
        return RequestAuth(ApiKey(; value))
    end
    kind == "external" && return external_auth(material["source"])
    kind == "local" && return RequestAuth(ApiKey(; value=something(nonempty_string(get(material, "key", nothing)), "local"));
        base_url=nonempty_string(get(material, "base_url", nothing)))
    kind == "cloud" && return RequestAuth(nothing; named=material["named"])
    throw(LoginDenied("unknown connection material $(repr(kind))"))
end
function flow_expiry(::RecipeFlow, material)
    get(material, "type", nothing) in ("api_key", "env", "local", "cloud") && return "never"
    return nothing  # external: the owning tool knows
end
pi_agent_auth_path(env=ENV) = joinpath(get(env, "HOME", homedir()), ".pi", "agent", "auth.json")
function probe_external(source)
    try
        source == "claude-code-cli" && return (stored_credential("claude-code"); nothing)
        source == "codex-cli" && return (stored_credential("openai-codex"); nothing)
        source == "pi-xai" && return (read_xai_credential(pi_agent_auth_path()); nothing)
    catch e
        e isa NotConfiguredError || rethrow()
        throw(LoginDenied("that tool has no saved login here; sign in with it first"))
    end
    throw(LoginDenied("unknown external source $(repr(source))"))
end
"""Resolve through the legacy loaders: locked, double-checked renewal written back to the
owning file (AUTH-3/4); nothing is copied into LM15's store."""
function external_auth(source)
    if source == "claude-code-cli"
        return RequestAuth(BearerToken(; value=get_local_credential("claude-code").access_token))
    elseif source == "codex-cli"
        c = get_local_credential("openai-codex")
        account = something(c.account_id, Some(nothing))
        return RequestAuth(BearerToken(; value=c.access_token);
            headers=account === nothing ? Dict{String,String}() : Dict("chatgpt-account-id" => account), account_id=account)
    elseif source == "pi-xai"
        return RequestAuth(BearerToken(; value=get_local_credential("xai", pi_agent_auth_path()).access_token))
    end
    throw(LoginDenied("unknown external source $(repr(source))"))
end

# ─── The provider table (definitions only; AUTH-13.2) ─────────────────────
const RADIUS_ID = "radius"
account_flow(provider) = get(Dict{String,ProviderFlow}(
    "xai" => XaiFlow(), "claude-code" => ClaudeFlow(), "openai-codex" => CodexFlow(), "openrouter" => OpenRouterFlow(),
    "meta" => MetaFlow(), "kimi-code" => KimiFlow(), "github-copilot" => CopilotFlow()), provider, nothing)
const SERVICE_LABELS = Dict(
    "anthropic" => "Anthropic", "claude-code" => "Anthropic", "openai" => "OpenAI", "openai-chat" => "OpenAI",
    "openai-codex" => "OpenAI", "gemini" => "Google", "vertex" => "Google Cloud", "vertex-anthropic" => "Google Cloud",
    "vertex-express" => "Google Cloud", "azure" => "Microsoft Azure", "azure-chat" => "Microsoft Azure",
    "azure-anthropic" => "Microsoft Azure", "aws-anthropic" => "AWS", "bedrock-anthropic" => "AWS",
    "bedrock-chat" => "AWS", "bedrock-mantle-chat" => "AWS", "meta" => "Meta", "meta-chat" => "Meta",
    "meta-anthropic" => "Meta", "moonshotai" => "Moonshot AI", "moonshotai-anthropic" => "Moonshot AI",
    "moonshotai-responses" => "Moonshot AI", "kimi-code" => "Moonshot AI", "deepseek" => "DeepSeek",
    "deepseek-anthropic" => "DeepSeek", "groq" => "Groq", "openrouter" => "OpenRouter", "xai" => "xAI",
    "zai" => "Z.AI", "typesafe" => "TypeSafe", "ollama" => "Local", "vllm" => "Local", "sglang" => "Local",
    "github-copilot" => "GitHub", "deepinfra" => "DeepInfra", "together" => "Together AI", "fireworks" => "Fireworks AI",
    "parasail" => "Parasail",
)
function account_methods(provider)
    profile = LOGIN_PROFILES[provider]
    out = LoginMethod[]
    for (id, m) in profile["methods"]
        fields = Tuple(MethodField(f["id"], f["id"] == "enterprise_domain" ? "GitHub Enterprise domain (blank for github.com)" : f["id"];
            type=f["type"], required=f["required"], help=f["id"] == "enterprise_domain" ? "e.g. company.ghe.com" : nothing)
            for f in get(m, "fields", []))
        push!(out, LoginMethod(id, account_method_label(provider, id), "account", m["flow"];
            availability=m["availability"], reason=get(m, "availability_reason", nothing), fields,
            delivery=Tuple(String(d) for d in m["delivery"] if d in ("loopback", "manual", "device")),
            subscription=m["subscription"]))
    end
    return out
end
const ACCOUNT_METHOD_LABELS = Dict(
    ("xai", "device") => "Sign in with SuperGrok or X Premium",
    ("claude-code", "browser") => "Sign in with Claude (paste code from hosted page)",
    ("claude-code", "loopback") => "Sign in with Claude (local browser callback)",
    ("openai-codex", "browser") => "Sign in with ChatGPT (browser)",
    ("openai-codex", "device") => "Sign in with ChatGPT (device code, for SSH/headless)",
    ("github-copilot", "device") => "Sign in with GitHub (Copilot subscription)",
    ("kimi-code", "device") => "Sign in with Kimi Code (subscription)",
    ("meta", "device") => "Sign in with Meta (Muse subscription)",
    ("openrouter", "browser") => "Sign in with OpenRouter (creates an API key for this app)",
)
account_method_label(provider, id) = get(ACCOUNT_METHOD_LABELS, (provider, id), "Sign in ($id)")
function recipe_methods(provider)
    methods = LoginMethod[]
    for (source, (route, _)) in EXTERNAL_SOURCES
        route == provider && push!(methods, external_method(source))
    end
    definition = get(PROVIDERS, provider, nothing)
    definition === nothing && return Tuple(methods)
    access = definition.access
    if endswith(access.credential_policy, "-chain")
        push!(methods, LoginMethod("cloud", "Use a named cloud identity", "cloud_identity", "source_recipe";
            fields=(MethodField("named", "Identity"; type="select", options=Tuple(SelectOption(n, n) for n in NAMED_CREDENTIALS)),),
            billing_note="Billed to that cloud account."))
    end
    if definition.placeholder_key !== nothing
        push!(methods, LoginMethod("local", "Local server (no key needed)", "local_server", "source_recipe";
            fields=(MethodField("base_url", "Server URL"; required=false),)))
        return Tuple(methods)
    end
    if access.credential_policy in ("key", "oauth-unless-explicit", "aws-chain", "azure-chain", "gcp-chain")
        push!(methods, api_key_method(definition.console_url))
        isempty(access.env_keys) || push!(methods, env_method(access.env_keys))
    end
    return Tuple(methods)
end
const LOGIN_TABLE = Ref{Union{Nothing,OrderedDict{String,Tuple{ProviderDescriptor,RecipeFlow,Any}}}}(nothing)
function login_table()
    LOGIN_TABLE[] === nothing || return LOGIN_TABLE[]
    table = OrderedDict{String,Tuple{ProviderDescriptor,RecipeFlow,Any}}()
    ids = sort!(unique(vcat(collect(keys(PROVIDERS)), ["xai", "claude-code", "openai-codex", "openrouter", "meta", "kimi-code", "github-copilot"])))
    for provider in ids
        account = account_flow(provider)
        recipes = recipe_methods(provider)
        definition = get(PROVIDERS, provider, nothing)
        console = definition === nothing ? nothing : definition.console_url
        descriptor = if account !== nothing
            profile = LOGIN_PROFILES[provider]
            routes = provider == "meta" ? ("meta", "meta-chat", "meta-anthropic") : (String(profile["route"]),)
            ProviderDescriptor(provider, profile["label"], get(SERVICE_LABELS, provider, provider), routes,
                (account_methods(provider)..., recipes...), nothing, console)
        else
            ProviderDescriptor(provider, provider, get(SERVICE_LABELS, provider, provider), (provider,), recipes, nothing, console)
        end
        table[provider] = (descriptor, RecipeFlow(provider), account)
    end
    # Radius: in the ratified inventory; its model protocol is not implemented,
    # so login alone would advertise a model connection that cannot make a request.
    table[RADIUS_ID] = (ProviderDescriptor(RADIUS_ID, "Radius", "Radius", (),
        (LoginMethod("browser", "Sign in with Radius", "account", "authorization_code"; availability="unavailable",
            reason="Radius's model protocol is not implemented in LM15.jl; login without inference would be a false 'supported' claim"),),
        nothing, nothing), RecipeFlow(RADIUS_ID), nothing)
    sort!(table)
    LOGIN_TABLE[] = table
    return table
end
login_provider_ids() = collect(keys(login_table()))
function login_descriptor(provider)
    entry = get(login_table(), canonical_provider(provider), nothing)
    entry === nothing && throw(KeyError(provider))
    return entry[1]
end
function login_flow(provider, method_id)
    _, recipe, account = login_table()[canonical_provider(provider)]
    account !== nothing && any(m -> m.id == method_id && m.kind == "account" && !startswith(m.id, "external:"),
        login_descriptor(provider).methods) && return account
    return recipe
end
function flow_for_material(provider, material)
    _, recipe, account = login_table()[canonical_provider(provider)]
    kind = get(material, "type", nothing)
    kind in ("api_key", "env", "external", "local", "cloud") && get(material, "minted", false) !== true && return recipe
    account === nothing || return account
    return recipe
end
