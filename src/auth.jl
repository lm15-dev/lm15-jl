const REFRESH_SKEW_MS = 300_000
const CLAUDE_HINT = "Log in again: run `claude` and use /login"
const CODEX_HINT = "Log in again: run `codex login`"
const XAI_HINT = "Run LM15.login(\"xai\")"

struct LocalOAuthCredential
    access_token::String
    refresh_token::Maybe{String}
    expires_at_ms::Maybe{Int64}
    account_id::Maybe{String}
end
access_token(c::LocalOAuthCredential) = c.access_token
access_token(c::Union{ApiKey,BearerToken}) = c.value
account_id(c::LocalOAuthCredential) = c.account_id
has_refresh_token(c::LocalOAuthCredential) = c.refresh_token !== nothing
function is_expired(c::LocalOAuthCredential; now=time())
    return c.expires_at_ms !== nothing && now*1000 + REFRESH_SKEW_MS >= c.expires_at_ms
end
is_expired(::ApiKey; now=time()) = false
function expiry_seconds(s)
    s === nothing && return nothing
    m=match(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?(Z|z|[+-]\d\d:\d\d)?$", s)
    m === nothing && throw(ArgumentError("expiry must be RFC 3339"))
    seconds=datetime2unix(DateTime(m[1]))
    m[2] === nothing || (seconds += parse(Float64, "0."*m[2]))
    if m[3] !== nothing && !(m[3] in ("Z", "z"))
        zone=m[3]
        sign=startswith(zone, "-") ? -1 : 1
        seconds -= sign*(parse(Int, zone[2:3])*3600 + parse(Int, zone[5:6])*60)
    end
    return seconds
end
function is_expired(c::Union{BearerToken,AwsCredentials}; now=time())
    return c.expires_at !== nothing && expiry_seconds(c.expires_at)-now <= 300
end
Base.show(io::IO, ::LocalOAuthCredential) = print(io, "LocalOAuthCredential(<redacted>)")
Base.show(io::IO, ::MIME"text/plain", c::LocalOAuthCredential) = show(io, c)
function string_field(d, k)
    return (v=get(d, k, nothing); v isa AbstractString && !isempty(v) ? String(v) : nothing)
end
dict_field(d, k) = (v=get(d, k, nothing); v isa AbstractDict ? v : nothing)
function int_field(d, k)
    return (
        v=get(d, k, nothing);
        v isa Real && !(v isa Bool) && isfinite(v) && isinteger(v) ? Int64(v) : nothing
    )
end
default_claude_credentials_path() = joinpath(homedir(), ".claude", ".credentials.json")
default_codex_auth_path() = joinpath(homedir(), ".codex", "auth.json")
function default_credentials_path(env=ENV)
    override=get(env, "LM15_CREDENTIALS_PATH", "")
    isempty(override) || return expanduser(override)
    return joinpath(
        get(env, "XDG_CONFIG_HOME", joinpath(get(env, "HOME", homedir()), ".config")),
        "lm15",
        "credentials.json",
    )
end
function read_json_object(path, provider, hint)
    try
        d=JSON.parse(read(path, String))
        d isa AbstractDict || throw(ArgumentError("not an object"))
        return d
    catch
        throw(
            NotConfiguredError(provider, "Missing, unreadable, or malformed credential file.", hint)
        )
    end
end
function jwt_payload(token)
    try
        pieces=split(token, '.')
        length(pieces)==3 || return nothing
        s=replace(pieces[2], '-'=>'+', '_'=>'/')
        d=JSON.parse(String(base64decode(s*"="^mod(-ncodeunits(s), 4))))
        return d isa AbstractDict ? d : nothing
    catch
        return nothing
    end
end
function read_claude_code_credential(path::AbstractString=default_claude_credentials_path())
    d=read_json_object(path, "claude-code", CLAUDE_HINT)
    raw=dict_field(d, "claudeAiOauth")
    raw === nothing &&
        throw(NotConfiguredError("claude-code", "Missing claudeAiOauth section.", CLAUDE_HINT))
    access=string_field(raw, "accessToken")
    access === nothing &&
        throw(NotConfiguredError("claude-code", "Missing access token.", CLAUDE_HINT))
    return LocalOAuthCredential(
        access, string_field(raw, "refreshToken"), int_field(raw, "expiresAt"), nothing
    )
end
function read_codex_cli_credential(path::AbstractString=default_codex_auth_path())
    d=read_json_object(path, "openai-codex", CODEX_HINT)
    raw=dict_field(d, "tokens")
    raw === nothing &&
        throw(NotConfiguredError("openai-codex", "Missing tokens section.", CODEX_HINT))
    access=string_field(raw, "access_token")
    access === nothing &&
        throw(NotConfiguredError("openai-codex", "Missing access token.", CODEX_HINT))
    payload=something(jwt_payload(access), obj())
    expires=int_field(payload, "exp")
    claim=something(dict_field(payload, "https://api.openai.com/auth"), obj())
    account=string_field(raw, "account_id")
    account === nothing && (account=string_field(claim, "chatgpt_account_id"))
    return LocalOAuthCredential(
        access,
        string_field(raw, "refresh_token"),
        expires === nothing ? nothing : expires*1000,
        account,
    )
end
function read_xai_with_source(path=nothing; env=ENV)
    paths=if path === nothing
        (
            default_credentials_path(env),
            joinpath(get(env, "HOME", homedir()), ".pi", "agent", "auth.json"),
        )
    else
        (path,)
    end
    for file in paths
        d=try
            read_json_object(file, "xai", XAI_HINT)
        catch e
            e isa NotConfiguredError || rethrow()
            continue
        end
        raw=dict_field(d, "xai")
        raw === nothing && continue
        access=string_field(raw, "access")
        access === nothing && continue
        return LocalOAuthCredential(
            access, string_field(raw, "refresh"), int_field(raw, "expires"), nothing
        ),
        file
    end
    return throw(NotConfiguredError("xai", "No stored subscription credential.", XAI_HINT))
end
read_xai_credential(path=nothing; kw...) = first(read_xai_with_source(path; kw...))
"""
    xai_stored_state(path=nothing; env=ENV)

The stored xAI subscription's state, offline (AUTH-1, ratified R2/R3 2026-09-22):
`:usable` (fresh, or expired with a refresh token), `:unusable` (stored, expired, no
refresh token: BLOCKS the environment key), `:logged_out` (signed out under managed
Auth: the non-secret marker blocks the key after a restart too), `:absent`.
"""
function xai_stored_state(path=nothing; env=ENV)
    paths=if path === nothing
        (default_credentials_path(env), joinpath(get(env, "HOME", homedir()), ".pi", "agent", "auth.json"))
    else
        (path,)
    end
    for file in paths
        d=try
            read_json_object(file, "xai", XAI_HINT)
        catch e
            e isa NotConfiguredError || rethrow()
            continue
        end
        raw=dict_field(d, "xai")
        if raw!==nothing && string_field(raw, "access")!==nothing
            c=LocalOAuthCredential(raw["access"], string_field(raw, "refresh"), int_field(raw, "expires"), nothing)
            return !is_expired(c) || has_refresh_token(c) ? :usable : :unusable
        end
        slots=dict_field(something(dict_field(d, "_lm15"), obj()), "slots")
        slot=slots===nothing ? nothing : dict_field(slots, "xai")
        slot!==nothing && get(slot, "logged_out", false)===true && return :logged_out
    end
    return :absent
end
function stored_credential(provider, path=nothing; env=ENV)
    home=get(env, "HOME", homedir())
    provider == "claude-code" && return read_claude_code_credential(
        path === nothing ? joinpath(home, ".claude", ".credentials.json") : path
    )
    provider == "openai-codex" && return read_codex_cli_credential(
        path === nothing ? joinpath(home, ".codex", "auth.json") : path
    )
    return read_xai_credential(path; env)
end
function usable_stored(provider, path=nothing; env=ENV)
    try
        c=stored_credential(provider, path; env)
        return !is_expired(c) || has_refresh_token(c)
    catch e
        e isa NotConfiguredError || rethrow()
        false
    end
end

struct AuthStep
    kind::String
    detail::String
    state::Symbol
end
struct AuthReport
    provider::String
    steps::Vector{AuthStep}
    configured::Bool
    settings::Dict{String,String}
    settings_from::OrderedDict{String,Any}
    base_url::Maybe{String}
    base_url_from::Maybe{String}
    named::Maybe{String}
    problems::Vector{String}
end
function AuthReport(p, s, c, settings=Dict{String,String}(); settings_from=OrderedDict{String,Any}(),
    base_url=nothing, base_url_from=nothing, named=nothing, problems=String[])
    return AuthReport(p, s, c, Dict{String,String}(settings), settings_from, base_url, base_url_from, named, problems)
end
function selected_step(r::AuthReport)
    i=findfirst(s->s.state===:selected, r.steps)
    return i === nothing ? nothing : r.steps[i]
end
function describe(r::AuthReport)
    marks=Dict(:selected=>"=>", :shadowed=>" ~", :absent=>" -", :unprobed=>" ?")
    lines=[
        "auth for provider $(repr(r.provider)):";
        ["  $(marks[s.state]) $(s.kind): $(s.detail)" for s in r.steps]
    ]
    r.named===nothing || insert!(lines, 2,
        "  named credential \"$(r.named)\": only its rungs are tried; the chain is not walked")
    push!(lines, "  configured: "*(r.configured ? "yes" : "no"))
    if isempty(r.settings_from)
        append!(lines, ["  setting $k: $(r.settings[k])" for k in sort!(collect(keys(r.settings)))])
    else
        for (k, v) in r.settings_from
            push!(lines, if v.state=="unprobed"
                "  setting $k: unprobed (the $(v.from) server is asked at request time)"
            elseif v.value===nothing
                "  setting $k: missing"
            else
                "  setting $k: $(v.value) (from $(v.from))"
            end)
        end
    end
    r.base_url===nothing || push!(lines, "  base URL: $(r.base_url) (from $(r.base_url_from))")
    append!(lines, ["  problem: $m" for m in r.problems])
    return join(lines, "\n")
end
Base.show(io::IO, r::AuthReport) = print(io, describe(r))
Base.show(io::IO, ::MIME"text/plain", r::AuthReport) = show(io, r)
function explicit_source(provider, api_keys)
    seen=Set{String}()
    for key in keys(api_keys)
        canonical=canonical_provider(key)
        canonical in seen &&
            throw(NotConfiguredError("duplicate spellings for provider $canonical"))
        push!(seen, canonical)
        haskey(PROVIDERS, canonical) ||
            throw(NotConfiguredError("unknown provider configuration key $key"))
    end
    exact=[k for k in keys(api_keys) if canonical_provider(k)==provider]
    if isempty(exact)
        env_keys=provider_definition(provider).access.env_keys
        isempty(env_keys) ||
            (exact=[k for k in keys(api_keys) if provider_definition(k).access.env_keys==env_keys])
    end
    length(exact)>1 && throw(
        NotConfiguredError("ambiguous explicit credentials for $provider; supply one exact entry"),
    )
    isempty(exact) && return nothing
    source=only(exact)
    v=api_keys[source]
    (v === nothing || (v isa AbstractString && isempty(v))) &&
        throw(NotConfiguredError("empty explicit credential; no environment fallback"))
    return source
end
function explain_auth(
    provider::AbstractString;
    env=nothing,
    api_keys=Dict(),
    api_key_providers=String[],
    claude_credentials_path=nothing,
    codex_auth_path=nothing,
    xai_credentials_path=nothing,
    settings=Dict{String,String}(),
    files=nothing,
    home=nothing,
    credential=nothing,
    base_url=nothing,
)
    environment=env === nothing ? ENV : env
    p=provider_definition(provider)
    policy=p.access
    steps=AuthStep[]
    check_named(policy, credential)
    if endswith(policy.credential_policy, "-chain")
        explicit=if isempty(api_key_providers)
            api_keys
        else
            Dict(k=>ApiKey("presence-only") for k in api_key_providers)
        end
        return explain_cloud(
            p.id;
            env=environment,
            api_keys=explicit,
            settings,
            files,
            home=home===nothing ? get(environment, "HOME", homedir()) : home,
            credential,
            base_url,
        )
    end
    path=if p.id=="claude-code"
        claude_credentials_path
    elseif p.id=="openai-codex"
        codex_auth_path
    else
        xai_credentials_path
    end
    function oauth_step(selected)
        usable=usable_stored(p.id, path; env=environment)
        return AuthStep(
            "oauth-file",
            if usable
                "stored login is usable (value never shown)"
            else
                "missing, unreadable, or unrefreshable"
            end,
            usable ? (selected ? :shadowed : :selected) : :absent,
        )
    end
    if policy.credential_policy=="oauth"
        s=oauth_step(false)
        return AuthReport(p.id, [s], s.state===:selected)
    end
    keysmap=if isempty(api_key_providers)
        api_keys
    else
        Dict(k=>ApiKey("presence-only") for k in api_key_providers)
    end
    source=explicit_source(p.id, keysmap)
    selected=source !== nothing
    push!(
        steps,
        AuthStep(
            "api_keys",
            selected ? "provided via $source (value never shown)" : "not provided",
            selected ? :selected : :absent,
        ),
    )
    blocked=false
    if policy.credential_policy=="oauth-unless-explicit"
        # A usable stored login outranks env keys (it spends no money per token);
        # an unusable or signed-out one BLOCKS them (R3, 2026-09-22).
        s=oauth_step(selected)
        state=xai_stored_state(path; env=environment)
        if state===:logged_out && !selected
            s=AuthStep("oauth-file", "signed out (marker present)", :absent)
            blocked=true
        elseif state===:unusable && !selected
            blocked=true
        end
        push!(steps, s)
        selected |= s.state===:selected
    end
    for key in policy.env_keys
        present=!isempty(get(environment, key, ""))
        detail=if !present
            "not set"
        elseif blocked && !selected
            "set, blocked by the failed/signed-out subscription (pass it explicitly to use it)"
        else
            "set (value never shown)"
        end
        push!(steps, AuthStep("env:$key", detail, present ? (selected || blocked ? :shadowed : :selected) : :absent))
        selected |= present && !blocked
    end
    if p.placeholder_key !== nothing
        push!(
            steps, AuthStep("placeholder", "local-server default", selected ? :shadowed : :selected)
        )
        selected=true
    end
    return AuthReport(p.id, steps, selected, Dict{String,String}(settings))
end

# An OS-backed lock is released by the kernel even if the process dies. It uses
# the shared canonical-path hash, so Julia and the other native ports cooperate.
function canonical_file_path(path)
    full=abspath(expanduser(path))
    tail=String[]
    parent=full
    while !ispath(parent)
        pushfirst!(tail, basename(parent))
        next=dirname(parent)
        next==parent && break
        parent=next
    end
    return joinpath(realpath(parent), tail...)
end
function hold_file_lock(f, path; timeout=60.0, env=ENV)
    isfinite(timeout) && timeout >= 0 ||
        throw(ArgumentError("lock timeout must be finite and non-negative"))
    canonical=canonical_file_path(path)
    directory=get(
        env,
        "LM15_LOCK_DIR",
        joinpath(
            get(env, "XDG_CACHE_HOME", joinpath(get(env, "HOME", homedir()), ".cache")),
            "lm15",
            "locks",
        ),
    )
    mkpath(directory)
    lockpath=joinpath(directory, bytes2hex(sha256(canonical))[1:32]*".lock")
    deadline=time_ns()/1e9+timeout
    if Sys.iswindows()
        wide=transcode(UInt16, lockpath*"\0")
        handle=ccall(
            (:CreateFileW, "kernel32"),
            Ptr{Cvoid},
            (Ptr{UInt16}, UInt32, UInt32, Ptr{Cvoid}, UInt32, UInt32, Ptr{Cvoid}),
            wide,
            0xc0000000,
            3,
            C_NULL,
            4,
            0x80,
            C_NULL,
        )
        handle==Ptr{Cvoid}(typemax(UInt)) &&
            throw(NotConfiguredError("cannot open credential lock file"))
        overlap=zeros(UInt8, 32)
        locked=false
        try
            while !locked
                locked=ccall(
                    (:LockFileEx, "kernel32"),
                    Int32,
                    (Ptr{Cvoid}, UInt32, UInt32, UInt32, UInt32, Ptr{UInt8}),
                    handle,
                    3,
                    0,
                    1,
                    0,
                    overlap,
                )!=0
                locked && break
                code=ccall((:GetLastError, "kernel32"), UInt32, ())
                code==33 || throw(NotConfiguredError("cannot acquire credential file lock"))
                time_ns()/1e9 >= deadline && throw(
                    LockTimeoutError(
                        "credential file is locked"; path=canonical, lock_path=lockpath
                    ),
                )
                sleep(0.05)
            end
            f()
        finally
            locked && ccall(
                (:UnlockFileEx, "kernel32"),
                Int32,
                (Ptr{Cvoid}, UInt32, UInt32, UInt32, Ptr{UInt8}),
                handle,
                0,
                1,
                0,
                overlap,
            )
            ccall((:CloseHandle, "kernel32"), Int32, (Ptr{Cvoid},), handle)
        end
    else
        open(lockpath, "a+") do io
            chmod(lockpath, 0o600)
            while ccall(:flock, Cint, (Cint, Cint), fd(io), 6)!=0
                errno=Base.Libc.errno()
                errno in (11, 35, 4) ||
                    throw(NotConfiguredError("cannot acquire credential file lock"))
                time_ns()/1e9 >= deadline && throw(
                    LockTimeoutError(
                        "credential file is locked"; path=canonical, lock_path=lockpath
                    ),
                )
                sleep(0.05)
            end
            try
                f()
            finally
                ccall(:flock, Cint, (Cint, Cint), fd(io), 8)
            end
        end
    end
end
function write_private_json_atomic(path, data)
    check_json(data)
    target=canonical_file_path(path)
    mkpath(dirname(target))
    temp, io=mktemp(dirname(target))
    try
        chmod(temp, 0o600)
        write(io, JSON.serialize(data), "\n")
        flush(io)
        if !Sys.iswindows()
            ccall(:fsync, Cint, (Cint,), fd(io))==0 ||
                throw(SystemError("credential fsync", Base.Libc.errno()))
        end
        close(io)
        if Sys.iswindows()
            ok=ccall(
                (:MoveFileExW, "kernel32"),
                Int32,
                (Ptr{UInt16}, Ptr{UInt16}, UInt32),
                transcode(UInt16, temp*"\0"),
                transcode(UInt16, target*"\0"),
                9,
            )
            ok!=0 || throw(NotConfiguredError("atomic credential replacement failed"))
        else
            Base.Filesystem.rename(temp, target)
            dirfd=ccall(:open, Cint, (Cstring, Cint), dirname(target), 0)
            if dirfd>=0
                try
                    ccall(:fsync, Cint, (Cint,), dirfd)
                finally
                    ccall(:close, Cint, (Cint,), dirfd)
                end
            end
        end
    finally
        isopen(io) && close(io)
        isfile(temp) && rm(temp)
    end
    return nothing
end
function refresh_local(provider, c::LocalOAuthCredential)
    c.refresh_token === nothing &&
        throw(AuthError("expired credential has no refresh token"; provider))
    url, client=if provider=="claude-code"
        ("https://platform.claude.com/v1/oauth/token", "9d1c250a-e61b-44d5-88ed-5944d1962f5e")
    elseif provider=="openai-codex"
        ("https://auth.openai.com/oauth/token", "app_EMoamEEZ73f0CkXaXp7hrann")
    else
        ("https://auth.x.ai/oauth2/token", "b1a00492-073a-47ea-816f-4c329264a828")
    end
    form=obj("grant_type"=>"refresh_token", "client_id"=>client, "refresh_token"=>c.refresh_token)
    json=provider=="claude-code"
    headers=["Content-Type"=>(json ? "application/json" : "application/x-www-form-urlencoded")]
    payload=json ? JSON.serialize(form) : form_encode(form)
    try
        r=http_request(
            "POST",
            url,
            headers,
            payload;
            status_exception=false,
            redirect=false,
            retry=false,
            readtimeout=30,
        )
        200<=r.status<300 || throw(AuthError("credential refresh was rejected"; provider))
        d=JSON.parse(String(r.body))
        token=string_field(d, "access_token")
        token === nothing &&
            throw(AuthError("credential refresh returned no access token"; provider))
        refresh=string_field(d, "refresh_token")
        provider=="claude-code" &&
            refresh===nothing &&
            throw(AuthError("credential refresh returned no refresh token"; provider))
        refresh === nothing && (refresh=c.refresh_token)
        expires=if provider=="openai-codex"
            int_field(something(jwt_payload(token), obj()), "exp")
        else
            nothing
        end
        expiry=if provider=="openai-codex"
            (expires===nothing ? nothing : expires*1000)
        else
            round(Int64, (time()+get(d, "expires_in", 3600))*1000)
        end
        LocalOAuthCredential(token, refresh, expiry, c.account_id)
    catch e
        e isa InterruptException && rethrow()
        throw(
            AuthError(
                "stored credential refresh failed";
                provider,
                credential_hint=if provider=="claude-code"
                    CLAUDE_HINT
                elseif provider=="openai-codex"
                    CODEX_HINT
                else
                    XAI_HINT
                end,
            ),
        )
    end
end
function get_local_credential(provider, path=nothing; env=ENV, refresh_fn=refresh_local)
    home=get(env, "HOME", homedir())
    if provider=="xai"
        credential, source=read_xai_with_source(path; env)
    else
        source=if path===nothing
            joinpath(
                home,
                provider=="claude-code" ? ".claude" : ".codex",
                provider=="claude-code" ? ".credentials.json" : "auth.json",
            )
        else
            path
        end
        credential=stored_credential(provider, source; env)
    end
    is_expired(credential) || return credential
    hold_file_lock(source; env) do
        current=stored_credential(provider, source; env)
        is_expired(current) || return current
        hint = if provider=="claude-code"
            CLAUDE_HINT
        elseif provider=="openai-codex"
            CODEX_HINT
        else
            XAI_HINT
        end
        has_refresh_token(current) || throw(
            AuthError("expired credential has no refresh token"; provider, credential_hint=hint)
        )
        failed = false
        fresh = try
            refresh_fn(provider, current)
        catch error
            error isa InterruptException && rethrow()
            failed = true
            nothing
        end
        failed &&
            throw(AuthError("stored credential refresh failed"; provider, credential_hint=hint))
        fresh isa LocalOAuthCredential && !isempty(fresh.access_token) || throw(
            AuthError(
                "credential refresh returned no usable access token";
                provider,
                credential_hint=hint,
            ),
        )
        d=read_json_object(source, provider, "Log in again")
        section=if provider=="claude-code"
            "claudeAiOauth"
        elseif provider=="openai-codex"
            "tokens"
        else
            "xai"
        end
        raw=get!(d, section, obj())
        if provider=="claude-code"
            raw["accessToken"]=fresh.access_token
            raw["refreshToken"]=fresh.refresh_token
            raw["expiresAt"]=fresh.expires_at_ms
        elseif provider=="openai-codex"
            raw["access_token"]=fresh.access_token
            raw["refresh_token"]=fresh.refresh_token
            fresh.account_id===nothing || (raw["account_id"]=fresh.account_id)
            d["last_refresh"]=Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS")*"Z"
        else
            raw["type"]="oauth"
            raw["access"]=fresh.access_token
            raw["refresh"]=fresh.refresh_token
            raw["expires"]=fresh.expires_at_ms
        end
        write_private_json_atomic(source, d)
        return fresh
    end
end
