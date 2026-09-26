struct ProviderLM{C,T,F}
    provider::String
    dialect::String
    access::AccessPolicy
    credential::C
    base_url::String
    upload_base_url::Maybe{String}
    compat::Union{Nothing,Compat}
    settings::Dict{String,String}
    account_id::Maybe{String}
    transport::T
    clock::F
    credentials_source::Symbol
    adaptations::String
    credential_origin::Maybe{String}
end
function Base.show(io::IO, l::ProviderLM)
    return print(
        io, "ProviderLM(", repr(l.provider), ", ", repr(l.dialect), ", <credentials withheld>)"
    )
end
function bound_definition(provider, access)
    provider isa ProviderDefinition && return provider
    original = provider_definition(provider)
    access === nothing && return original
    access isa AccessPolicy || throw(ArgumentError("access must be an AccessPolicy"))
    validate(access)
    id = canonical_provider(access.provider)
    if id != access.provider
        fields = (; (name => getfield(access, name) for name in fieldnames(AccessPolicy))...)
        access = AccessPolicy(; merge(fields, (; provider=id))...)
    end
    registered = get(PROVIDERS, id, nothing)
    if registered !== nothing
        registered.dialect == original.dialect ||
            throw(ArgumentError("access policy $id uses a different wire dialect"))
    end
    return ProviderDefinition(;
        id,
        dialect=original.dialect,
        access,
        compat=registered === nothing ? nothing : registered.compat,
        placeholder_key=registered === nothing ? nothing : registered.placeholder_key,
    )
end

function ProviderLM(
    provider::Union{AbstractString,ProviderDefinition};
    api_key=nothing,
    base_url=nothing,
    compat=nothing,
    settings=Dict{String,String}(),
    access=nothing,
    credentials_path=nothing,
    account_id=nothing,
    transport=nothing,
    env=nothing,
    clock=time,
    upload_base_url=nothing,
    adaptations="note",
    credential=nothing,
    credential_origin=nothing,
)
    adaptations=check_policy(adaptations)
    named_compat = compat isa AbstractString ? compat : nothing
    definition=bound_definition(provider, access)
    policy=validate(definition.access)
    source=:explicit
    named=credential
    credential=api_key
    # Ordinary direct clients do not select ambient API keys; the router (which
    # passes `env`) enables the ordinary key chain. A cloud door's host settings
    # (region, project, location, endpoint) do follow the environment and the
    # cloud's own configuration on a direct client too, as the cloud SDKs and the
    # TypeScript port do (the Python reference reads them in the router only).
    router_mode = env !== nothing
    env = env === nothing ? Dict(ENV) : copy(env)
    applicable(clock) || throw(ArgumentError("clock must be a zero-argument callable"))
    # An explicit base_url on a cloud door is the endpoint root; the door path is
    # appended unless already present (AUTH-10, amended 2026-09-19).
    endpoint=nothing
    if policy.host!==nothing
        endpoint=base_url===nothing ? first(endpoint_from_env(policy.host, env)) : base_url
        base_url=nothing
    end
    if named!==nothing
        check_named(policy, named)
        credential===nothing || (credential isa AbstractString && isempty(credential)) || throw(NotConfiguredError(
            "$(definition.id): both api_key and credential=$(repr(named)) were given; a door has one identity — pass the credential value, or name the identity, not both";
            provider=definition.id))
    end
    if policy.host!==nothing
        settings=resolve_settings(
            policy, settings; env,
            profile=profile_settings(policy, ChainContext(; env, clock=()->clock())),
            endpoint,
        )
    end
    if endswith(policy.credential_policy, "-chain") && credential===nothing
        credential=cloud_credential_provider(
            policy, ChainContext(; env, settings, clock=()->clock()); named
        )
        source=named===nothing ? :chain : :named
    end
    if credential===nothing
        if policy.credential_policy=="oauth" || (
            policy.credential_policy=="oauth-unless-explicit" &&
            usable_stored(definition.id, credentials_path; env)
        )
            c=stored_credential(policy.provider, credentials_path; env)
            credential=()->ApiKey(
                get_local_credential(policy.provider, credentials_path; env).access_token
            )
            account_id===nothing && (account_id=c.account_id)
            source=:stored
        else
            if policy.credential_policy=="oauth-unless-explicit"
                # R3 (2026-09-22): a failed or signed-out subscription is never
                # silently replaced by a metered environment key.
                state=xai_stored_state(credentials_path; env)
                if state in (:unusable, :logged_out)
                    present=[k for k in policy.env_keys if !isempty(get(env, k, ""))]
                    what=state===:logged_out ? "was signed out" : "is expired and cannot be renewed"
                    throw(MissingCredentialError(
                        "the $(repr(definition.id)) subscription login $what. " *
                        (isempty(present) ? "" : "\$$(first(present)) is set but is used only when passed explicitly: ") *
                        "sign in again, or pass the key deliberately with api_key=... (or RouterConfig(api_keys=...)).";
                        provider=definition.id, env_keys=policy.env_keys, credential_hint=policy.login_hint))
                end
            end
            if router_mode
                for key in policy.env_keys
                    isempty(get(env, key, "")) || (
                        credential=env[key]; credential_origin="env \$$key (value never shown)"; break
                    )
                end
            end
            if credential===nothing && definition.placeholder_key!==nothing
                credential=definition.placeholder_key
                credential_origin="the local server's placeholder key"
            end
        end
    end
    credential===nothing && throw(
        MissingCredentialError(
            if isempty(policy.env_keys)
                "no credential found for provider $(repr(definition.id)); pass api_key=... (a string, a credential value, or a zero-argument function)"
            elseif router_mode
                "no credential found for provider $(repr(definition.id)). Set $(join(policy.env_keys, " or ")) in the environment, or pass RouterConfig(api_keys=Dict(\"$(definition.id)\" => \"...\"))."
            else
                "no credential given for provider $(repr(definition.id)); pass api_key=... (a bare client never reads $(join(policy.env_keys, " or ")); LMRouter does)"
            end;
            provider=definition.id,
            env_keys=policy.env_keys,
            credential_hint=policy.login_hint,
        ),
    )
    credential isa AbstractString &&
        isempty(credential) &&
        throw(NotConfiguredError("empty explicit credential; no fallback"; provider=definition.id))
    (credential isa Union{AbstractString,CredentialValue}) &&
        select_scheme(policy, resolve_credential(credential))
    policy.host===nothing && (settings=Dict{String,String}(settings))
    CT=if definition.dialect=="openai-chat"
        OpenAIChatCompat
    elseif definition.dialect=="openai-responses"
        OpenAIResponsesCompat
    elseif definition.dialect=="anthropic"
        AnthropicCompat
    else
        nothing
    end
    chosen=compat===nothing ? definition.compat : compat
    if chosen isa AbstractString
        CT===nothing && throw(ArgumentError("this dialect has no compatibility presets"))
        compat=preset(CT, chosen)
    elseif chosen===nothing
        compat=CT===nothing ? nothing : CT()
    else
        CT !== nothing && chosen isa CT ||
            throw(ArgumentError("wrong compatibility policy for dialect"))
        compat=chosen
    end
    compat === nothing || validate(compat)
    if base_url===nothing
        if policy.host !== nothing
            base_url = render_base_url(policy.host, settings, endpoint; provider=definition.id)
        elseif named_compat !== nothing
            base_url = preset_url(CT, named_compat)
        elseif policy.base_url !== nothing
            base_url = policy.base_url
        elseif chosen isa AbstractString
            base_url = preset_url(CT, chosen)
        elseif !haskey(PROVIDERS, policy.provider)
            throw(
                NotConfiguredError(
                    "a custom access policy needs base_url"; provider=policy.provider
                ),
            )
        else
            base_url = if definition.dialect=="gemini"
                "https://generativelanguage.googleapis.com/v1beta"
            elseif definition.dialect=="typesafe"
                TYPESAFE_BASE_URL
            elseif definition.dialect=="anthropic"
                "https://api.anthropic.com/v1"
            else
                "https://api.openai.com/v1"
            end
        end
    end
    uri=HTTP.URI(base_url)
    uri.scheme in ("http", "https") &&
    !isempty(uri.host) &&
    isempty(uri.userinfo) &&
    isempty(uri.query) &&
    isempty(uri.fragment) || throw(
        ArgumentError("base_url must be an HTTP(S) root without credentials, query, or fragment"),
    )
    if upload_base_url === nothing && base_url == "https://generativelanguage.googleapis.com/v1beta"
        upload_base_url = "https://generativelanguage.googleapis.com/upload/v1beta"
    end
    if policy.backend=="chatgpt-codex" && account_id===nothing
        if credential isa Union{AbstractString,ApiKey,BearerToken}
            token=access_token(resolve_credential(credential))
            claim=getobject(something(jwt_payload(token), obj()), "https://api.openai.com/auth")
            account_id=string_field(claim, "chatgpt_account_id")
        end
        account_id===nothing && throw(
            NotConfiguredError(
                "Codex requires a ChatGPT account id";
                provider=definition.id,
                credential_hint=CODEX_HINT,
            ),
        )
    end
    return ProviderLM(
        definition.id,
        definition.dialect,
        policy,
        credential,
        String(rstrip(base_url, '/')),
        upload_base_url === nothing ? nothing : String(rstrip(upload_base_url, '/')),
        compat,
        settings,
        account_id,
        transport,
        clock,
        source,
        adaptations,
        credential_origin===nothing ? origin_label(credential, source) : credential_origin,
    )
end
"""The provenance label for a credential the client was handed (AUTH-1): honest about
what LM15 can see; a callable's identity is not inspected. Never the value."""
function origin_label(credential, source)
    (credential===nothing || credential=="") && return "no credential"
    source===:stored && return "the stored local login (credentials file)"
    credential isa CloudCredentialProvider && return nothing
    credential isa Union{AbstractString,CredentialValue} || return "an application-supplied callable (identity not inspected by lm15)"
    return "an explicit api_key (value never shown)"
end
"""
    credential_origin(client)

Where this client's credential comes from, as a phrase with no secret in it (AUTH-1
provenance). For a cloud chain, the rung that last answered, or what will be walked
before the first request.
"""
function credential_origin(l::ProviderLM)
    c=l.credential
    if c isa CloudCredentialProvider
        c.source===nothing || return describe(c.source; now=l.clock())
        c.named===nothing ||
            return "named credential \"$(c.named)\" ($(named_meaning(l.access, c.named)); not yet resolved)"
        return "the $(l.access.credential_policy) (not yet resolved)"
    end
    return something(l.credential_origin, origin_label(c, l.credentials_source))
end
OpenAILM(; kw...) = ProviderLM("openai"; kw...)
OpenAIChatLM(; kw...) = ProviderLM("openai-chat"; kw...)
AnthropicLM(; kw...) = ProviderLM("anthropic"; kw...)
GeminiLM(; kw...) = ProviderLM("gemini"; kw...)
XaiLM(; kw...) = ProviderLM("xai"; kw...)
ClaudeCodeLM(; kw...) = ProviderLM("claude-code"; kw...)
OpenAICodexLM(; kw...) = ProviderLM("openai-codex"; kw...)
function require_surface(l, s::Symbol)
    return supports_endpoint(l.access.supports, s) || unsupported(l.provider, string(s))
end
function effective_compat(l, request)
    override=nothing
    if l.compat isa OpenAIResponsesCompat
        ext=something(request.config.extensions, obj())
        override=first_nonempty(
            get(ext, "openai_responses_compat", nothing),
            get(ext, "openai_compat", nothing),
            get(getobject(ext, "compat"), "openai_responses", nothing),
            get(getobject(ext, "compat"), "openai", nothing),
        )
        override===nothing || compat_from_dict(OpenAIResponsesCompat, override)
    end
    return resolved_compat(l.compat, request.model; override)
end
function base_headers(l; request=nothing, content_type="application/json")
    headers=Dict{String,String}()
    content_type===nothing || (headers["content-type"]=content_type)
    l.dialect=="anthropic" && (headers["anthropic-version"]="2023-06-01")
    for (k, v) in l.access.headers
        headers[lowercase(k)]=v
    end
    if l.dialect=="anthropic" &&
        request!==nothing &&
        any(t->t isa BuiltinTool && t.name=="code_execution", request.tools)
        headers["anthropic-beta"]=join(
            filter(!isempty, [get(headers, "anthropic-beta", ""), "code-execution-2025-05-22"]), ","
        )
    end
    l.access.backend=="chatgpt-codex" &&
        l.account_id!==nothing &&
        (headers["chatgpt-account-id"]=l.account_id)
    return headers
end
function emit(
    l;
    method="POST",
    url,
    payload=nothing,
    body=UInt8[],
    headers=base_headers(l),
    params=obj(),
    endpoint=nothing,
    stream=false,
    model="",
)
    credential=resolve_credential(l.credential) # AUTH-2: exactly once per built request.
    scheme=select_scheme(l.access, credential)
    headers=copy(headers)
    params=copy(params)
    merge!(
        headers,
        credential_headers(
            l.access,
            credential;
            api_key_header=l.dialect=="gemini" ? "x-goog-api-key" : "x-api-key",
        ),
    )
    scheme=="query-key" && (params["key"]=credential.value)
    host=l.access.host
    if host!==nothing
        key=if stream && haskey(host.paths, string(endpoint)*"/stream")
            string(endpoint)*"/stream"
        else
            endpoint
        end
        if key!==nothing && haskey(host.paths, key)
            url=l.base_url*replace(host.paths[key], "{model}"=>percent_encode(model; safe=":@"))
        end
        if payload isa AbstractDict
            payload=copy(payload)
            host.model_in=="path" && delete!(payload, "model")
            if startswith(host.anthropic_version_in, "body:")
                payload["anthropic_version"]=split(host.anthropic_version_in, ':'; limit=2)[2]
                delete!(headers, "anthropic-version")
            end
        end
        for (header, setting) in host.required_headers
            headers[lowercase(header)]=l.settings[setting]
        end
        stream &&
            host.stream_framing!="sse" &&
            unsupported(l.provider, "$(host.stream_framing) framing")
    end
    if payload!==nothing
        body=Vector{UInt8}(codeunits(wire_json(payload)))
        get!(headers, "content-type", "application/json")
    end
    url=with_query(url, params)
    if credential isa AwsCredentials
        host===nothing && throw(NotConfiguredError("AWS credentials require a signed host"))
        signature=sigv4_sign(
            method,
            url,
            headers,
            body,
            credential,
            l.settings["region"],
            host.sigv4_service;
            now=l.clock(),
        )
        headers=signature.headers
    end
    return WireRequest(
        method, url, Pair{String,String}[String(k)=>String(v) for (k, v) in headers], body
    )
end
function retry_after_seconds(value)
    value===nothing && return nothing
    seconds=tryparse(Float64, strip(value))
    if seconds===nothing
        seconds=try
            datetime2unix(DateTime(strip(value), dateformat"e, dd u yyyy HH:MM:SS \G\M\T"))-time()
        catch
            nothing
        end
        seconds===nothing || (seconds=max(0.0, seconds))
    end
    return seconds!==nothing && isfinite(seconds) && seconds>=0 ? seconds : nothing
end
function milliseconds_seconds(value)
    value===nothing && return nothing
    ncodeunits(value)<=256 && occursin(r"^\+?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$", strip(value)) || return nothing
    n=parse(Float64, strip(value))
    return isfinite(n) && n>=0 ? n/1000 : nothing
end
const REQUEST_ID_HEADERS=("x-request-id", "request-id", "x-amzn-requestid", "x-amz-request-id",
    "x-ms-request-id", "apim-request-id", "x-typesafe-request-id")
"""
Fill HTTP diagnostics the body did not give (docs/error-diagnostics.md): the bounded
rate-limit header snapshot, a retry hint (body value, else the first Retry-After, else
retry-after-ms, else x-ms-retry-after-ms), and the request id. Never invents a field.
"""
function attach_error_metadata(e::LM15Error, headers)
    headers=headers isa HttpResponse ? headers.headers : headers
    kw=Dict{Symbol,Any}()
    snapshot=capture_rate_limits(headers)
    isempty(snapshot) || (kw[:rate_limit_headers]=snapshot)
    first_header(name)=begin
        for (k, v) in headers
            lowercase(k)==name && return String(v)
        end
        nothing
    end
    if e.retry_after===nothing
        seconds=retry_after_seconds(first_header("retry-after"))
        for name in ("retry-after-ms", "x-ms-retry-after-ms")
            seconds===nothing || break
            seconds=milliseconds_seconds(first_header(name))
        end
        seconds===nothing || (kw[:retry_after]=seconds)
    end
    if e.request_id===nothing
        for name in REQUEST_ID_HEADERS
            v=first_header(name)
            v===nothing || isempty(v) || (kw[:request_id]=v; break)
        end
    end
    return isempty(kw) ? e : with_metadata(e; kw...)
end
"""The handshake evidence for an error event inside a successful stream: request id,
retry hint and rate-limit headers (never a status: the handshake's 200 is not the error's)."""
function stream_http_response(headers)
    probe=attach_error_metadata(GenericProviderError(""), headers)
    http=JSONObject()
    probe.request_id===nothing || (http["request_id"]=probe.request_id)
    probe.retry_after===nothing || (http["retry_after"]=probe.retry_after)
    probe.rate_limit_headers===nothing || isempty(probe.rate_limit_headers) ||
        (http["rate_limit_headers"]=JSONObject(k=>collect(v) for (k, v) in probe.rate_limit_headers))
    return http
end
"""
The body of a success reply as JSON. A body that is not JSON (a gateway's HTML page behind
a 200, a truncated reply) is a `ProviderError` carrying the status, content type, the first
200 bytes and the request id (INV-054); never retried, never a ServerError.
"""
function reply_json(l, r::HttpResponse)
    text=String(copy(r.body))
    failure=nothing
    if isvalid(text)
        try
            return JSON.parse(text)
        catch e
            e isa InterruptException && rethrow()
            failure=e
        end
    end
    content_type=something(header(r, "content-type"), "no content-type")
    excerpt=first(isvalid(text) ? text : String(map(c->isvalid(c) ? c : '\ufffd', collect(text))), 200)
    length(text)>200 && (excerpt*="…")
    throw(attach_error_metadata(GenericProviderError(
        "the reply (HTTP $(r.status), $content_type, $(length(r.body)) bytes) is not JSON. Body starts: $(repr(excerpt)). A gateway or proxy in front of the provider is the usual cause; the request may or may not have been served.";
        provider=l.provider, status=r.status), r))
end
function send_request(l, r::WireRequest)
    open_response(client_transport(l), r) do head, io
        body = try
            read(io)
        catch error
            error isa InterruptException && rethrow()
            throw(TransportError("provider response could not be read"; provider=l.provider))
        end
        response = HttpResponse(; status=head.status, headers=head.headers, body)
        response.status >= 400 && throw(
            attach_error_metadata(
                normalize_error(l, response.status, String(copy(body))), response
            ),
        )
        return response
    end
end
function complete(l::ProviderLM, request::Request)
    require_surface(l, :complete)
    validate(request)
    judgments_via_token_scoring(l, request) && return judgment_complete(l, request)
    l.access.backend=="chatgpt-codex" && return materialize_response(stream(l, request), request)
    wire, records=build_request_adapted(l, request; stream=false)
    # MAP-13: a stop sequence the wire cannot take is honoured by streaming and
    # closing the connection at the cut; the usage report is then not reported.
    client_side_stop(records) && return materialize_response(stream(l, request), request)
    response=parse_response(l, request, send_request(l, wire))
    visible=visible_adaptations(l, records)
    return isempty(visible) || !isempty(response.adaptations) ? response : reconstruct(response; adaptations=visible)
end
function build_models_request(l::ProviderLM)
    require_surface(l, :models)
    params=if l.access.backend=="chatgpt-codex"
        obj("client_version"=>get(l.access.backend_options, "client_version", ""))
    elseif l.dialect=="anthropic"
        obj("limit"=>1000)
    elseif l.dialect=="gemini"
        obj("pageSize"=>1000)
    else
        obj()
    end
    return emit(
        l;
        method="GET",
        url=l.base_url*(l.dialect=="typesafe" ? "/v1/models" : "/models"),
        params,
        headers=base_headers(l; content_type=l.dialect=="gemini" ? nothing : "application/json"),
    )
end
function parse_models_response(l::ProviderLM, r::HttpResponse)
    r.status>=400 &&
        throw(attach_error_metadata(normalize_error(l, r.status, String(copy(r.body))), r))
    data=reply_json(l, r)
    codex=l.access.backend=="chatgpt-codex"
    entries=if l.dialect=="openai-chat"
        # A chat server answers {"data": [...]} or a bare array (Together); anything
        # else is a malformed reply, never an empty catalog (2026-09-26).
        if data isa AbstractVector
            data
        elseif data isa AbstractDict && get(data, "data", nothing) isa AbstractVector
            data["data"]
        else
            throw(GenericProviderError("model listing is neither an array nor an object with a data array"; provider=l.provider))
        end
    else
        getarray(data, codex || l.dialect in ("gemini", "typesafe") ? "models" : "data")
    end
    family=if l.dialect=="openai-responses"
        "openai_responses"
    elseif l.dialect=="openai-chat"
        "openai_chat"
    elseif l.dialect=="anthropic"
        "anthropic_messages"
    elseif l.dialect=="typesafe"
        "typesafe_systemone"
    else
        "gemini_generate_content"
    end
    out=ModelInfo[]
    for entry in entries
        entry isa AbstractDict || continue
        id=string_field(
            entry,
            if codex
                "slug"
            elseif l.dialect in ("gemini", "typesafe")
                "name"
            else
                "id"
            end,
        )
        id===nothing && continue
        l.dialect=="gemini" && (id=replace(id, r"^models/"=>""))
        push!(
            out,
            ModelInfo(;
                id,
                provider=l.provider,
                api_family=family,
                origin=ModelOrigin(; provider_data=entry),
            ),
        )
    end
    return out
end
list_models(l::ProviderLM) = parse_models_response(l, send_request(l, build_models_request(l)))
Base.close(::ProviderLM) = nothing
