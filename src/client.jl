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
end
function Base.show(io::IO, l::ProviderLM)
    return print(
        io, "ProviderLM(", repr(l.provider), ", ", repr(l.dialect), ", <credentials withheld>)"
    )
end
function bound_definition(provider, access)
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
    provider::AbstractString;
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
)
    named_compat = compat isa AbstractString ? compat : nothing
    definition=bound_definition(provider, access)
    policy=validate(definition.access)
    source=:explicit
    credential=api_key
    # Ordinary direct clients do not select ambient API keys. Cloud profiles
    # and stored-login path overrides still need the real environment to pick
    # the correct identity; the router explicitly enables the ordinary chain.
    resolve_environment_keys = env !== nothing
    env = env === nothing ? Dict(ENV) : copy(env)
    applicable(clock) || throw(ArgumentError("clock must be a zero-argument callable"))
    if endswith(policy.credential_policy, "-chain")
        settings=resolve_settings(policy, cloud_profile_settings(policy, settings, env); env)
        credential===nothing && (
            credential=cloud_credential_provider(
                policy, ChainContext(; env, settings, clock=()->clock())
            )
        )
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
            if resolve_environment_keys
                for key in policy.env_keys
                    isempty(get(env, key, "")) || (credential=env[key]; break)
                end
            end
            credential===nothing && (credential=definition.placeholder_key)
        end
    end
    credential===nothing && throw(
        NotConfiguredError(
            "no credential found; configure an explicit key or token provider";
            provider=definition.id,
            credential_hint=policy.login_hint,
        ),
    )
    credential isa AbstractString &&
        isempty(credential) &&
        throw(NotConfiguredError("empty explicit credential; no fallback"; provider=definition.id))
    (credential isa Union{AbstractString,CredentialValue}) &&
        select_scheme(policy, resolve_credential(credential))
    settings=resolve_settings(policy, settings; env)
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
            base_url = host_base_url(policy.host, settings)
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
    )
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
        check_json(payload)
        body=Vector{UInt8}(codeunits(JSON.serialize(payload)))
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
function attach_error_metadata(e::LM15Error, r::HttpResponse)
    kw=Dict{Symbol,Any}(n=>getfield(e.metadata, n) for n in fieldnames(ErrorMetadata))
    if e.request_id===nothing
        for key in (
            "x-request-id", "request-id", "x-amzn-requestid", "x-amz-request-id", "x-ms-request-id"
        )
            v=header(r, key)
            v===nothing || (kw[:request_id]=v; break)
        end
    end
    if e.retry_after===nothing
        value=header(r, "retry-after")
        if value!==nothing
            seconds=tryparse(Float64, value)
            if seconds===nothing
                try
                    seconds=max(
                        0.0,
                        datetime2unix(DateTime(value, dateformat"e, dd u yyyy HH:MM:SS GMT"))-time(),
                    )
                catch
                end
            end
            seconds===nothing || !isfinite(seconds) || seconds<0 || (kw[:retry_after]=seconds)
        end
    end
    return typeof(e)(ErrorMetadata(; kw...))
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
    l.access.backend=="chatgpt-codex" && return materialize_response(stream(l, request), request)
    return parse_response(l, request, send_request(l, build_request(l, request; stream=false)))
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
        url=l.base_url*"/models",
        params,
        headers=base_headers(l; content_type=l.dialect=="gemini" ? nothing : "application/json"),
    )
end
function parse_models_response(l::ProviderLM, r::HttpResponse)
    r.status>=400 &&
        throw(attach_error_metadata(normalize_error(l, r.status, String(copy(r.body))), r))
    data=JSON.parse(String(copy(r.body)))
    codex=l.access.backend=="chatgpt-codex"
    entries=getarray(data, codex || l.dialect=="gemini" ? "models" : "data")
    family=if l.dialect=="openai-responses"
        "openai_responses"
    elseif l.dialect=="openai-chat"
        "openai_chat"
    elseif l.dialect=="anthropic"
        "anthropic_messages"
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
            elseif l.dialect=="gemini"
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
