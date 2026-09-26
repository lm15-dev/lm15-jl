# Thin JSONL adapter for the external contract harness. No network driver is
# called here; build and parse use the same public code as applications.
const VET_OPS=(
    "capabilities",
    "build_request",
    "parse_response",
    "replay_stream",
    "normalize_error",
    "serde_roundtrip",
    "validate",
    "surface_dump",
    "explain_auth",
    "resolve_model",
    "sigv4_sign",
    "token_exchange_build",
    "token_exchange_parse",
    "build_models_request",
    "parse_models_response",
    "file_op_build",
    "file_op_parse",
    "cache_op_build",
    "cache_op_parse",
    "generation_build",
    "generation_parse",
    "batch_op_build",
    "batch_op_parse",
    "video_op_build",
    "video_op_parse",
    "replay_live",
    "ingest_openai_chat",
)
function wire_dict(r::WireRequest)
    uri=HTTP.URI(r.url)
    params=obj()
    for pair in split(uri.query, '&'; keepempty=false)
        bits=split(pair, '='; limit=2)
        decode(s) = HTTP.URIs.unescapeuri(replace(s, "+"=>" "))
        params[decode(bits[1])]=decode(length(bits)==2 ? bits[2] : "")
    end
    url=first(split(r.url, '?'; limit=2))
    headers=obj((lowercase(k)=>v for (k, v) in r.headers)...)
    out=obj("method"=>r.method, "url"=>url, "params"=>params, "headers"=>headers, "body"=>nothing)
    if !isempty(r.body)
        if occursin("json", lowercase(get(headers, "content-type", "")))
            out["body"]=JSON.parse(String(copy(r.body)))
        else
            out["body_b64"]=base64encode(r.body)
        end
    end
    return out
end
function vet_adapter(msg; parse_only=false)
    provider=msg["provider"]
    credential=if parse_only
        ApiKey("vet-parse-only")
    elseif haskey(msg, "credential")
        from_dict(CredentialValue, msg["credential"])
    else
        ApiKey(get(msg, "api_key", "vet-parse-only"))
    end
    clock=haskey(msg, "now") ? (()->expiry_seconds(msg["now"])) : time
    return ProviderLM(
        provider;
        api_key=credential,
        base_url=get(msg, "base_url", nothing),
        settings=get(msg, "settings", Dict{String,String}()),
        clock,
        account_id=canonical_provider(provider)=="openai-codex" ? "test-account" : nothing,
        # The harness's own (empty) environment: nothing ambient is read.
        env=Dict{String,String}("NO_GCE_CHECK"=>"1", "HOME"=>"/nonexistent-lm15-vet-home"),
    )
end
function vet_http(msg; bodykey="body_b64", status=Int(get(msg, "status", 200)))
    return HttpResponse(;
        status,
        headers=Pair{String,String}[String(k)=>String(v) for (k, v) in get(msg, "headers", obj())],
        body=base64decode(get(msg, bodykey, "")),
    )
end
function vet_response(r::Response)
    out=obj("canonical_response"=>to_dict(r))
    data=something(r.provider_data, obj())
    haskey(data, "_lm15_unmapped") && (out["unmapped"]=data["_lm15_unmapped"])
    return out
end
function token_exchange_build(provider, rung, inputs, ctx::ChainContext)
    if rung in ("adc-env", "adc-file", "service-account")
        url, assertion=gcp_assertion(
            ctx, inputs["credential_file"]; scope=get(inputs, "scope", GCP_SCOPE)
        )
        fields=obj("grant_type"=>JWT_BEARER, "assertion"=>assertion)
    elseif rung=="environment"
        url, fields=azure_environment_fields(ctx; jti=get(inputs, "jti", nothing))
    else
        unsupported(provider, "deterministic token exchange for rung $rung")
    end
    return obj(
        "method"=>"POST",
        "url"=>url,
        "headers"=>obj("content-type"=>"application/x-www-form-urlencoded"),
        "body_encoding"=>"form",
        "body"=>fields,
    )
end
function token_exchange_parse(provider, rung, status, body, ctx::ChainContext)
    if rung=="credential_process"
        status==0 && get(body, "Version", nothing)==1 ||
            throw(AuthError("credential command failed or returned unsupported Version"; provider))
        return aws_credential(body)
    end
    200<=status<300 || throw(AuthError("credential endpoint rejected request"; provider))
    return rung in ("imds", "container") ? aws_credential(body) : oauth_credential(body, ctx)
end
function vet_operation(msg)
    op=msg["op"]
    if op=="capabilities"
        return obj("language"=>"julia", "ops"=>sort!(collect(VET_OPS)), "impl_version"=>"0.3.0-dev")
    elseif op in ("serde_roundtrip", "validate")
        value=to_dict(from_dict(msg["kind"], msg["value"]))
        return op=="validate" ? obj("ok"=>true, "normalized"=>value) : obj("value"=>value)
    elseif op=="resolve_model"
        registry=haskey(msg, "catalog") ? ModelRegistry(msg["catalog"]) : nothing
        resolution=resolve(
            LMRouter(RouterConfig(; registry, env=get(msg, "env", Dict()))), msg["model"]
        )
        return obj(
            "provider"=>resolution.provider, "model"=>resolution.model, "source"=>resolution.source
        )
    elseif op=="explain_auth"
        provider=canonical_provider(msg["provider"])
        path=get(msg, "credentials_path", nothing)
        report=explain_auth(
            provider;
            env=get(msg, "env", Dict()),
            api_keys=Dict(k=>ApiKey(msg["sentinel"]) for k in get(msg, "api_keys_providers", [])),
            claude_credentials_path=provider=="claude-code" ? path : nothing,
            codex_auth_path=provider=="openai-codex" ? path : nothing,
            xai_credentials_path=provider=="xai" ? path : nothing,
            settings=get(msg, "settings", Dict()),
            files=get(msg, "files", nothing),
        )
        reply=obj(
            "configured"=>report.configured,
            "steps"=>[obj("kind"=>s.kind, "state"=>string(s.state)) for s in report.steps],
            "report_text"=>describe(report),
        )
        if !isempty(report.settings_from)
            reply["settings"]=obj((
                k=>(v.state===nothing ? obj("value"=>v.value, "from"=>v.from) :
                    obj("value"=>v.value, "from"=>v.from, "state"=>v.state))
                for (k, v) in report.settings_from)...)
        end
        return reply
    elseif op=="surface_dump"
        types=obj(
            (
                name=>obj("fields"=>string.(collect(fieldnames(T)))) for
                (name, T) in CANONICAL_TYPES
            )...,
        )
        enums=obj((string(key)=>collect(values) for (key, values) in VOCABULARIES)...)
        providers=obj(
            (
                p.id=>obj(
                    "supports"=>obj(
                        (
                            string(key)=>getfield(p.access.supports, key) for
                            key in fieldnames(EndpointSupport)
                        )...,
                    ),
                    "auth_modes"=>collect(p.access.auth_modes),
                    "env_keys"=>collect(p.access.env_keys),
                ) for p in values(PROVIDERS)
            )...,
        )
        return obj("types"=>types, "enums"=>enums, "providers"=>providers)
    elseif op=="sigv4_sign"
        r=msg["request"]
        headers=Dict(
            String(k)=>(v isa AbstractVector ? join(v, ",") : String(v)) for
            (k, v) in get(r, "headers", obj())
        )
        c=from_dict(CredentialValue, msg["credential"])
        c isa AwsCredentials || throw(ArgumentError("sigv4 needs AWS credentials"))
        signature=sigv4_sign(
            r["method"],
            r["url"],
            headers,
            codeunits(get(r, "body", "")),
            c,
            msg["region"],
            msg["service"];
            now=expiry_seconds(msg["now"]),
        )
        return obj(
            "canonical_request"=>signature.canonical_request,
            "string_to_sign"=>signature.string_to_sign,
            "authorization"=>signature.authorization,
            "headers"=>signature.headers,
        )
    elseif op in ("token_exchange_build", "token_exchange_parse")
        inputs=get(msg, "input", get(msg, "credential", obj()))
        env=get(inputs, "env", Dict())
        files=Dict{String,String}()
        if haskey(inputs, "certificate_pem") && haskey(env, "AZURE_CLIENT_CERTIFICATE_PATH")
            files[env["AZURE_CLIENT_CERTIFICATE_PATH"]]=inputs["certificate_pem"]*"\n"*get(
                inputs, "private_key_pem", ""
            )
        end
        ctx=ChainContext(;
            env,
            files,
            settings=get(inputs, "settings", get(msg, "settings", Dict())),
            clock=()->expiry_seconds(msg["now"]),
        )
        op=="token_exchange_build" &&
            return token_exchange_build(msg["provider"], msg["rung"], inputs, ctx)
        body=get(msg, "body", nothing)
        body===nothing && (body=JSON.parse(String(base64decode(msg["body_b64"]))))
        try
            c=token_exchange_parse(
                msg["provider"], msg["rung"], Int(get(msg, "status", 200)), body, ctx
            )
            return obj("ok"=>true, "credential"=>to_dict(c))
        catch e
            e isa LM15Error || rethrow()
            return obj("ok"=>false, "error"=>obj("class"=>class_name(e), "code"=>e.code))
        end
    end
    l=vet_adapter(
        msg;
        parse_only=(!(
            op in (
                "build_request",
                "build_models_request",
                "file_op_build",
                "cache_op_build",
                "generation_build",
                "batch_op_build",
                "video_op_build",
            )
        )),
    )
    if op=="build_request"
        wire, records=build_request_adapted(
            l, from_dict(Request, msg["canonical_request"]); stream=get(msg, "stream", false)
        )
        out=wire_dict(wire)
        isempty(records) || (out["adaptations"]=[
            merge(obj("field"=>a.field, "action"=>a.action),
                a.asked===nothing ? obj() : obj("asked"=>a.asked),
                a.applied===nothing ? obj() : obj("applied"=>a.applied))
            for a in records])
        return out
    elseif op=="parse_response"
        return vet_response(
            parse_response(l, from_dict(Request, msg["canonical_request"]), vet_http(msg))
        )
    elseif op=="replay_stream"
        request=from_dict(Request, msg["canonical_request"])
        raw=StreamEvent[]
        parse_sse(IOBuffer(base64decode(msg["body_b64"]))) do frame
            return append!(raw, parse_stream_events(l, request, frame))
        end
        events=collect(coalesce_stream(raw; model=request.model))
        trace=to_dict.(events)
        try
            response=materialize_response(events, request)
            return merge(obj("events"=>trace), vet_response(response))
        catch e
            e isa StreamAssemblyError || rethrow()
            throw(VetFailure(e, obj("events"=>trace)))
        end
    elseif op=="normalize_error"
        e=normalize_error(l, Int(msg["status"]), msg["body_text"])
        return obj(
            "class"=>class_name(e),
            "code"=>e.code,
            "provider_code"=>e.provider_code,
            "message"=>e.message,
        )
    elseif op=="ingest_openai_chat"
        return obj("canonical_request"=>to_dict(request_from_openai_chat(l, msg["body"])))
    elseif op=="build_models_request"
        return wire_dict(build_models_request(l))
    elseif op=="parse_models_response"
        return obj("models"=>to_dict.(parse_models_response(l, vet_http(msg))))
    elseif op=="file_op_build"
        request=if haskey(msg, "upload_request")
            from_dict(FileUploadRequest, msg["upload_request"])
        else
            nothing
        end
        return wire_dict(
            build_file_request(
                l,
                Symbol(msg["file_op"]);
                request,
                id=get(msg, "file_id", nothing),
                limit=Int(get(msg, "limit", 20)),
                cursor=get(msg, "cursor", nothing),
            ),
        )
    elseif op=="file_op_parse"
        page=msg["kind"]=="page"
        value=parse_file_response(l, vet_http(msg); page)
        return obj((page ? "page" : "file")=>to_dict(value))
    elseif op=="cache_op_build"
        prefix=haskey(msg, "prefix_request") ? from_dict(Request, msg["prefix_request"]) : nothing
        return wire_dict(
            build_cache_request(
                l,
                Symbol(msg["cache_op"]);
                prefix,
                id=get(msg, "cache_id", nothing),
                ttl_seconds=get(msg, "ttl_seconds", nothing),
                label=get(msg, "label", nothing),
                limit=Int(get(msg, "limit", 20)),
                cursor=get(msg, "cursor", nothing),
            ),
        )
    elseif op=="cache_op_parse"
        page=msg["kind"]=="page"
        value=parse_cache_response(l, vet_http(msg); page)
        return obj((page ? "page" : "cache")=>to_dict(value))
    elseif op in ("generation_build", "generation_parse")
        request=from_dict(
            msg["kind"]=="image" ? ImageGenerationRequest : SpeechGenerationRequest,
            msg["generation_request"],
        )
        return if op=="generation_build"
            wire_dict(build_generation_request(l, request))
        else
            to_dict(parse_generation_response(l, request, vet_http(msg)))
        end
    elseif op=="batch_op_build"
        request=if haskey(msg, "batch_request")
            from_dict(BatchRequest, msg["batch_request"])
        else
            nothing
        end
        requests=build_batch_requests(
            l,
            Symbol(msg["action"]);
            request,
            id=get(msg, "batch_id", nothing),
            limit=Int(get(msg, "limit", 20)),
            upload_body=get(msg, "upload_body", nothing),
            status_body=get(msg, "status_body", nothing),
        )
        return obj("requests"=>wire_dict.(requests))
    elseif op=="batch_op_parse"
        kind=msg["kind"]
        kind=="entries" && return obj(
            "entries"=>to_dict.(
                parse_batch_entries(
                    l,
                    msg["status_body"],
                    [String(base64decode(s)) for s in get(msg, "fetched_b64", [])],
                ),
            ),
        )
        result=parse_batch_response(l, vet_http(msg); list=kind=="list")
        return kind=="list" ? obj("jobs"=>to_dict.(result)) : obj("job"=>to_dict(result))
    elseif op=="video_op_build"
        request=if haskey(msg, "video_request")
            from_dict(VideoGenerationRequest, msg["video_request"])
        else
            nothing
        end
        requests=build_video_requests(
            l,
            Symbol(msg["action"]);
            request,
            id=get(msg, "video_id", nothing),
            status_body=get(msg, "status_body", nothing),
            limit=Int(get(msg, "limit", 20)),
            model=get(msg, "model", nothing),
        )
        return obj("requests"=>wire_dict.(requests))
    elseif op=="video_op_parse"
        kind=msg["kind"]
        if kind=="part"
            fetched=if get(msg, "fetched_b64", nothing)===nothing
                nothing
            else
                vet_http(msg; bodykey="fetched_b64")
            end
            return obj("part"=>to_dict(parse_video_part(l, msg["status_body"], fetched)))
        end
        result=parse_video_response(
            l, vet_http(msg); list=kind=="list", id=get(msg, "video_id", nothing)
        )
        return kind=="list" ? obj("jobs"=>to_dict.(result)) : obj("job"=>to_dict(result))
    elseif op=="replay_live"
        config=from_dict(LiveConfig, msg["live_config"])
        return obj(
            "setup_frames"=>live_setup_frames(l, config),
            "client_frames"=>[
                encode_live_event(l, config, from_dict(LiveClientEvent, e)) for
                e in get(msg, "client_events", [])
            ],
            "events"=>[
                to_dict.(decode_live_frame(l, base64decode(frame))) for
                frame in get(msg, "server_frames_b64", [])
            ],
        )
    end
    return throw(ArgumentError("unknown vet operation"))
end
struct VetFailure <: Exception
    cause::Exception
    extra::JsonObject
end
function vet_handle_line(line)
    id=nothing
    try
        msg=JSON.parse(line)
        msg isa AbstractDict || throw(ArgumentError("protocol line must be an object"))
        id=get(msg, "id", nothing)
        return obj("id"=>id, "ok"=>true, "result"=>vet_operation(msg))
    catch caught
        extra=caught isa VetFailure ? caught.extra : obj()
        e=caught isa VetFailure ? caught.cause : caught
        error=obj(
            "type"=>e isa LM15Error ? class_name(e) : string(nameof(typeof(e))),
            "message"=>sprint(showerror, e),
        )
        if e isa LM15Error
            error["code"]=e.code
            e.feature===nothing || (error["feature"]=e.feature)
            e.partial===nothing || (error["partial_response"]=to_dict(e.partial))
            if e isa Union{UnknownModelError,AmbiguousModelError}
                error["model"]=e.model
                e isa AmbiguousModelError && (error["providers"]=collect(e.providers))
            end
        end
        merge!(error, extra)
        return obj("id"=>id, "ok"=>false, "error"=>error)
    end
end
function vet_main(input::IO=stdin, output::IO=stdout)
    for line in eachline(input)
        isempty(strip(line)) && continue
        println(output, JSON.serialize(vet_handle_line(line)))
        flush(output)
    end
    return nothing
end
