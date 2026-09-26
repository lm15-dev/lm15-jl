function multipart_form(fields, files)
    boundary="lm15-"*randstring(32)
    io=IOBuffer()
    for (name, value) in fields
        any(c->c in ('\r', '\n', '"'), name) && throw(ArgumentError("unsafe multipart field name"))
        write(
            io,
            "--$boundary\r\nContent-Disposition: form-data; name=\"$name\"\r\n\r\n",
            value,
            "\r\n",
        )
    end
    for (name, filename, mime, data) in files
        any(c->c in ('\r', '\n'), filename*name*mime) &&
            throw(ArgumentError("unsafe multipart header"))
        filename=replace(filename, '"'=>"%22")
        write(
            io,
            "--$boundary\r\nContent-Disposition: form-data; name=\"$name\"; filename=\"$filename\"\r\nContent-Type: $mime\r\n\r\n",
            data,
            "\r\n",
        )
    end
    write(io, "--$boundary--\r\n")
    return "multipart/form-data; boundary=$boundary", take!(io)
end
function multipart_related(metadata, mime, data)
    occursin(r"[\r\n]", mime) && throw(ArgumentError("unsafe multipart media type"))
    boundary="lm15-"*randstring(32)
    io=IOBuffer()
    write(
        io,
        "--$boundary\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n",
        wire_json(metadata),
        "\r\n--$boundary\r\nContent-Type: $mime\r\n\r\n",
        data,
        "\r\n--$boundary--\r\n",
    )
    return "multipart/related; boundary=$boundary", take!(io)
end
function file_resource(id)
    if occursin("://", id)
        parts=split(rstrip(id, '/'), "/files/"; limit=2)
        length(parts)==2 && !isempty(parts[2]) && return "files/"*parts[2]
    end
    return startswith(id, "files/") ? id : "files/"*id
end
cache_resource(id) = startswith(id, "cachedContents/") ? id : "cachedContents/"*id
function build_file_request(
    l::ProviderLM, operation::Symbol; request=nothing, id=nothing, limit=20, cursor=nothing
)
    require_surface(l, :files)
    gemini=l.dialect=="gemini"
    anthropic=l.dialect=="anthropic"
    if operation===:upload
        request isa FileUploadRequest || throw(ArgumentError("upload needs FileUploadRequest"))
        validate(request)
        ext=something(request.extensions, obj())
        if gemini
            l.upload_base_url === nothing && throw(
                NotConfiguredError(
                    "a custom Gemini host needs upload_base_url for file uploads";
                    provider=l.provider,
                ),
            )
            content_type, body=multipart_related(
                obj("file"=>obj("display_name"=>request.filename)),
                request.media_type,
                bytes(request),
            )
            headers=base_headers(l; content_type)
            headers["x-goog-upload-protocol"]="multipart"
            return emit(l; url=l.upload_base_url*"/files", headers, body, params=ext)
        end
        fields=Pair{String,String}[]
        anthropic || push!(fields, "purpose"=>string(get(ext, "purpose", "user_data")))
        append!(fields, [k=>legacy_string(v) for (k, v) in ext if anthropic || k!="purpose"])
        content_type, body=multipart_form(
            fields, [("file", request.filename, request.media_type, bytes(request))]
        )
        return emit(l; url=l.base_url*"/files", headers=base_headers(l; content_type), body)
    elseif operation===:list
        limit>0 || throw(ArgumentError("limit must be positive"))
        params=obj((gemini ? "pageSize" : "limit")=>limit)
        cursor===nothing || (params[if gemini
            "pageToken"
        elseif anthropic
            "page"
        else
            "after"
        end]=cursor)
        return emit(
            l;
            method="GET",
            url=l.base_url*"/files",
            params,
            headers=base_headers(l; content_type=gemini ? nothing : "application/json"),
        )
    end
    id isa AbstractString && !isempty(id) || throw(ArgumentError("file id required"))
    path=gemini ? "/"*path_id(file_resource(id); resource_name=true) : "/files/"*path_id(id)
    method=operation===:delete ? "DELETE" : "GET"
    params=obj()
    if operation===:download
        path*=gemini ? ":download" : "/content"
        gemini && (params["alt"]="media")
    elseif !(operation in (:get, :delete))
        throw(ArgumentError("unknown file operation"))
    end
    return emit(
        l;
        method,
        url=l.base_url*path,
        params,
        headers=base_headers(l; content_type=gemini ? nothing : "application/json"),
    )
end
function file_info(l, d)
    if l.dialect=="gemini"
        id=first_nonempty(string_field(d, "uri"), string_field(d, "name"))
        id===nothing && throw(GenericProviderError("file has no uri or name"; provider=l.provider))
        state=wire_string(get(d, "state", nothing))
        ready=if endswith(state, "PROCESSING")
            "pending"
        elseif endswith(state, "FAILED")
            "failed"
        else
            "ready"
        end
        size=get(d, "sizeBytes", nothing)
        size=if size isa AbstractString
            tryparse(Int, size)
        elseif size isa Integer && !(size isa Bool)
            size
        else
            nothing
        end
        downloadable=if string_field(d, "downloadUri")!==nothing
            true
        elseif get(d, "source", nothing)=="UPLOADED"
            false
        else
            nothing
        end
        return FileInfo(;
            id,
            filename=string_field(d, "displayName"),
            media_type=string_field(d, "mimeType"),
            size_bytes=size,
            created_at=normalized_time(get(d, "createTime", nothing)),
            expires_at=normalized_time(get(d, "expirationTime", nothing)),
            readiness=ready,
            downloadable,
            provider_data=d,
        )
    end
    id=string_field(d, "id")
    id===nothing && throw(GenericProviderError("file has no id"; provider=l.provider))
    anthropic=l.dialect=="anthropic"
    state=get(d, "status", nothing)
    ready=if anthropic
        "ready"
    elseif state in ("uploaded", "pending")
        "pending"
    elseif state in ("error", "failed")
        "failed"
    else
        "ready"
    end
    download=anthropic && get(d, "downloadable", nothing) isa Bool ? d["downloadable"] : nothing
    return FileInfo(;
        id,
        filename=string_field(d, "filename"),
        media_type=anthropic ? string_field(d, "mime_type") : nothing,
        size_bytes=int_field(d, anthropic ? "size_bytes" : "bytes"),
        created_at=normalized_time(get(d, "created_at", nothing)),
        expires_at=normalized_time(get(d, "expires_at", nothing)),
        readiness=ready,
        downloadable=download,
        provider_data=d,
    )
end
function parse_file_response(l::ProviderLM, r::HttpResponse; page=false)
    check_response_status(l, r)
    d=reply_json(l, r)
    if !page
        l.dialect=="gemini" && get(d, "file", nothing) isa AbstractDict && (d=d["file"])
        return file_info(l, d)
    end
    gemini=l.dialect=="gemini"
    anthropic=l.dialect=="anthropic"
    items=Tuple(
        file_info(l, x) for x in getarray(d, gemini ? "files" : "data") if x isa AbstractDict
    )
    cursor=if gemini
        string_field(d, "nextPageToken")
    elseif anthropic
        string_field(d, "next_page")
    elseif get(d, "has_more", false)===true && !isempty(items)
        string_field(d, "last_id")
    else
        nothing
    end
    return FilePage(; items, next_cursor=cursor)
end
function check_response_status(l, r)
    return if r.status>=400
        throw(attach_error_metadata(normalize_error(l, r.status, String(copy(r.body))), r))
    else
        nothing
    end
end
function file_upload(l::ProviderLM, r::FileUploadRequest)
    return parse_file_response(l, send_request(l, build_file_request(l, :upload; request=r)))
end
function file_get(l::ProviderLM, id)
    return parse_file_response(l, send_request(l, build_file_request(l, :get; id)))
end
function file_list(l::ProviderLM; limit=20, cursor=nothing)
    return parse_file_response(
        l, send_request(l, build_file_request(l, :list; limit, cursor)); page=true
    )
end
file_delete(l::ProviderLM, id) = (send_request(l, build_file_request(l, :delete; id)); nothing)
file_download(l::ProviderLM, id) = send_request(l, build_file_request(l, :download; id)).body
function file_wait_ready(l, id; poll_every=2.0, timeout=nothing)
    return wait_until(()->file_get(l, id), f->f.readiness!="pending"; poll_every, timeout)
end
function check_cache_prefix(prefix, ttl)
    validate(prefix)
    isempty(to_dict(prefix.config)) ||
        throw(ArgumentError("cache prefix cannot carry generation settings"))
    return ttl===nothing ||
           (integer_value(ttl)>0) ||
           throw(ArgumentError("cache TTL must be positive"))
end
function build_cache_request(
    l::ProviderLM,
    operation::Symbol;
    prefix=nothing,
    id=nothing,
    ttl_seconds=nothing,
    label=nothing,
    limit=20,
    cursor=nothing,
)
    require_surface(l, :caches)
    l.dialect=="gemini" || unsupported(l.provider, "stored cache resources")
    if operation===:create
        prefix isa Request || throw(ArgumentError("cache_create needs a prefix Request"))
        check_cache_prefix(prefix, ttl_seconds)
        model=startswith(prefix.model, "models/") ? prefix.model : "models/"*prefix.model
        d=obj("model"=>model, "contents"=>gemini_messages(l, prefix.messages))
        prefix.system===nothing || (
            d["systemInstruction"]=obj(
                "parts"=>[obj("text"=>system_text(prefix.system; provider=l.provider))]
            )
        )
        isempty(prefix.tools) || (d["tools"]=gemini_tools(prefix.tools))
        ttl_seconds===nothing || (d["ttl"]="$(integer_value(ttl_seconds))s")
        label===nothing || (d["displayName"]=label)
        return emit(l; url=l.base_url*"/cachedContents", payload=d)
    elseif operation===:list
        limit>0 || throw(ArgumentError("limit must be positive"))
        params=obj("pageSize"=>limit)
        cursor===nothing || (params["pageToken"]=cursor)
        return emit(
            l;
            method="GET",
            url=l.base_url*"/cachedContents",
            params,
            headers=base_headers(l; content_type=nothing),
        )
    end
    id isa AbstractString && !isempty(id) || throw(ArgumentError("cache id required"))
    url=l.base_url*"/"*path_id(cache_resource(id); resource_name=true)
    if operation===:update
        ttl_seconds!==nothing && integer_value(ttl_seconds)>0 ||
            throw(ArgumentError("cache TTL must be positive"))
        return emit(l; method="PATCH", url, payload=obj("ttl"=>"$(integer_value(ttl_seconds))s"))
    end
    operation in (:get, :delete) || throw(ArgumentError("unknown cache operation"))
    return emit(
        l;
        method=operation===:get ? "GET" : "DELETE",
        url,
        headers=base_headers(l; content_type=nothing),
    )
end
function cache_info(l, d)
    id=string_field(d, "name")
    model=string_field(d, "model")
    if id===nothing || model===nothing
        throw(GenericProviderError("cache object has no name or model"; provider=l.provider))
    else
        nothing
    end
    tokens=get(getobject(d, "usageMetadata"), "totalTokenCount", nothing)
    tokens=if tokens isa AbstractString
        tryparse(Int, tokens)
    elseif tokens isa Integer && !(tokens isa Bool)
        tokens
    else
        nothing
    end
    return CacheInfo(;
        id,
        model=replace(model, r"^models/"=>""),
        tokens,
        created_at=normalized_time(get(d, "createTime", nothing)),
        expires_at=normalized_time(get(d, "expireTime", nothing)),
        label=string_field(d, "displayName"),
        provider_data=d,
    )
end
function parse_cache_response(l, r; page=false)
    check_response_status(l, r)
    d=reply_json(l, r)
    return if page
        CachePage(;
            items=Tuple(
                cache_info(l, x) for x in getarray(d, "cachedContents") if x isa AbstractDict
            ),
            next_cursor=string_field(d, "nextPageToken"),
        )
    else
        cache_info(l, d)
    end
end
function cache_create(l::ProviderLM, prefix::Request; ttl_seconds=nothing, label=nothing)
    return parse_cache_response(
        l, send_request(l, build_cache_request(l, :create; prefix, ttl_seconds, label))
    )
end
function cache_get(l::ProviderLM, id)
    return parse_cache_response(l, send_request(l, build_cache_request(l, :get; id)))
end
function cache_list(l::ProviderLM; limit=20, cursor=nothing)
    return parse_cache_response(
        l, send_request(l, build_cache_request(l, :list; limit, cursor)); page=true
    )
end
cache_delete(l::ProviderLM, id) = (send_request(l, build_cache_request(l, :delete; id)); nothing)
function cache_update(l::ProviderLM, id; ttl_seconds)
    return parse_cache_response(
        l, send_request(l, build_cache_request(l, :update; id, ttl_seconds))
    )
end
function cache(l::ProviderLM, prefix::Request; ttl_seconds=nothing, label=nothing)
    check_cache_prefix(prefix, ttl_seconds)
    return CachedPrefix(;
        prefix,
        resource=l.access.supports.caches ? cache_create(l, prefix; ttl_seconds, label) : nothing,
    )
end
function cache(router::LMRouter, prefix::Request; kw...)
    resolution=resolve(router, prefix.model)
    return cache(lm(router, prefix.model), reconstruct(prefix; model=resolution.model); kw...)
end
function cache_config(c::CachedPrefix)
    return CacheConfig(;
        prefix_until_index=length(c.prefix.messages)-1,
        resource=c.resource===nothing ? nothing : c.resource.id,
    )
end
function request(c::CachedPrefix, messages; config=nothing)
    suffix=if messages isa Request
        messages.model==c.prefix.model && messages.system===nothing && isempty(messages.tools) ||
            throw(ArgumentError("suffix Request cannot change model, system or tools"))
        config===nothing && (config=messages.config)
        messages.messages
    elseif messages isa AbstractString
        (user(messages),)
    elseif messages isa Message
        (messages,)
    else
        Tuple(messages)
    end
    isempty(suffix) && throw(ArgumentError("cache suffix must not be empty"))
    config===nothing && (config=Config())
    config.cache===nothing || throw(ArgumentError("CachedPrefix owns config.cache"))
    return Request(
        c.prefix.model,
        (c.prefix.messages..., suffix...);
        system=c.prefix.system,
        tools=c.prefix.tools,
        config=reconstruct(config; cache=cache_config(c)),
    )
end

function generation_request(l, request::ImageGenerationRequest)
    ext=copy(something(request.extensions, obj()))
    if request.size!==nothing
        gen=copy(getobject(ext, "generationConfig"))
        image=copy(getobject(gen, "imageConfig"))
        get!(image, "aspectRatio", request.size)
        gen["imageConfig"]=image
        ext["generationConfig"]=gen
    end
    return Request(
        request.model,
        (user((TextPart(request.prompt), request.images...)),);
        config=Config(; extensions=ext),
    )
end
function generation_request(l, request::SpeechGenerationRequest)
    request.format===nothing || unsupported(l.provider, "speech output format selection")
    gen=obj("responseModalities"=>["AUDIO"])
    request.voice===nothing || (
        gen["speechConfig"]=obj(
            "voiceConfig"=>obj("prebuiltVoiceConfig"=>obj("voiceName"=>request.voice))
        )
    )
    return Request(
        request.model,
        (user(request.prompt),);
        config=Config(;
            extensions=merge(obj("generationConfig"=>gen), something(request.extensions, obj()))
        ),
    )
end
function build_generation_request(l::ProviderLM, r::ImageGenerationRequest)
    require_surface(l, :images)
    validate(r)
    l.dialect=="gemini" && return build_request(l, generation_request(l, r))
    ext=something(r.extensions, obj())
    payload=merge(obj("model"=>r.model, "prompt"=>r.prompt), ext)
    if l.provider=="xai"
        r.size===nothing || unsupported(l.provider, "image size")
        length(r.images)<=1 || unsupported(l.provider, "multiple edit images")
        if !isempty(r.images)
            p=only(r.images)
            payload["image"]=if p.file_id!==nothing
                obj("file_id"=>p.file_id)
            else
                obj("url"=>p.url===nothing ? media_uri(p) : p.url)
            end
        end
        return emit(
            l; url=l.base_url*(isempty(r.images) ? "/images/generations" : "/images/edits"), payload
        )
    end
    if isempty(r.images)
        r.size===nothing || get!(payload, "size", r.size)
        return emit(l; url=l.base_url*"/images/generations", payload)
    end
    all(p->p.data!==nothing || p.path!==nothing, r.images) ||
        unsupported(l.provider, "URL/file-id image edits; upload bytes explicitly")
    fields=["model"=>r.model, "prompt"=>r.prompt]
    r.size===nothing || push!(fields, "size"=>r.size)
    append!(fields, [k=>legacy_string(v) for (k, v) in ext])
    compat=resolved_compat(l.compat, r.model)
    files=[
        (
            compat.edit_image_field=="indexed" ? "image[$(i-1)]" : "image[]",
            "image-$(i-1)",
            p.media_type,
            bytes(p),
        ) for (i, p) in enumerate(r.images)
    ]
    content_type, body=multipart_form(fields, files)
    return emit(l; url=l.base_url*"/images/edits", headers=base_headers(l; content_type), body)
end
function build_generation_request(l::ProviderLM, r::SpeechGenerationRequest)
    require_surface(l, :speech)
    validate(r)
    l.dialect=="gemini" && return build_request(l, generation_request(l, r))
    payload=merge(obj("model"=>r.model, "input"=>r.prompt), something(r.extensions, obj()))
    r.voice===nothing || (payload["voice"]=r.voice)
    r.format===nothing || (payload["response_format"]=r.format)
    return emit(l; url=l.base_url*"/audio/speech", payload)
end
function parse_generation_response(l, r::ImageGenerationRequest, http::HttpResponse)
    check_response_status(l, http)
    if l.dialect=="gemini"
        response=parse_response(l, generation_request(l, r), http)
        images=Tuple(parts_of(ImagePart, response.message))
        isempty(images) &&
            throw(GenericProviderError("model returned no image"; provider=l.provider))
        words=join((p.text for p in parts_of(TextPart, response.message)))
        return ImageGenerationResponse(;
            images,
            text=isempty(words) ? nothing : words,
            id=response.id,
            model=response.model,
            usage=response.usage,
            provider_data=response.provider_data,
        )
    end
    d=JSON.parse(String(copy(http.body)))
    images=ImagePart[]
    format=string_field(d, "output_format")
    for item in getarray(d, "data")
        item isa AbstractDict || continue
        mime=if l.provider=="xai"
            something(string_field(item, "mime_type"), "application/octet-stream")
        elseif format===nothing
            "application/octet-stream"
        else
            "image/$format"
        end
        data=string_field(item, "b64_json")
        url=string_field(item, "url")
        if data===nothing
            (url===nothing || push!(images, ImagePart(; media_type=mime, url)))
        else
            push!(images, ImagePart(; media_type=mime, data))
        end
    end
    isempty(images) &&
        throw(GenericProviderError("provider returned no images"; provider=l.provider))
    return ImageGenerationResponse(;
        images, usage=if l.provider=="xai"
            Usage()
        else
            usage_from("openai-responses", get(d, "usage", nothing))
        end, provider_data=d
    )
end
function parse_generation_response(l, r::SpeechGenerationRequest, http::HttpResponse)
    check_response_status(l, http)
    if l.dialect=="gemini"
        response=parse_response(l, generation_request(l, r), http)
        audio=parts_of(AudioPart, response.message)
        isempty(audio) &&
            throw(GenericProviderError("model returned no audio"; provider=l.provider))
        return SpeechGenerationResponse(;
            audio=first(audio),
            id=response.id,
            model=response.model,
            usage=response.usage,
            provider_data=response.provider_data,
        )
    end
    mime=header(http, "content-type")
    empty_optional(mime) &&
        throw(GenericProviderError("speech response has no media type"; provider=l.provider))
    return SpeechGenerationResponse(;
        audio=AudioPart(; media_type=mime, data=base64encode(http.body)),
        provider_data=obj("content_type"=>mime),
    )
end
function image_generate(l::ProviderLM, r::ImageGenerationRequest)
    return parse_generation_response(l, r, send_request(l, build_generation_request(l, r)))
end
function speech_generate(l::ProviderLM, r::SpeechGenerationRequest)
    return parse_generation_response(l, r, send_request(l, build_generation_request(l, r)))
end
