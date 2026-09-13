const BATCH_TERMINAL_STATUSES=("completed", "failed", "cancelled", "expired")
const VIDEO_TERMINAL_STATUSES=("completed", "failed", "cancelled")
done(info::BatchJobInfo) = info.status in BATCH_TERMINAL_STATUSES
done(info::VideoJobInfo) = info.status in VIDEO_TERMINAL_STATUSES
function batch_status_word(l, d)
    if l.dialect=="anthropic"
        state=get(d, "processing_status", nothing)
        state=="in_progress" && return "running"
        state=="canceling" && return "cancelling"
        if state=="ended"
            counts=getobject(d, "request_counts")
            n(k) = get(counts, k, 0)
            n("canceled")>0 && n("succeeded")==n("errored")==n("expired")==0 && return "cancelled"
            n("expired")>0 && n("succeeded")==n("errored")==n("canceled")==0 && return "expired"
            return "completed"
        end
        return "queued"
    elseif l.dialect=="gemini"
        state=get(getobject(d, "metadata"), "state", nothing)
        return get(
            Dict(
                "BATCH_STATE_PENDING"=>"queued",
                "BATCH_STATE_RUNNING"=>"running",
                "BATCH_STATE_CANCELLING"=>"cancelling",
                "BATCH_STATE_SUCCEEDED"=>"completed",
                "BATCH_STATE_FAILED"=>"failed",
                "BATCH_STATE_CANCELLED"=>"cancelled",
                "BATCH_STATE_EXPIRED"=>"expired",
            ),
            state,
            get(d, "done", false)===true ? "completed" : "queued",
        )
    end
    state=get(d, "status", nothing)
    state in BATCH_TERMINAL_STATUSES && return state
    state in ("cancelling", "canceling") && return "cancelling"
    return state in ("in_progress", "finalizing") ? "running" : "queued"
end
function batch_info(l, d)
    gemini=l.dialect=="gemini"
    id=string_field(d, gemini ? "name" : "id")
    id===nothing &&
        throw(GenericProviderError("batch response carries no ticket"; provider=l.provider))
    meta=getobject(d, "metadata")
    return BatchJobInfo(;
        id,
        status=batch_status_word(l, d),
        label=if l.dialect=="anthropic"
            nothing
        else
            string_field(meta, gemini ? "displayName" : "label")
        end,
        created_at=normalized_time(
            get(gemini ? meta : d, gemini ? "createTime" : "created_at", nothing)
        ),
        provider_data=d,
    )
end
function build_batch_requests(
    l::ProviderLM,
    action::Symbol;
    request=nothing,
    id=nothing,
    limit=20,
    upload_body=nothing,
    status_body=nothing,
)
    require_surface(l, :batches)
    gemini=l.dialect=="gemini"
    anthropic=l.dialect=="anthropic"
    root=anthropic ? "/messages/batches" : "/batches"
    headers=base_headers(l; content_type=gemini ? nothing : "application/json")
    if action in (:upload, :submit)
        request isa BatchRequest || throw(ArgumentError("batch operation needs BatchRequest"))
        validate(request)
        if action===:upload
            gemini ||
                anthropic ||
                begin
                    lines=[
                        JSON.serialize(
                            obj(
                                "custom_id"=>string(i-1),
                                "method"=>"POST",
                                "url"=>"/v1/responses",
                                "body"=>build_payload(l, r),
                            ),
                        ) for (i, r) in enumerate(request.requests)
                    ]
                    content_type, body=multipart_form(
                        ["purpose"=>"batch"],
                        [(
                            "file",
                            "lm15-batch.jsonl",
                            "application/jsonl",
                            Vector{UInt8}(codeunits(join(lines, "\n")*"\n")),
                        )],
                    )
                    return [
                        emit(
                            l; url=l.base_url*"/files", headers=base_headers(l; content_type), body
                        ),
                    ]
                end
            return WireRequest[]
        end
        ext=something(request.extensions, obj())
        if anthropic
            request.label===nothing || unsupported(l.provider, "batch label")
            payload=merge(
                obj(
                    "requests"=>[
                        obj("custom_id"=>string(i-1), "params"=>build_payload(l, r)) for
                        (i, r) in enumerate(request.requests)
                    ],
                ),
                ext,
            )
            return [emit(l; url=l.base_url*root, payload)]
        elseif gemini
            batch=obj(
                "inputConfig"=>obj(
                    "requests"=>obj(
                        "requests"=>[
                            obj(
                                "request"=>build_payload(l, r), "metadata"=>obj("key"=>string(i-1))
                            ) for (i, r) in enumerate(request.requests)
                        ],
                    ),
                ),
            )
            request.label===nothing || (batch["displayName"]=request.label)
            model=startswith(request.model, "models/") ? request.model : "models/"*request.model
            return [
                emit(
                    l;
                    url=l.base_url*"/"*percent_encode(model; safe="/:@")*":batchGenerateContent",
                    payload=merge(obj("batch"=>batch), ext),
                ),
            ]
        end
        file_id=string_field(asobject(upload_body), "id")
        file_id===nothing &&
            throw(GenericProviderError("batch upload returned no file id"; provider=l.provider))
        payload=obj(
            "input_file_id"=>file_id,
            "endpoint"=>get(ext, "endpoint", "/v1/responses"),
            "completion_window"=>get(ext, "completion_window", "24h"),
        )
        request.label===nothing || (payload["metadata"]=obj("label"=>request.label))
        merge!(payload, ext)
        return [emit(l; url=l.base_url*root, payload)]
    elseif action===:list
        limit>0 || throw(ArgumentError("limit must be positive"))
        return [
            emit(
                l;
                method="GET",
                url=l.base_url*root,
                headers,
                params=obj((gemini ? "pageSize" : "limit")=>limit),
            ),
        ]
    elseif action===:result_fetches
        d=asobject(status_body)
        gemini && return WireRequest[]
        if anthropic
            url=string_field(d, "results_url")
            url===nothing &&
                throw(GenericProviderError("ended batch has no results URL"; provider=l.provider))
            return [emit(l; method="GET", url, headers)]
        end
        return [
            emit(l; method="GET", url=l.base_url*"/files/"*path_id(d[key])*"/content", headers) for
            key in ("output_file_id", "error_file_id") if string_field(d, key)!==nothing
        ]
    end
    action in (:status, :cancel) || throw(ArgumentError("unknown batch operation"))
    id isa AbstractString && !isempty(id) || throw(ArgumentError("batch id required"))
    path=gemini ? "/"*path_id(id; resource_name=true) : root*"/"*path_id(id)
    action===:cancel && (path*=gemini ? ":cancel" : "/cancel")
    return [
        emit(
            l;
            method=action===:cancel ? "POST" : "GET",
            url=l.base_url*path,
            headers,
            payload=gemini && action===:cancel ? obj() : nothing,
        ),
    ]
end
function parse_batch_response(l, r::HttpResponse; list=false)
    check_response_status(l, r)
    d=JSON.parse(String(copy(r.body)))
    return if list
        [
            batch_info(l, x) for
            x in getarray(d, l.dialect=="gemini" ? "operations" : "data") if x isa AbstractDict
        ]
    else
        batch_info(l, d)
    end
end
function batch_entry_response(l, d)
    model=first_nonempty(get(d, "model", nothing), get(d, "modelVersion", nothing), "batch")
    return parse_response(
        l,
        Request(string(model), (user("-"),)),
        HttpResponse(; body=Vector{UInt8}(codeunits(JSON.serialize(d)))),
    )
end
function batch_error_entry(l, index, status, raw)
    e=normalize_error(l, status, JSON.serialize(raw))
    return BatchEntry(;
        index,
        outcome="errored",
        error=ErrorDetail(; code=e.code, message=e.message, provider_code=e.provider_code),
    )
end
function parse_batch_entries(l, status_body, fetched)
    entries=BatchEntry[]
    if l.dialect=="gemini"
        raw=get(getobject(status_body, "response"), "inlinedResponses", nothing)
        raw isa AbstractDict && (raw=get(raw, "inlinedResponses", nothing))
        for (position, item) in enumerate(asarray(raw))
            item isa AbstractDict || continue
            key=get(getobject(item, "metadata"), "key", nothing)
            index=tryparse(Int, string(key))
            index===nothing && (index=position-1)
            if get(item, "response", nothing) isa AbstractDict
                push!(
                    entries,
                    BatchEntry(;
                        index,
                        outcome="succeeded",
                        response=batch_entry_response(l, item["response"]),
                    ),
                )
            else
                err=getobject(item, "error")
                code=first_nonempty(get(err, "status", nothing), get(err, "code", nothing))
                push!(
                    entries,
                    BatchEntry(;
                        index,
                        outcome="errored",
                        error=ErrorDetail(;
                            code="provider",
                            message=wire_string(get(err, "message", "batch entry errored")),
                            provider_code=code===nothing ? nothing : string(code),
                        ),
                    ),
                )
            end
        end
        return sort!(entries; by=e->e.index)
    end
    found=Dict{Int,BatchEntry}()
    for text in fetched, line in split(text, '\n'; keepempty=false)
        isempty(strip(line)) && continue
        d=JSON.parse(line)
        index=parse(Int, string(d["custom_id"]))
        if haskey(found, index)
            throw(GenericProviderError("duplicate batch result index"; provider=l.provider))
        end
        if l.dialect=="anthropic"
            result=getobject(d, "result")
            type=get(result, "type", nothing)
            entry=if type=="succeeded"
                BatchEntry(;
                    index,
                    outcome="succeeded",
                    response=batch_entry_response(l, getobject(result, "message")),
                )
            elseif type=="errored"
                raw=getobject(result, "error")
                batch_error_entry(l, index, 400, haskey(raw, "error") ? raw : obj("error"=>raw))
            elseif type in ("canceled", "expired")
                BatchEntry(; index, outcome=type=="canceled" ? "cancelled" : "expired")
            else
                BatchEntry(;
                    index,
                    outcome="errored",
                    error=ErrorDetail(; code="provider", message="unrecognized batch result type"),
                )
            end
        else
            response=getobject(d, "response")
            status=Int(get(response, "status_code", 0))
            body=getobject(response, "body")
            entry=if status==200 && !isempty(body)
                BatchEntry(; index, outcome="succeeded", response=batch_entry_response(l, body))
            else
                batch_error_entry(
                    l,
                    index,
                    status==0 ? 400 : status,
                    isempty(body) ? get(d, "error", obj()) : body,
                )
            end
        end
        found[index]=entry
    end
    l.dialect=="anthropic" && return [found[i] for i in sort!(collect(keys(found)))]
    total=Int(get(getobject(status_body, "request_counts"), "total", 0))
    total==0 && !isempty(found) && (total=maximum(keys(found))+1)
    total<0 && throw(GenericProviderError("batch total is negative"; provider=l.provider))
    any(i->i<0 || i>=total, keys(found)) &&
        throw(GenericProviderError("batch index exceeds reported total"; provider=l.provider))
    status=batch_status_word(l, status_body)
    fill=if status=="expired"
        "expired"
    elseif status=="cancelled"
        "cancelled"
    else
        "errored"
    end
    for i in 0:(total - 1)
        push!(
            entries,
            get(
                found,
                i,
                if fill=="errored"
                    BatchEntry(;
                        index=i,
                        outcome=fill,
                        error=ErrorDetail(;
                            code="provider", message="entry missing from batch output files"
                        ),
                    )
                else
                    BatchEntry(; index=i, outcome=fill)
                end,
            ),
        )
    end
    return entries
end
function batch_submit(l::ProviderLM, r::BatchRequest)
    uploads=build_batch_requests(l, :upload; request=r)
    upload=nothing
    for wire in uploads
        upload=JSON.parse(String(send_request(l, wire).body))
    end
    return parse_batch_response(
        l, send_request(l, only(build_batch_requests(l, :submit; request=r, upload_body=upload)))
    )
end
function batch_status(l::ProviderLM, id)
    return parse_batch_response(l, send_request(l, only(build_batch_requests(l, :status; id))))
end
function batch_cancel(l::ProviderLM, id)
    return parse_batch_response(l, send_request(l, only(build_batch_requests(l, :cancel; id))))
end
function batch_list(l::ProviderLM; limit=20)
    return parse_batch_response(
        l, send_request(l, only(build_batch_requests(l, :list; limit))); list=true
    )
end
function batch_results(l::ProviderLM, id)
    info=batch_status(l, id)
    done(info) || throw(ArgumentError("batch is not finished; explicitly wait or poll"))
    d=info.provider_data
    fetched=String[]
    for wire in build_batch_requests(l, :result_fetches; status_body=d)
        push!(fetched, String(send_request(l, wire).body))
    end
    return parse_batch_entries(l, d, fetched)
end

function build_video_requests(
    l::ProviderLM,
    action::Symbol;
    request=nothing,
    id=nothing,
    status_body=nothing,
    limit=20,
    model=nothing,
)
    require_surface(l, :video)
    gemini=l.dialect=="gemini"
    xai=l.provider=="xai"
    headers=base_headers(l; content_type=gemini ? nothing : "application/json")
    if action===:submit
        request isa VideoGenerationRequest ||
            throw(ArgumentError("video submit needs VideoGenerationRequest"))
        validate(request)
        isempty(request.images) ||
            unsupported(l.provider, "video input images until a provider mapping is recorded")
        ext=something(request.extensions, obj())
        if gemini
            payload=merge(obj("instances"=>[obj("prompt"=>request.prompt)]), ext)
            request.seconds===nothing ||
                get!(payload, "parameters", obj("durationSeconds"=>request.seconds))
            model=startswith(request.model, "models/") ? request.model : "models/"*request.model
            url=l.base_url*"/"*percent_encode(model; safe="/:@")*":predictLongRunning"
        else
            xai && request.seconds!==nothing && unsupported(l.provider, "video duration")
            payload=merge(obj("model"=>request.model, "prompt"=>request.prompt), ext)
            request.seconds===nothing || (payload["seconds"]=string(request.seconds))
            url=l.base_url*(xai ? "/videos/generations" : "/videos")
        end
        return [emit(l; url, payload)]
    elseif action===:list
        xai && unsupported(
            l.provider, "video listing; this provider retains only the ticket you received"
        )
        if gemini
            model isa AbstractString && !isempty(model) ||
                unsupported(l.provider, "video listing without a model")
            model=startswith(model, "models/") ? model : "models/"*model
            return [
                emit(
                    l;
                    method="GET",
                    url=l.base_url*"/"*percent_encode(model; safe="/:@")*"/operations",
                    headers,
                    params=obj("pageSize"=>limit),
                ),
            ]
        end
        return [
            emit(l; method="GET", url=l.base_url*"/videos", headers, params=obj("limit"=>limit))
        ]
    elseif action===:result_fetch
        xai && return WireRequest[]
        d=asobject(status_body)
        if gemini
            samples=getarray(
                getobject(getobject(d, "response"), "generateVideoResponse"), "generatedSamples"
            )
            isempty(samples) && throw(
                GenericProviderError(
                    "video operation carries no generated sample"; provider=l.provider
                ),
            )
            url=string_field(getobject(asobject(first(samples)), "video"), "uri")
            url===nothing && throw(
                GenericProviderError("video operation carries no result URI"; provider=l.provider),
            )
        else
            id=string_field(d, "id")
            id===nothing &&
                throw(GenericProviderError("video response has no id"; provider=l.provider))
            url=l.base_url*"/videos/"*path_id(id)*"/content"
        end
        return [emit(l; method="GET", url, headers)]
    elseif action===:status
        id isa AbstractString && !isempty(id) || throw(ArgumentError("video id required"))
        url=l.base_url*(gemini ? "/"*path_id(id; resource_name=true) : "/videos/"*path_id(id))
        return [emit(l; method="GET", url, headers)]
    end
    return throw(ArgumentError("unknown video operation"))
end
function video_info(l, d; id=nothing)
    if l.dialect=="gemini"
        id=string_field(d, "name")
        id===nothing &&
            throw(GenericProviderError("video operation has no name"; provider=l.provider))
        status=if get(d, "done", false)===true
            (get(d, "error", nothing) isa AbstractDict ? "failed" : "completed")
        else
            "running"
        end
        return VideoJobInfo(; id, status, provider_data=d)
    end
    if l.provider=="xai"
        ticket=string_field(d, "request_id")
        ticket===nothing || return VideoJobInfo(; id=ticket, status="queued", provider_data=d)
        id===nothing &&
            throw(GenericProviderError("video response has no ticket"; provider=l.provider))
        mapping=Dict("pending"=>"running", "done"=>"completed", "failed"=>"failed")
    else
        id=string_field(d, "id")
        id===nothing && throw(GenericProviderError("video response has no id"; provider=l.provider))
        mapping=Dict(
            "queued"=>"queued",
            "in_progress"=>"running",
            "completed"=>"completed",
            "failed"=>"failed",
            "cancelled"=>"cancelled",
        )
    end
    status=get(mapping, get(d, "status", nothing), nothing)
    status===nothing && throw(
        GenericProviderError(
            "unknown video status; refusing to guess a polling state"; provider=l.provider
        ),
    )
    return VideoJobInfo(;
        id,
        status,
        progress=int_field(d, "progress"),
        model=string_field(d, "model"),
        created_at=normalized_time(get(d, "created_at", nothing)),
        provider_data=d,
    )
end
function parse_video_response(l, r::HttpResponse; list=false, id=nothing)
    check_response_status(l, r)
    d=JSON.parse(String(copy(r.body)))
    return if list
        [
            video_info(l, x) for
            x in getarray(d, l.dialect=="gemini" ? "operations" : "data") if x isa AbstractDict
        ]
    else
        video_info(l, d; id)
    end
end
function parse_video_part(l, d, fetched)
    if l.provider=="xai"
        url=string_field(getobject(d, "video"), "url")
        url===nothing &&
            throw(GenericProviderError("terminal video carries no URL"; provider=l.provider))
        return VideoPart(; media_type="video/mp4", url)
    end
    fetched isa HttpResponse || throw(ArgumentError("video result needs a content fetch"))
    check_response_status(l, fetched)
    mime=header(fetched, "content-type")
    empty_optional(mime) &&
        throw(GenericProviderError("video content carries no media type"; provider=l.provider))
    return VideoPart(;
        media_type=String(strip(first(split(mime, ';')))), data=base64encode(fetched.body)
    )
end
function video_submit(l::ProviderLM, r::VideoGenerationRequest)
    return parse_video_response(
        l, send_request(l, only(build_video_requests(l, :submit; request=r)))
    )
end
function video_status(l::ProviderLM, id)
    return parse_video_response(l, send_request(l, only(build_video_requests(l, :status; id))); id)
end
function video_list(l::ProviderLM; limit=20, model=nothing)
    return parse_video_response(
        l, send_request(l, only(build_video_requests(l, :list; limit, model))); list=true
    )
end
function video_result(l::ProviderLM, id)
    info=video_status(l, id)
    done(info) || throw(ArgumentError("video is not finished; explicitly wait or poll"))
    info.status=="completed" || throw(
        GenericProviderError("video job ended without a successful result"; provider=l.provider)
    )
    d=info.provider_data
    fetches=build_video_requests(l, :result_fetch; status_body=d)
    return parse_video_part(l, d, isempty(fetches) ? nothing : send_request(l, only(fetches)))
end

struct WaitTimeout <: Exception
    snapshot::Any
end
function Base.showerror(io::IO, ::WaitTimeout)
    return print(io, "Job wait reached its deadline; the provider job may still be running")
end
function validate_wait(poll_every, timeout)
    poll_every isa Real && !(poll_every isa Bool) && isfinite(poll_every) && poll_every>0 ||
        throw(ArgumentError("poll_every must be finite and positive"))
    return timeout===nothing ||
           (timeout isa Real && !(timeout isa Bool) && isfinite(timeout) && timeout>=0) ||
           throw(ArgumentError("timeout must be finite and non-negative"))
end
function wait_until(fetch, terminal; poll_every, timeout=nothing)
    validate_wait(poll_every, timeout)
    deadline=timeout===nothing ? Inf : time_ns()/1e9+timeout
    snapshot=fetch()
    while !terminal(snapshot)
        remaining=deadline-time_ns()/1e9
        remaining>0 || throw(WaitTimeout(snapshot))
        sleep(min(poll_every, remaining))
        time_ns()/1e9<deadline || throw(WaitTimeout(snapshot))
        snapshot=fetch()
    end
    return snapshot
end
mutable struct BatchJob{L}
    lm::L
    info::BatchJobInfo
end
mutable struct VideoJob{L}
    lm::L
    info::VideoJobInfo
end
done(job::Union{BatchJob,VideoJob}) = done(job.info)
function Base.show(io::IO, j::Union{BatchJob,VideoJob})
    return print(
        io, nameof(typeof(j)), "(id=", repr(j.info.id), ", status=", repr(j.info.status), ")"
    )
end
refresh!(j::BatchJob) = (j.info=batch_status(j.lm, j.info.id); j)
refresh!(j::VideoJob) = (j.info=video_status(j.lm, j.info.id); j)
cancel!(j::BatchJob) = (j.info=batch_cancel(j.lm, j.info.id); j)
results(j::BatchJob) = batch_results(j.lm, j.info.id)
result(j::VideoJob) = video_result(j.lm, j.info.id)
function wait!(
    job::Union{BatchJob,VideoJob}; poll_every=job isa BatchJob ? 30.0 : 5.0, timeout=nothing
)
    validate_wait(poll_every, timeout)
    done(job) && return job
    deadline=timeout===nothing ? Inf : time_ns()/1e9+timeout
    while !done(job)
        remaining=deadline-time_ns()/1e9
        remaining>0 || throw(WaitTimeout(job.info))
        sleep(min(poll_every, remaining))
        time_ns()/1e9<deadline || throw(WaitTimeout(job.info))
        refresh!(job)
    end
    return job
end
function batch(l::ProviderLM, requests; kw...)
    return BatchJob(
        l,
        batch_submit(
            l,
            requests isa BatchRequest ? requests : BatchRequest(; requests=Tuple(requests), kw...),
        ),
    )
end
batch_job(l::ProviderLM, id) = BatchJob(l, batch_status(l, id))
batches(l::ProviderLM; kw...) = [BatchJob(l, info) for info in batch_list(l; kw...)]
video_generate(l::ProviderLM, r::VideoGenerationRequest) = VideoJob(l, video_submit(l, r))
video_job(l::ProviderLM, id) = VideoJob(l, video_status(l, id))
video_jobs(l::ProviderLM; kw...) = [VideoJob(l, info) for info in video_list(l; kw...)]
