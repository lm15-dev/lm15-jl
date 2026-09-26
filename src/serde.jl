const REQUIRED_SHAPE = Dict{DataType,Tuple}(
    TextPart=>(:type, :text),
    ThinkingPart=>(:type, :text),
    RefusalPart=>(:type, :text),
    ToolCallPart=>(:type, :id, :name, :input),
    ToolResultPart=>(:type, :id, :content),
    FunctionTool=>(:type, :name, :parameters),
    ContinuationState=>(:provider, :kind, :data),
    Message=>(:role, :parts),
    Request=>(:model, :messages),
    Response=>(:model, :message, :finish_reason),
    TokenLogprob=>(:token, :logprob),
    TopLogprob=>(:token, :logprob),
)
function to_dict(value::Canonical; include_provider_data=false)
    validate(value)
    T = typeof(value)
    required = get(REQUIRED_SHAPE, T, ())
    d = obj()
    for name in fieldnames(T)
        name === :provider_data && value isa Response && !include_provider_data && continue
        v = getfield(value, name)
        if value isa DataPart && name === :value
            d["value"] = v  # an opaque payload: always emitted, null included (serde-rules.md)
            continue
        end
        if name === :logprobs_complete
            v || (d["logprobs_complete"] = false)  # omitted when true; false is data
            continue
        end
        v === nothing && continue
        encoded = if name === :expires_at && value isa CredentialValue
            normalized_time(v)
        elseif name === :bytes_data && value isa FileUploadRequest
            base64encode(v)
        elseif v isa Canonical
            to_dict(v; include_provider_data=value isa BatchEntry || include_provider_data)
        elseif v isa Tuple
            [
                if item isa Canonical
                    to_dict(item; include_provider_data=include_provider_data)
                else
                    item
                end for item in v
            ]
        else
            v
        end
        if value isa ModelInfo && name === :origin && encoded == obj("type"=>"provider")
            continue
        end
        if value isa ToolResultPart && name === :is_error && !v
            continue
        end
        value isa InferenceModelInfo && name === :supports_reasoning && !v && continue
        always =
            name in required ||
            name === :type ||
            (value isa MediaPart && name === :media_type) ||
            (value isa Delta && (name !== :logprobs || !isempty(v))) ||
            value isa LiveClientEvent ||
            (value isa LiveServerTextEvent && name === :text) ||
            (value isa LiveServerToolCallEvent && name in (:input, :id, :name)) ||
            (value isa Union{TokenLogprob,TopLogprob} && name === :bytes)
        always || !empty_optional(encoded) || continue
        d[string(name)] = encoded
    end
    return d
end

to_json(value::Canonical; kw...) = JSON.serialize(to_dict(value; kw...))
from_json(::Type{T}, value::AbstractString) where {T} = from_dict(T, JSON.parse(value))

function variant(T, d)
    table = if T === Part
        PART_VARIANTS
    elseif T === Delta
        DELTA_VARIANTS
    elseif T === StreamEvent
        STREAM_VARIANTS
    elseif T === LiveClientEvent
        LIVE_CLIENT_VARIANTS
    elseif T === LiveServerEvent
        LIVE_SERVER_VARIANTS
    else
        nothing
    end
    T === Tool && return get(d, "type", nothing) == "builtin" ? BuiltinTool : FunctionTool
    table === nothing && return T
    tag = get(d, "type", nothing)
    haskey(table, tag) || throw(ArgumentError("unknown $T discriminator"))
    return table[tag]
end
legacy_string(x::Nothing) = "None"
legacy_string(x::Bool) = x ? "True" : "False"
legacy_string(x::AbstractString) = String(x)
legacy_string(x) = string(x)
function read_parts(raw; lenient=false)
    raw isa AbstractVector || throw(ArgumentError("parts must be an array"))
    return Tuple(if p isa AbstractDict
        from_dict(Part, p)
    elseif lenient
        TextPart(legacy_string(p))
    else
        throw(ArgumentError("part must be an object"))
    end for p in raw)
end
function from_dict(::Type{T}, data::AbstractDict) where {T<:Canonical}
    C = variant(T, data)
    kw = Dict{Symbol,Any}()
    for name in fieldnames(C)
        key = string(name)
        haskey(data, key) || continue
        v = data[key]
        if name === :type && C === FunctionTool
            continue # INV-034: unknown tool tags read as FunctionTool.
        elseif name === :continuation
            v === nothing && (kw[name]=(); continue)
            v isa AbstractVector || throw(ArgumentError("continuation must be an array"))
            kw[name] = Tuple(from_dict(ContinuationState, s) for s in v)
        elseif name === :parts
            kw[name] = read_parts(v; lenient=C === Message)
        elseif name === :content && C <: Union{ToolResultPart,LiveClientToolResultEvent}
            kw[name] = if v isa AbstractString && C === ToolResultPart
                isempty(v) ? () : (TextPart(v),)
            elseif v isa AbstractVector
                read_parts(v; lenient=C === ToolResultPart)
            else
                ()
            end
        elseif name === :system
            kw[name] = v isa AbstractVector ? read_parts(v) : v
        elseif name === :adaptations
            v isa AbstractVector || throw(ArgumentError("adaptations must be an array"))
            kw[name] = Tuple(from_dict(Adaptation, item) for item in v)
        elseif name in (:messages, :requests, :tools, :images, :top, :items) ||
            (name === :logprobs && C <: Union{Response,TextDelta})
            if v === nothing && name === :logprobs
                C === TextDelta || (kw[name] = nothing)
                continue
            end
            v isa AbstractVector || throw(ArgumentError("$key must be an array"))
            itemtype = if name === :messages
                Message
            elseif name === :requests
                Request
            elseif name === :tools
                Tool
            elseif name === :images
                ImagePart
            elseif name === :top
                TopLogprob
            elseif name === :logprobs
                TokenLogprob
            elseif C === FilePage
                FileInfo
            else
                CacheInfo
            end
            kw[name] = Tuple(from_dict(itemtype, item) for item in v)
        elseif name === :bytes_data && C === FileUploadRequest
            kw[name] = v isa AbstractString ? base64decode(v) : nothing
        elseif name === :prefix && C === CachedPrefix
            kw[name] = from_dict(Request, v)

        elseif any(T -> T <: Canonical, Base.uniontypes(fieldtype(C, name)))
            # Field names are not types: ErrorDetail.message is text, and
            # CacheConfig.resource is an id, unlike Response.message and
            # CachedPrefix.resource. Dispatch using the declared field type.
            NT = only(T for T in Base.uniontypes(fieldtype(C, name)) if T <: Canonical)
            if v isa AbstractDict
                kw[name] = from_dict(NT, v)
            elseif v === nothing
                if Nothing <: fieldtype(C, name)
                    kw[name] = nothing
                elseif name in (:usage, :origin, :input_format, :output_format)
                    continue
                else
                    throw(ArgumentError("$C.$key must be an object, not null"))
                end
            elseif name in
                (:usage, :origin, :inference, :input_format, :output_format, :pricing, :resource)
                continue # INV-042: telemetry nests are lenient.
            else
                throw(ArgumentError("$key must be an object"))
            end
        else
            kw[name] = v
        end
    end
    if C <: MediaPart && !haskey(data, "media_type")
        kw[:media_type] = "" # Canonical media JSON requires the MIME type.
    elseif C === Reasoning
        effort = get(data, "effort", get(data, "enabled", nothing) === false ? "off" : "medium")
        effort == "adaptive" && (effort="medium")
        kw[:effort] = effort
        if effort == "off"
            delete!(kw, :thinking_budget)
            delete!(kw, :summary)
        elseif !haskey(kw, :thinking_budget) && haskey(data, "budget")
            kw[:thinking_budget] = data["budget"]
        end
    elseif C === Usage && get(kw, :total_tokens, nothing) === nothing
        delete!(kw, :total_tokens) # Let the constructor apply INV-029.
    elseif C === BatchRequest && get(kw, :model, nothing) === nothing
        delete!(kw, :model)
    end
    return C(; kw...)
end
from_dict(::Type{T}, value) where {T<:Canonical} = throw(ArgumentError("$T must be a JSON object"))

const SERDE_KINDS = Dict(
    "part"=>Part,
    "message"=>Message,
    "tool"=>Tool,
    "tool_choice"=>ToolChoice,
    "reasoning"=>Reasoning,
    "config"=>Config,
    "cache_config"=>CacheConfig,
    "cache_info"=>CacheInfo,
    "cache_page"=>CachePage,
    "cached_prefix"=>CachedPrefix,
    "token_logprob"=>TokenLogprob,
    "continuation_state"=>ContinuationState,
    "error_detail"=>ErrorDetail,
    "delta"=>Delta,
    "usage"=>Usage,
    "stream_event"=>StreamEvent,
    "request"=>Request,
    "response"=>Response,
    "model_info"=>ModelInfo,
    "batch_request"=>BatchRequest,
    "batch_job"=>BatchJobInfo,
    "batch_entry"=>BatchEntry,
    "file_upload_request"=>FileUploadRequest,
    "file_info"=>FileInfo,
    "file_page"=>FilePage,
    "image_generation_request"=>ImageGenerationRequest,
    "image_generation_response"=>ImageGenerationResponse,
    "speech_generation_request"=>SpeechGenerationRequest,
    "speech_generation_response"=>SpeechGenerationResponse,
    "video_generation_request"=>VideoGenerationRequest,
    "video_job"=>VideoJobInfo,
    "audio_format"=>AudioFormat,
    "live_config"=>LiveConfig,
    "live_client_event"=>LiveClientEvent,
    "live_server_event"=>LiveServerEvent,
)
function from_dict(kind::AbstractString, d)
    haskey(SERDE_KINDS, kind) || throw(ArgumentError("unknown serde kind"))
    return from_dict(SERDE_KINDS[kind], d)
end
