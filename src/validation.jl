const VOCABULARIES = Dict(
    :role => ("user", "assistant", "tool", "developer"),
    :finish_reason => ("stop", "length", "tool_call", "content_filter", "error"),
    :effort => ("off", "minimal", "low", "medium", "high", "xhigh", "max"),
    :summary => ("auto", "concise", "detailed"),
    :retention => ("short", "long"),
    :prefix => ("stable", "history"),
    :readiness => ("pending", "ready", "failed"),
    :outcome => ("succeeded", "errored", "cancelled", "expired"),
    :encoding => ("pcm16", "opus", "mp3", "aac"),
    :detail => ("low", "high", "auto"),
)
const PART_VARIANTS = Dict(
    "text"=>TextPart,
    "thinking"=>ThinkingPart,
    "refusal"=>RefusalPart,
    "citation"=>CitationPart,
    "image"=>ImagePart,
    "audio"=>AudioPart,
    "video"=>VideoPart,
    "document"=>DocumentPart,
    "binary"=>BinaryPart,
    "tool_call"=>ToolCallPart,
    "tool_result"=>ToolResultPart,
)
const DELTA_VARIANTS = Dict(
    "text"=>TextDelta,
    "thinking"=>ThinkingDelta,
    "audio"=>AudioDelta,
    "image"=>ImageDelta,
    "citation"=>CitationDelta,
    "tool_call"=>ToolCallDelta,
    "continuation"=>ContinuationDelta,
)
const STREAM_VARIANTS = Dict(
    "start"=>StreamStartEvent,
    "delta"=>StreamDeltaEvent,
    "end"=>StreamEndEvent,
    "error"=>StreamErrorEvent,
)
const LIVE_CLIENT_VARIANTS = Dict(
    "turn"=>LiveClientTurnEvent,
    "audio"=>LiveClientAudioEvent,
    "image"=>LiveClientImageEvent,
    "text"=>LiveClientTextEvent,
    "tool_result"=>LiveClientToolResultEvent,
    "interrupt"=>LiveClientInterruptEvent,
    "end_audio"=>LiveClientEndAudioEvent,
)
const LIVE_SERVER_VARIANTS = Dict(
    "audio"=>LiveServerAudioEvent,
    "text"=>LiveServerTextEvent,
    "tool_call"=>LiveServerToolCallEvent,
    "tool_call_delta"=>LiveServerToolCallDeltaEvent,
    "interrupted"=>LiveServerInterruptedEvent,
    "turn_end"=>LiveServerTurnEndEvent,
    "usage"=>LiveServerUsageEvent,
    "error"=>LiveServerErrorEvent,
)
const TAGS = Dict{DataType,String}(
    T=>tag for table in (
        PART_VARIANTS,
        DELTA_VARIANTS,
        STREAM_VARIANTS,
        LIVE_CLIENT_VARIANTS,
        LIVE_SERVER_VARIANTS,
        Dict("function"=>FunctionTool, "builtin"=>BuiltinTool),
    ) for (tag, T) in table
)

function integer_value(value)
    value isa Real && !(value isa Bool) && isfinite(value) && isinteger(value) ||
        throw(ArgumentError("expected an integer, not a boolean or a fractional number"))
    typemin(Int) <= value <= typemax(Int) ||
        throw(ArgumentError("integer outside machine Int range"))
    return Int(value)
end
checked_token_sum(a, b) = Base.Checked.checked_add(integer_value(a), integer_value(b))
function normalize_field(T, name, value, declared)
    name === :tools && T in (Request, LiveConfig) && return normalize_tools(value)
    name === :extensions && value isa AbstractDict && isempty(value) && return nothing
    if name === :system && value !== nothing
        value isa AbstractString && return String(value)
        return normalize_content(value)
    end
    value === nothing && Nothing <: declared && return nothing
    if declared === Int || declared === Maybe{Int}
        return integer_value(value)
    elseif declared === Float64 || declared === Maybe{Float64}
        value isa Real && !(value isa Bool) && isfinite(value) ||
            throw(ArgumentError("$name must be a finite number"))
        result = Float64(value)
        isfinite(result) || throw(ArgumentError("$name is outside Float64 range"))
        return result
    elseif declared === Tuple || declared === Maybe{Tuple}
        value === nothing && name === :continuation && return ()
        value isa AbstractString &&
            !(name in (:allowed, :stop, :aliases)) &&
            throw(ArgumentError("$name must be a sequence of typed values"))
        return value isa Union{Canonical,AbstractString} ? (value,) : Tuple(value)
    elseif declared === String || declared === Maybe{String}
        value isa AbstractString || throw(ArgumentError("$name must be a string"))
        return String(value)
    elseif declared === Bool || declared === Maybe{Bool}
        value isa Bool || throw(ArgumentError("$name must be a boolean"))
    end
    value isa declared || throw(ArgumentError("$name has the wrong type"))
    return value
end

function base64_payload(value)
    startswith(value, "data:") &&
        occursin(";base64,", value) &&
        (value = split(value, ";base64,"; limit=2)[2])
    return join(split(value))
end
function check_base64(value)
    isempty(value) && throw(ArgumentError("inline data must not be empty"))
    payload = base64_payload(value)
    return length(payload) % 4 == 0 && occursin(r"^[A-Za-z0-9+/]*={0,2}$", payload) ||
           throw(ArgumentError("invalid base64 data"))
end
function require_items(items, T, name; nonempty=false)
    nonempty && isempty(items) && throw(ArgumentError("$name must not be empty"))
    all(v -> v isa T, items) || throw(ArgumentError("$name must contain $T values"))
    return foreach(validate, items)
end
prompt_part(p) = p isa Union{TextPart,MediaPart}
result_part(p) = p isa Union{TextPart,MediaPart,CitationPart}
function validate(x::Canonical)
    T = typeof(x)
    get(CANONICAL_TYPES, string(nameof(T)), nothing) === T ||
        throw(ArgumentError("unknown canonical variant"))
    for name in fieldnames(T)
        v = getfield(x, name)
        v === nothing && continue
        v isa AbstractDict && check_json(v)
        v isa Canonical && validate(v)
        if v isa AbstractString
            allowempty =
                name in (:text, :token, :message, :description, :input_delta) ||
                (name === :input && x isa ToolCallDelta) ||
                (
                    x isa Union{AudioDelta,ImageDelta,CitationDelta} &&
                    name in (:data, :url, :file_id, :title)
                )
            isempty(v) && !allowempty && throw(ArgumentError("$T.$name must not be empty"))
        end
        if haskey(VOCABULARIES, name) && v isa AbstractString
            v in VOCABULARIES[name] || throw(ArgumentError("unknown $name"))
        end
        if name === :continuation
            require_items(v, ContinuationState, "continuation")
        elseif name in (
            :stop, :allowed, :aliases, :input_modalities, :output_modalities, :reasoning_efforts
        )
            all(s -> s isa AbstractString && !isempty(s), v) ||
                throw(ArgumentError("$name must contain non-empty strings"))
        end
        if name in (
            :max_tokens,
            :top_k,
            :thinking_budget,
            :seconds,
            :sample_rate,
            :channels,
            :context_window,
            :max_output_tokens,
        )
            v > 0 || throw(ArgumentError("$name must be positive"))
        elseif v isa Real && name in (
            :part_index,
            :index,
            :logprobs,
            :prefix_until_index,
            :size_bytes,
            :tokens,
            :progress,
            :temperature,
            :input_per_million,
            :output_per_million,
            :cache_read_per_million,
            :cache_write_per_million,
        )
            v >= 0 || throw(ArgumentError("$name must not be negative"))
        end
    end
    if haskey(TAGS, T)
        x.type == TAGS[T] || throw(ArgumentError("incorrect type discriminator for $T"))
    end
    if x isa MediaPart
        count(v -> v !== nothing, (x.data, x.url, x.file_id, x.path)) == 1 ||
            throw(ArgumentError("media needs exactly one source"))
        x.data === nothing || check_base64(x.data)
    elseif x isa Union{AudioDelta,ImageDelta}
        count(v -> v !== nothing, (x.data, x.url, x.file_id)) <= 1 ||
            throw(ArgumentError("media delta has multiple sources"))
    elseif x isa Union{CitationPart,CitationDelta}
        any(v -> v !== nothing, (x.text, x.url, x.title)) ||
            throw(ArgumentError("citation needs text, title, or URL"))
        if x isa CitationPart
            all(v -> v === nothing || !isempty(v), (x.text, x.url, x.title)) ||
                throw(ArgumentError("citation fields must not be empty"))
        end
    elseif x isa RefusalPart
        isempty(x.text) && throw(ArgumentError("refusal must not be empty"))
    elseif x isa Union{ToolResultPart,LiveClientToolResultEvent}
        require_items(x.content, Part, "content"; nonempty=true)
        all(result_part, x.content) ||
            throw(ArgumentError("tool results cannot contain protocol parts"))
    elseif x isa Message
        require_items(x.parts, Part, "parts"; nonempty=true)
        x.role == "tool" &&
            !all(p -> p isa ToolResultPart, x.parts) &&
            throw(ArgumentError("tool messages contain only tool results"))
        x.role == "assistant" &&
            any(p -> p isa ToolResultPart, x.parts) &&
            throw(ArgumentError("assistant messages cannot contain tool results"))
        x.role in ("user", "developer") &&
            !all(prompt_part, x.parts) &&
            throw(ArgumentError("prompt messages cannot contain protocol parts"))
    elseif x isa Reasoning
        x.effort == "off" &&
            (x.thinking_budget !== nothing || x.summary !== nothing) &&
            throw(ArgumentError("reasoning off cannot carry a budget or summary"))
    elseif x isa CacheConfig
        x.mode in ("auto", "off") || throw(ArgumentError("unknown cache mode"))
        x.mode == "off" &&
            any(
                v -> v !== nothing, (x.retention, x.key, x.prefix, x.prefix_until_index, x.resource)
            ) &&
            throw(ArgumentError("cache off cannot carry other settings"))
        x.prefix !== nothing &&
            x.prefix_until_index !== nothing &&
            throw(ArgumentError("cache prefix settings are mutually exclusive"))
    elseif x isa ToolChoice
        x.mode in ("auto", "required", "none") || throw(ArgumentError("unknown tool choice mode"))
        x.mode == "none" &&
            (!isempty(x.allowed) || x.parallel !== nothing) &&
            throw(ArgumentError("tool choice none cannot carry allowed or parallel"))
    elseif x isa Config
        x.top_p !== nothing &&
            !(0 <= x.top_p <= 1) &&
            throw(ArgumentError("top_p must lie in [0,1]"))
        if x.response_format !== nothing
            f = x.response_format
            tag = get(f, "type", nothing)
            tag in ("json_object", "json_schema") ||
                throw(ArgumentError("response_format requires json_object or json_schema"))
            allowed = tag == "json_object" ? ("type",) : ("type", "schema", "name", "strict")
            all(k -> k in allowed, keys(f)) || throw(ArgumentError("unknown response_format field"))
            if tag == "json_schema"
                get(f, "schema", nothing) isa AbstractDict ||
                    throw(ArgumentError("json_schema needs a schema object"))
                haskey(f, "name") &&
                    !(f["name"] isa AbstractString && !isempty(f["name"])) &&
                    throw(ArgumentError("schema name must be non-empty"))
                haskey(f, "strict") &&
                    !(f["strict"] isa Bool) &&
                    throw(ArgumentError("schema strict must be boolean"))
            end
        end
    elseif x isa Usage
        all(n -> getfield(x, n) === nothing || getfield(x, n) >= 0, fieldnames(Usage)) ||
            throw(ArgumentError("usage counters must be non-negative"))
    elseif x isa Response
        x.message.role == "assistant" || throw(ArgumentError("response message must be assistant"))
        x.logprobs === nothing || require_items(x.logprobs, TokenLogprob, "logprobs")
    elseif x isa Union{TokenLogprob,TopLogprob}
        x.bytes === nothing ||
            all(v -> v isa Integer && !(v isa Bool) && v >= 0, x.bytes) ||
            throw(ArgumentError("token bytes must be non-negative integers"))
        x isa TokenLogprob && require_items(x.top, TopLogprob, "top")
    elseif x isa TextDelta
        require_items(x.logprobs, TokenLogprob, "logprobs")
    elseif x isa ErrorDetail
        haskey(ERROR_TYPES, x.code) || throw(ArgumentError("unknown error code"))
    elseif x isa FileUploadRequest
        (x.bytes_data === nothing) != (x.path === nothing) ||
            throw(ArgumentError("upload needs exactly one of bytes_data and path"))
        x.bytes_data === nothing ||
            !isempty(x.bytes_data) ||
            throw(ArgumentError("upload data must not be empty"))
    elseif x isa Union{FilePage,CachePage}
        require_items(x.items, x isa FilePage ? FileInfo : CacheInfo, "items")
    elseif x isa CachedPrefix
        isempty(to_dict(x.prefix.config)) ||
            throw(ArgumentError("cached prefix cannot have generation settings"))
        x.resource === nothing ||
            x.resource.model == x.prefix.model ||
            throw(ArgumentError("cache model differs from prefix"))
    elseif x isa BatchRequest
        require_items(x.requests, Request, "requests"; nonempty=true)
    elseif x isa BatchJobInfo
        x.status in
        ("queued", "running", "cancelling", "completed", "failed", "cancelled", "expired") ||
            throw(ArgumentError("unknown batch status"))
    elseif x isa VideoJobInfo
        x.status in ("queued", "running", "completed", "failed", "cancelled") ||
            throw(ArgumentError("unknown video status"))
        x.progress === nothing || x.progress <= 100 || throw(ArgumentError("progress exceeds 100"))
    elseif x isa BatchEntry
        ((x.outcome == "succeeded") == (x.response !== nothing)) &&
        ((x.outcome == "errored") == (x.error !== nothing)) ||
            throw(ArgumentError("batch outcome does not match response/error"))
    elseif x isa Union{ImageGenerationRequest,ImageGenerationResponse,VideoGenerationRequest}
        require_items(x.images, ImagePart, "images"; nonempty=x isa ImageGenerationResponse)
    elseif x isa LiveClientTurnEvent
        require_items(x.parts, Part, "parts"; nonempty=true)
        all(prompt_part, x.parts) || throw(ArgumentError("live turn requires prompt parts"))
    elseif x isa Union{LiveClientAudioEvent,LiveClientImageEvent,LiveServerAudioEvent}
        check_base64(x.data)
        prefix = x isa LiveClientImageEvent ? "image/" : "audio/"
        x.media_type === nothing ||
            startswith(x.media_type, prefix) ||
            throw(ArgumentError("incorrect live media type"))
    end
    if x isa Union{Request,LiveConfig}
        require_items(x.tools, Tool, "tools")
        names = [t.name for t in x.tools]
        length(unique(names)) == length(names) || throw(ArgumentError("duplicate tool names"))
        x.system isa String && isempty(x.system) && throw(ArgumentError("system must not be empty"))
        if x.system isa Tuple
            require_items(x.system, Part, "system"; nonempty=true)
            all(prompt_part, x.system) || throw(ArgumentError("system must contain prompt parts"))
        end
        if x isa Request
            require_items(x.messages, Message, "messages"; nonempty=true)
            tc = x.config.tool_choice
            tc === nothing ||
                all(n -> n in names, tc.allowed) ||
                throw(ArgumentError("tool choice names an undeclared tool"))
        end
    end
    return x
end
