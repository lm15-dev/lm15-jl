abstract type Canonical end
abstract type Part <: Canonical end
abstract type Delta <: Canonical end
abstract type StreamEvent <: Canonical end
abstract type Tool <: Canonical end
abstract type LiveClientEvent <: Canonical end
abstract type LiveServerEvent <: Canonical end
const Maybe{T} = Union{Nothing,T}
const CANONICAL_TYPES = Dict{String,DataType}()

# Inner constructors prevent positional construction from bypassing validation.
# The schema is the struct itself: serializers and surface reflection read it.
macro canonical(head, body)
    name, parent = head isa Symbol ? (head, :Canonical) : (head.args[1], head.args[2])
    fields = Any[]
    keywords = Any[]
    values = Any[]
    for line in body.args
        line isa LineNumberNode && continue
        declaration, default =
            line isa Expr && line.head === :(=) ? (line.args[1], line.args[2]) : (line, nothing)
        field, typ = declaration.args
        push!(fields, declaration)
        if line isa Expr && line.head === :(=)
            push!(keywords, Expr(:kw, field, default))
        else
            push!(keywords, field)
        end
        normalized = :(normalize_field($name, $(QuoteNode(field)), $field, $typ))
        if name === :Usage && field === :total_tokens
            normalized = :(
                if total_tokens === nothing &&
                    input_tokens !== nothing &&
                    output_tokens !== nothing
                    checked_token_sum(input_tokens, output_tokens)
                else
                    $normalized
                end
            )
        end
        push!(values, normalized)
    end
    constructor = Expr(
        :function, Expr(:call, name, Expr(:parameters, keywords...)), quote
            value = new($(values...))
            validate(value)
            value
        end
    )
    return esc(quote
        struct $name <: $parent
            $(fields...)
            $constructor
        end
        CANONICAL_TYPES[$(string(name))] = $name
    end)
end

@canonical ContinuationState begin
    provider::String
    kind::String
    data::JsonObject = obj()
end
@canonical TextPart <: Part begin
    text::String = ""
    continuation::Tuple = ()
    type::String = "text"
end
@canonical ThinkingPart <: Part begin
    text::String = ""
    continuation::Tuple = ()
    type::String = "thinking"
end
@canonical RefusalPart <: Part begin
    text::String
    continuation::Tuple = ()
    type::String = "refusal"
end
@canonical CitationPart <: Part begin
    url::Maybe{String} = nothing
    title::Maybe{String} = nothing
    text::Maybe{String} = nothing
    continuation::Tuple = ()
    type::String = "citation"
end
for (name, tag, mime) in (
    (:ImagePart, "image", "image/png"),
    (:AudioPart, "audio", "audio/wav"),
    (:VideoPart, "video", "video/mp4"),
    (:DocumentPart, "document", "application/pdf"),
    (:BinaryPart, "binary", "application/octet-stream"),
)
    extra = name === :ImagePart ? [:(detail::Maybe{String} = nothing)] : []
    @eval @canonical $name <: Part begin
        media_type::String = $mime
        data::Maybe{String} = nothing
        url::Maybe{String} = nothing
        file_id::Maybe{String} = nothing
        path::Maybe{String} = nothing
        $(extra...)
        continuation::Tuple = ()
        type::String = $tag
    end
end
const MediaPart = Union{ImagePart,AudioPart,VideoPart,DocumentPart,BinaryPart}
@canonical ToolCallPart <: Part begin
    id::String
    name::String
    input::JsonObject = obj()
    continuation::Tuple = ()
    type::String = "tool_call"
end
# Structured data as content (changes/2026-09-17-judgments.md D2): input in a
# user/system message; in an assistant message, the answer to a judgment request
# (MAP-14) with an optional measured distribution and its method (INV-052).
@canonical DataPart <: Part begin
    value::Any = nothing
    probabilities::Maybe{JsonObject} = nothing
    method::Maybe{String} = nothing
    continuation::Tuple = ()
    type::String = "data"
end
@canonical ToolResultPart <: Part begin
    id::String
    content::Tuple
    name::Maybe{String} = nothing
    is_error::Bool = false
    continuation::Tuple = ()
    type::String = "tool_result"
end
@canonical Message begin
    role::String
    parts::Tuple
    continuation::Tuple = ()
end
@canonical FunctionTool <: Tool begin
    name::String
    description::Maybe{String} = nothing
    parameters::JsonObject = obj("type"=>"object", "properties"=>obj())
    type::String = "function"
end
@canonical BuiltinTool <: Tool begin
    name::String
    config::Maybe{JsonObject} = nothing
    type::String = "builtin"
end
@canonical ToolChoice begin
    mode::String = "auto"
    allowed::Tuple = ()
    parallel::Maybe{Bool} = nothing
end
@canonical Reasoning begin
    effort::String
    thinking_budget::Maybe{Int} = nothing
    summary::Maybe{String} = nothing
end
@canonical CacheConfig begin
    mode::String = "auto"
    retention::Maybe{String} = nothing
    key::Maybe{String} = nothing
    prefix_until_index::Maybe{Int} = nothing
    prefix::Maybe{String} = nothing
    resource::Maybe{String} = nothing
end
@canonical Config begin
    max_tokens::Maybe{Int} = nothing
    temperature::Maybe{Float64} = nothing
    top_p::Maybe{Float64} = nothing
    top_k::Maybe{Int} = nothing
    seed::Maybe{Int} = nothing
    frequency_penalty::Maybe{Float64} = nothing
    presence_penalty::Maybe{Float64} = nothing
    stop::Tuple = ()
    response_format::Maybe{JsonObject} = nothing
    tool_choice::Maybe{ToolChoice} = nothing
    reasoning::Maybe{Reasoning} = nothing
    cache::Maybe{CacheConfig} = nothing
    service_tier::Maybe{String} = nothing
    user_id::Maybe{String} = nothing
    store::Maybe{Bool} = nothing
    logprobs::Maybe{Int} = nothing
    probabilities::Maybe{String} = nothing
    extensions::Maybe{JsonObject} = nothing
end
@canonical Request begin
    model::String
    messages::Tuple
    system::Union{Nothing,String,Tuple} = nothing
    tools::Tuple = ()
    config::Config = Config()
end
@canonical Usage begin
    input_tokens::Maybe{Int} = nothing
    output_tokens::Maybe{Int} = nothing
    total_tokens::Maybe{Int} = if input_tokens === nothing || output_tokens === nothing
        nothing
    else
        checked_token_sum(input_tokens, output_tokens)
    end
    cache_read_tokens::Maybe{Int} = nothing
    cache_write_tokens::Maybe{Int} = nothing
    reasoning_tokens::Maybe{Int} = nothing
    input_audio_tokens::Maybe{Int} = nothing
    output_audio_tokens::Maybe{Int} = nothing
end
@canonical TopLogprob begin
    token::String
    logprob::Float64
    bytes::Maybe{Tuple} = nothing
    token_id::Maybe{Int} = nothing
end
@canonical TokenLogprob begin
    token::String
    logprob::Float64
    bytes::Maybe{Tuple} = nothing
    token_id::Maybe{Int} = nothing
    top::Tuple = ()
end
@canonical Response begin
    id::Maybe{String} = nothing
    model::String
    message::Message
    finish_reason::String
    usage::Usage = Usage()
    logprobs::Maybe{Tuple} = nothing
    logprobs_complete::Bool = true
    provider_data::Maybe{JsonObject} = nothing
    adaptations::Tuple = ()
end
# MAP-13: one thing the wire got that differs from what was asked.
@canonical Adaptation begin
    field::String
    action::String
    reason::String
    asked::Any = nothing
    applied::Any = nothing
end
@canonical TextDelta <: Delta begin
    text::String = ""
    part_index::Int = 0
    logprobs::Tuple = ()
    logprobs_complete::Bool = true
    type::String = "text"
end
@canonical ThinkingDelta <: Delta begin
    text::String = ""
    part_index::Int = 0
    type::String = "thinking"
end
for (name, tag) in ((:AudioDelta, "audio"), (:ImageDelta, "image"))
    @eval @canonical $name <: Delta begin
        data::Maybe{String} = nothing
        url::Maybe{String} = nothing
        file_id::Maybe{String} = nothing
        part_index::Int = 0
        media_type::Maybe{String} = nothing
        type::String = $tag
    end
end
@canonical ToolCallDelta <: Delta begin
    input::String = ""
    part_index::Int = 0
    id::Maybe{String} = nothing
    name::Maybe{String} = nothing
    type::String = "tool_call"
end
@canonical CitationDelta <: Delta begin
    text::Maybe{String} = nothing
    url::Maybe{String} = nothing
    title::Maybe{String} = nothing
    part_index::Int = 0
    type::String = "citation"
end
@canonical ContinuationDelta <: Delta begin
    provider::String
    kind::String
    data::JsonObject = obj()
    part_index::Maybe{Int} = nothing
    type::String = "continuation"
end
@canonical ErrorDetail begin
    code::String
    message::String = ""
    provider_code::Maybe{String} = nothing
    http_response::Maybe{JsonObject} = nothing
end
@canonical StreamStartEvent <: StreamEvent begin
    id::Maybe{String} = nothing
    model::Maybe{String} = nothing
    adaptations::Tuple = ()
    type::String = "start"
end
@canonical StreamDeltaEvent <: StreamEvent begin
    delta::Delta
    type::String = "delta"
end
@canonical StreamEndEvent <: StreamEvent begin
    finish_reason::Maybe{String} = nothing
    usage::Maybe{Usage} = nothing
    provider_data::Maybe{JsonObject} = nothing
    type::String = "end"
end
@canonical StreamErrorEvent <: StreamEvent begin
    error::ErrorDetail
    type::String = "error"
end

@canonical FileUploadRequest begin
    filename::String
    bytes_data::Maybe{Vector{UInt8}} = nothing
    media_type::String = "application/octet-stream"
    extensions::Maybe{JsonObject} = nothing
    path::Maybe{String} = nothing
end
@canonical FileInfo begin
    id::String
    filename::Maybe{String} = nothing
    media_type::Maybe{String} = nothing
    size_bytes::Maybe{Int} = nothing
    created_at::Maybe{String} = nothing
    expires_at::Maybe{String} = nothing
    readiness::String = "ready"
    downloadable::Maybe{Bool} = nothing
    provider_data::Maybe{JsonObject} = nothing
end
@canonical FilePage begin
    items::Tuple = ()
    next_cursor::Maybe{String} = nothing
end
@canonical CacheInfo begin
    id::String
    model::String
    tokens::Maybe{Int} = nothing
    created_at::Maybe{String} = nothing
    expires_at::Maybe{String} = nothing
    label::Maybe{String} = nothing
    provider_data::Maybe{JsonObject} = nothing
end
@canonical CachePage begin
    items::Tuple = ()
    next_cursor::Maybe{String} = nothing
end
@canonical CachedPrefix begin
    prefix::Request
    resource::Maybe{CacheInfo} = nothing
    provider::Maybe{String} = nothing
end
@canonical BatchRequest begin
    requests::Tuple
    model::String = isempty(requests) ? "" : first(requests).model
    label::Maybe{String} = nothing
    extensions::Maybe{JsonObject} = nothing
end
@canonical BatchJobInfo begin
    id::String
    status::String
    label::Maybe{String} = nothing
    created_at::Maybe{String} = nothing
    provider_data::Maybe{JsonObject} = nothing
end
@canonical BatchEntry begin
    index::Int
    outcome::String
    response::Maybe{Response} = nothing
    error::Maybe{ErrorDetail} = nothing
end
@canonical ImageGenerationRequest begin
    model::String
    prompt::String
    size::Maybe{String} = nothing
    images::Tuple = ()
    extensions::Maybe{JsonObject} = nothing
end
@canonical ImageGenerationResponse begin
    images::Tuple
    text::Maybe{String} = nothing
    id::Maybe{String} = nothing
    model::Maybe{String} = nothing
    usage::Usage = Usage()
    provider_data::Maybe{JsonObject} = nothing
end
@canonical SpeechGenerationRequest begin
    model::String
    prompt::String
    voice::Maybe{String} = nothing
    format::Maybe{String} = nothing
    extensions::Maybe{JsonObject} = nothing
end
@canonical SpeechGenerationResponse begin
    audio::AudioPart
    id::Maybe{String} = nothing
    model::Maybe{String} = nothing
    usage::Usage = Usage()
    provider_data::Maybe{JsonObject} = nothing
end
@canonical VideoGenerationRequest begin
    model::String
    prompt::String
    seconds::Maybe{Int} = nothing
    images::Tuple = ()
    extensions::Maybe{JsonObject} = nothing
end
@canonical VideoJobInfo begin
    id::String
    status::String
    progress::Maybe{Int} = nothing
    created_at::Maybe{String} = nothing
    model::Maybe{String} = nothing
    provider_data::Maybe{JsonObject} = nothing
end
@canonical AudioFormat begin
    encoding::String
    sample_rate::Int
    channels::Int = 1
end
@canonical LiveConfig begin
    model::String
    system::Union{Nothing,String,Tuple} = nothing
    tools::Tuple = ()
    voice::Maybe{String} = nothing
    input_format::Maybe{AudioFormat} = nothing
    output_format::Maybe{AudioFormat} = nothing
    extensions::Maybe{JsonObject} = nothing
end
@canonical LiveClientTurnEvent <: LiveClientEvent begin
    parts::Tuple
    turn_complete::Bool = true
    type::String = "turn"
end
@canonical LiveClientAudioEvent <: LiveClientEvent begin
    data::String
    media_type::String = "audio/pcm;rate=16000"
    type::String = "audio"
end
@canonical LiveClientImageEvent <: LiveClientEvent begin
    data::String
    media_type::String = "image/jpeg"
    type::String = "image"
end
@canonical LiveClientTextEvent <: LiveClientEvent begin
    text::String = ""
    type::String = "text"
end
@canonical LiveClientToolResultEvent <: LiveClientEvent begin
    id::String
    content::Tuple
    type::String = "tool_result"
end
@canonical LiveClientInterruptEvent <: LiveClientEvent begin
    type::String = "interrupt"
end
@canonical LiveClientEndAudioEvent <: LiveClientEvent begin
    type::String = "end_audio"
end
@canonical LiveServerAudioEvent <: LiveServerEvent begin
    data::String
    media_type::Maybe{String} = nothing
    type::String = "audio"
end
@canonical LiveServerTextEvent <: LiveServerEvent begin
    text::String = ""
    type::String = "text"
end
@canonical LiveServerToolCallEvent <: LiveServerEvent begin
    id::String
    name::String
    input::JsonObject = obj()
    type::String = "tool_call"
end
@canonical LiveServerToolCallDeltaEvent <: LiveServerEvent begin
    input_delta::String = ""
    id::Maybe{String} = nothing
    name::Maybe{String} = nothing
    type::String = "tool_call_delta"
end
@canonical LiveServerInterruptedEvent <: LiveServerEvent begin
    type::String = "interrupted"
end
@canonical LiveServerTurnEndEvent <: LiveServerEvent begin
    usage::Usage = Usage()
    type::String = "turn_end"
end
@canonical LiveServerUsageEvent <: LiveServerEvent begin
    usage::Usage = Usage()
    type::String = "usage"
end
@canonical LiveServerErrorEvent <: LiveServerEvent begin
    error::ErrorDetail
    type::String = "error"
end
@canonical ToolCallInfo begin
    id::String
    name::String
    input::JsonObject
end
@canonical InferencePricing begin
    input_per_million::Maybe{Float64} = nothing
    output_per_million::Maybe{Float64} = nothing
    cache_read_per_million::Maybe{Float64} = nothing
    cache_write_per_million::Maybe{Float64} = nothing
    currency::String = "USD"
    dimensions::Maybe{JsonObject} = nothing
end
@canonical InferenceModelInfo begin
    input_modalities::Tuple = ("text",)
    output_modalities::Tuple = ("text",)
    context_window::Maybe{Int} = nothing
    max_output_tokens::Maybe{Int} = nothing
    supports_reasoning::Bool = false
    reasoning_efforts::Tuple = ()
    pricing::Maybe{InferencePricing} = nothing
    extensions::Maybe{JsonObject} = nothing
end
@canonical ModelOrigin begin
    type::String = "provider"
    id::Maybe{String} = nothing
    base_model::Maybe{String} = nothing
    provider_data::Maybe{JsonObject} = nothing
end
@canonical ModelInfo begin
    id::String
    provider::String
    api_family::String
    aliases::Tuple = ()
    origin::ModelOrigin = ModelOrigin()
    inference::Maybe{InferenceModelInfo} = nothing
    extensions::Maybe{JsonObject} = nothing
end

kind(x::Union{Part,Delta,StreamEvent,Tool,LiveClientEvent,LiveServerEvent}) = x.type
TextPart(content::AbstractString; kw...) = TextPart(; text=content, kw...)
ThinkingPart(content::AbstractString; kw...) = ThinkingPart(; text=content, kw...)
RefusalPart(content::AbstractString; kw...) = RefusalPart(; text=content, kw...)
Message(role::AbstractString, parts; kw...) = Message(; role, parts, kw...)
Request(model::AbstractString, messages; kw...) = Request(; model, messages, kw...)
text(content::AbstractString; kw...) = TextPart(content; kw...)
thinking(content::AbstractString; kw...) = ThinkingPart(content; kw...)
refusal(content::AbstractString; kw...) = RefusalPart(content; kw...)
citation(; kw...) = CitationPart(; kw...)
"""
    data(value; probabilities=nothing, method=nothing, continuation=())

A `DataPart`: structured JSON content. As input it is a JSON value the provider reads
as such (or its compact JSON text on a text-only wire); `probabilities` and `method`
belong to an assistant's answer only (INV-052).
"""
data(value; kw...) = DataPart(; value, kw...)
function normalize_content(content)
    content isa AbstractString && return (TextPart(content),)
    content isa Part && return (content,)
    content isa Union{Tuple,AbstractVector} ||
        throw(ArgumentError("content must be text, a Part, or a sequence"))
    isempty(content) && throw(ArgumentError("content must not be empty"))
    return Tuple(p isa AbstractString ? TextPart(p) : p for p in content)
end
user(content) = Message("user", normalize_content(content))
assistant(content) = Message("assistant", normalize_content(content))
developer(content) = Message("developer", normalize_content(content))
tool_call(id, name, input; kw...) = ToolCallPart(; id, name, input, kw...)
tool_result(id, content; kw...) = ToolResultPart(; id, content=normalize_content(content), kw...)
tool_message(id::AbstractString, output; kw...) = Message("tool", (tool_result(id, output; kw...),))
tool_message(results::ToolResultPart...) = Message("tool", results)
function tool_message(results::AbstractDict)
    return Message("tool", Tuple(tool_result(id, value) for (id, value) in results))
end
for (fn, T) in (
    (:image, :ImagePart),
    (:audio, :AudioPart),
    (:video, :VideoPart),
    (:document, :DocumentPart),
    (:binary, :BinaryPart),
)
    @eval $fn(; kwargs...) = media_part($T; kwargs...)
end
function bytes(p::MediaPart)
    p.data !== nothing && return base64decode(base64_payload(p.data))
    p.path !== nothing && return read(p.path)
    return throw(ArgumentError("URL and file-id media must be fetched explicitly"))
end
bytes(r::FileUploadRequest) = r.bytes_data === nothing ? read(r.path) : copy(r.bytes_data)
parts_of(::Type{T}, m::Message) where {T<:Part} = T[p for p in m.parts if p isa T]
function text(m::Message)
    all(p -> p isa TextPart, m.parts) || return nothing
    return join((p.text for p in m.parts), "\n")
end
function text(r::Response)
    all(p -> p isa Union{TextPart,ThinkingPart,CitationPart}, r.message.parts) || return nothing
    parts = parts_of(TextPart, r.message)
    return isempty(parts) ? nothing : join((p.text for p in parts), "\n")
end
tool_calls(r::Response) = parts_of(ToolCallPart, r.message)
citations(r::Response) = parts_of(CitationPart, r.message)
function parse_json(r::Response)
    value = text(r)
    value === nothing && throw(ArgumentError("response is not text"))
    return JSON.parse(value)
end
function continuation_data(value, provider, kind)
    states = value isa Tuple ? value : value.continuation
    for state in states
        state.provider == provider && state.kind == kind && return state.data
    end
    return nothing
end
function reconstruct(x::T; kwargs...) where {T<:Canonical}
    names = fieldnames(T)
    unknown = setdiff(keys(kwargs), names)
    isempty(unknown) || throw(ArgumentError("unknown $T field(s): $(join(unknown, ", "))"))
    fields = NamedTuple{names}(ntuple(i -> getfield(x, i), fieldcount(T)))
    return T(; merge(fields, (; kwargs...))...)
end
