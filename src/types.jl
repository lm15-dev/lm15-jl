# ── DataSource ─────────────────────────────────────────────────────

struct DataSource
    type::String
    media_type::Union{String,Nothing}
    data::Union{String,Nothing}
    url::Union{String,Nothing}
    file_id::Union{String,Nothing}
    detail::Union{String,Nothing}
end

DataSource(; type, media_type=nothing, data=nothing, url=nothing, file_id=nothing, detail=nothing) =
    DataSource(type, media_type, data, url, file_id, detail)

"""Decode base64 data to bytes."""
function decode_bytes(ds::DataSource)
    ds.type == "base64" && ds.data !== nothing || error("DataSource(type=$(ds.type)) has no inline bytes")
    base64decode(ds.data)
end

# ── Part ───────────────────────────────────────────────────────────

struct Part
    type::String
    text::Union{String,Nothing}
    source::Union{DataSource,Nothing}
    id::Union{String,Nothing}
    name::Union{String,Nothing}
    input::Union{Dict{String,Any},Nothing}
    content::Union{Vector{Part},Nothing}
    is_error::Union{Bool,Nothing}
    redacted::Union{Bool,Nothing}
    summary::Union{String,Nothing}
    url::Union{String,Nothing}
    title::Union{String,Nothing}
    metadata::Union{Dict{String,Any},Nothing}
end

Part(; type="text", text=nothing, source=nothing, id=nothing, name=nothing,
       input=nothing, content=nothing, is_error=nothing, redacted=nothing,
       summary=nothing, url=nothing, title=nothing, metadata=nothing) =
    Part(type, text, source, id, name, input, content, is_error, redacted, summary, url, title, metadata)

# Factory functions
TextPart(text::String) = Part(type="text", text=text)
ThinkingPart(text::String; redacted=nothing, summary=nothing) = Part(type="thinking", text=text, redacted=redacted, summary=summary)
RefusalPart(text::String) = Part(type="refusal", text=text)
CitationPart(; text=nothing, url=nothing, title=nothing) = Part(type="citation", text=text, url=url, title=title)

ImageURL(url::String) = Part(type="image", source=DataSource(type="url", url=url, media_type="image/png"))
ImageBase64(data::String, media_type::String) = Part(type="image", source=DataSource(type="base64", data=data, media_type=media_type))
AudioBase64(data::String, media_type::String) = Part(type="audio", source=DataSource(type="base64", data=data, media_type=media_type))
DocumentURL(url::String) = Part(type="document", source=DataSource(type="url", url=url, media_type="application/pdf"))

ToolCallPart(id::String, name::String, input::Dict{String,Any}) =
    Part(type="tool_call", id=id, name=name, input=input)
ToolResultPart(id::String, content::Vector{Part}; name=nothing) =
    Part(type="tool_result", id=id, content=content, name=name)

# ── Tool ───────────────────────────────────────────────────────────

struct Tool
    type::String
    name::String
    description::Union{String,Nothing}
    parameters::Union{Dict{String,Any},Nothing}
    fn_::Union{Function,Nothing}
end

FunctionTool(name, description; parameters=Dict{String,Any}("type"=>"object","properties"=>Dict{String,Any}()), fn_=nothing) =
    Tool("function", name, description, parameters, fn_)

BuiltinTool(name) = Tool("builtin", name, nothing, nothing, nothing)

struct ToolCallInfo
    id::String
    name::String
    input::Dict{String,Any}
end

# ── Config ─────────────────────────────────────────────────────────

struct Config
    max_tokens::Union{Int,Nothing}
    temperature::Union{Float64,Nothing}
    top_p::Union{Float64,Nothing}
    stop::Union{Vector{String},Nothing}
    reasoning::Union{Dict{String,Any},Nothing}
    provider::Union{Dict{String,Any},Nothing}
    response_format::Union{Dict{String,Any},Nothing}
end

Config(; max_tokens=nothing, temperature=nothing, top_p=nothing, stop=nothing,
         reasoning=nothing, provider=nothing, response_format=nothing) =
    Config(max_tokens, temperature, top_p, stop, reasoning, provider, response_format)

# ── Message ────────────────────────────────────────────────────────

struct Message
    role::String
    parts::Vector{Part}
    name::Union{String,Nothing}
end

Message(; role, parts, name=nothing) = Message(role, parts, name)

UserMessage(text::String) = Message(role="user", parts=[TextPart(text)])
AssistantMessage(text::String) = Message(role="assistant", parts=[TextPart(text)])
function ToolResultMessage(results::Vector{Pair{String,String}})
    parts = [ToolResultPart(id, [TextPart(val)]) for (id, val) in results]
    Message(role="tool", parts=parts)
end

# ── Request / Response ─────────────────────────────────────────────

struct LMRequest
    model::String
    messages::Vector{Message}
    system::Union{String,Nothing}
    tools::Vector{Tool}
    config::Config
end

LMRequest(; model, messages, system=nothing, tools=Tool[], config=Config()) =
    LMRequest(model, messages, system, tools, config)

struct Usage
    input_tokens::Int
    output_tokens::Int
    total_tokens::Int
    cache_read_tokens::Union{Int,Nothing}
    cache_write_tokens::Union{Int,Nothing}
    reasoning_tokens::Union{Int,Nothing}
    input_audio_tokens::Union{Int,Nothing}
    output_audio_tokens::Union{Int,Nothing}
end

Usage(; input_tokens=0, output_tokens=0, total_tokens=0,
        cache_read_tokens=nothing, cache_write_tokens=nothing,
        reasoning_tokens=nothing, input_audio_tokens=nothing,
        output_audio_tokens=nothing) =
    Usage(input_tokens, output_tokens, total_tokens,
          cache_read_tokens, cache_write_tokens, reasoning_tokens,
          input_audio_tokens, output_audio_tokens)

struct LMResponse
    id::String
    model::String
    message::Message
    finish_reason::String
    usage::Usage
    provider::Union{Dict{String,Any},Nothing}
end

# Accessors
function text(r::LMResponse)
    texts = [p.text for p in r.message.parts if p.type == "text" && p.text !== nothing]
    isempty(texts) ? nothing : join(texts, "\n")
end

function thinking(r::LMResponse)
    texts = [p.text for p in r.message.parts if p.type == "thinking" && p.text !== nothing]
    isempty(texts) ? nothing : join(texts, "\n")
end

tool_calls(r::LMResponse) = [p for p in r.message.parts if p.type == "tool_call"]
citations(r::LMResponse) = [p for p in r.message.parts if p.type == "citation"]
image(r::LMResponse) = findfirst(p -> p.type == "image", r.message.parts) |> i -> i === nothing ? nothing : r.message.parts[i]
audio(r::LMResponse) = findfirst(p -> p.type == "audio", r.message.parts) |> i -> i === nothing ? nothing : r.message.parts[i]

function json(r::LMResponse)
    t = text(r)
    t === nothing && error("response contains no text")
    JSON.parse(t)
end

function image_bytes(r::LMResponse)
    img = image(r)
    img === nothing && error("response contains no image")
    decode_bytes(img.source)
end

function audio_bytes(r::LMResponse)
    aud = audio(r)
    aud === nothing && error("response contains no audio")
    decode_bytes(aud.source)
end

# ── Streaming ──────────────────────────────────────────────────────

struct ErrorInfo
    code::String
    message::String
    provider_code::Union{String,Nothing}
end

struct PartDelta
    type::String
    text::Union{String,Nothing}
    data::Union{String,Nothing}
    input::Union{String,Nothing}
end

struct StreamEvent
    type::String
    id::Union{String,Nothing}
    model::Union{String,Nothing}
    part_index::Union{Int,Nothing}
    delta::Union{PartDelta,Nothing}
    delta_raw::Union{Dict{String,Any},Nothing}
    part_type::Union{String,Nothing}
    finish_reason::Union{String,Nothing}
    usage::Union{Usage,Nothing}
    error::Union{ErrorInfo,Nothing}
end

StreamEvent(; type, id=nothing, model=nothing, part_index=nothing, delta=nothing,
              delta_raw=nothing, part_type=nothing, finish_reason=nothing,
              usage=nothing, error=nothing) =
    StreamEvent(type, id, model, part_index, delta, delta_raw, part_type, finish_reason, usage, error)

struct StreamChunk
    type::String
    text::Union{String,Nothing}
    name::Union{String,Nothing}
    input::Union{Dict{String,Any},Nothing}
    response::Union{LMResponse,Nothing}
end

# ── Auxiliary ───────────────────────────────────────────────────────

struct EmbeddingRequest
    model::String
    inputs::Vector{String}
    provider::Union{Dict{String,Any},Nothing}
end

struct EmbeddingResponse
    model::String
    vectors::Vector{Vector{Float64}}
    usage::Usage
    provider::Union{Dict{String,Any},Nothing}
end

struct FileUploadRequest
    model::Union{String,Nothing}
    filename::String
    bytes_data::Vector{UInt8}
    media_type::String
end

struct FileUploadResponse
    id::String
    provider::Union{Dict{String,Any},Nothing}
end

struct ImageGenerationRequest
    model::String
    prompt::String
    size::Union{String,Nothing}
    provider::Union{Dict{String,Any},Nothing}
end

ImageGenerationRequest(; model, prompt, size=nothing, provider=nothing) =
    ImageGenerationRequest(model, prompt, size, provider)
