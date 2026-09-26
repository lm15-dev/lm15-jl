# Convenience methods preserve the canonical Request/Response boundary. They do
# not introduce a second conversation representation or perform hidden I/O.

const ContentAtom = Union{AbstractString,Part}

for factory in (:user, :assistant, :developer)
    @eval $factory(first::ContentAtom, second::ContentAtom, rest::ContentAtom...) =
        $factory((first, second, rest...))
end

"""
    Request(model, message::Message, more::Message...; system=nothing, tools=(), config=Config())
    Request(existing::Request; changes...)

Build a request from ordinary messages, or make a validated copy with selected
fields changed. A single `user("hello")` needs neither an array nor a one-tuple.
Generation settings remain explicit in `config`.
"""
function Request(model::AbstractString, message::Message, more::Message...; kwargs...)
    return Request(; model, messages=(message, more...), kwargs...)
end
Request(model::AbstractString; messages, kwargs...) = Request(; model, messages, kwargs...)

# A same-type constructor is Julia's copy-and-change spelling. Validation still
# runs, and opaque dictionaries remain shallowly shared by contract (INV-002).
for T in values(CANONICAL_TYPES)
    @eval (::Type{$T})(value::$T; changes...) = reconstruct(value; changes...)
end

"""
    tool_result(call::ToolCallPart, content; is_error=false)
    tool_message(call::ToolCallPart, content; is_error=false)

Answer a call while preserving its id and function name. The name is needed by
Gemini even when the surrounding transcript is not available to the adapter.
"""
function tool_result(call::ToolCallPart, content; name=call.name, kwargs...)
    return tool_result(call.id, content; name, kwargs...)
end
function tool_message(call::ToolCallPart, content; kwargs...)
    return tool_message(tool_result(call, content; kwargs...))
end
ToolCallInfo(part::ToolCallPart) = ToolCallInfo(; id=part.id, name=part.name, input=part.input)
function ToolCallPart(info::ToolCallInfo; kwargs...)
    return ToolCallPart(; id=info.id, name=info.name, input=info.input, kwargs...)
end
parts_of(::Type{T}, answer::Response) where {T<:Part} = parts_of(T, answer.message)

# Deterministic extension lookup, not OS configuration or eager file sniffing.
# The constructor default remains authoritative when the path gives no hint.
const MEDIA_PATH_TYPES = Dict(
    ".png" => "image/png",
    ".jpg" => "image/jpeg",
    ".jpeg" => "image/jpeg",
    ".gif" => "image/gif",
    ".webp" => "image/webp",
    ".svg" => "image/svg+xml",
    ".bmp" => "image/bmp",
    ".tif" => "image/tiff",
    ".tiff" => "image/tiff",
    ".avif" => "image/avif",
    ".heic" => "image/heic",
    ".apng" => "image/apng",
    ".wav" => "audio/wav",
    ".mp3" => "audio/mpeg",
    ".ogg" => "audio/ogg",
    ".opus" => "audio/opus",
    ".flac" => "audio/flac",
    ".aac" => "audio/aac",
    ".m4a" => "audio/mp4",
    ".mp4" => "video/mp4",
    ".webm" => "video/webm",
    ".mov" => "video/quicktime",
    ".mpeg" => "video/mpeg",
    ".mpg" => "video/mpeg",
    ".pdf" => "application/pdf",
    ".txt" => "text/plain",
    ".md" => "text/markdown",
    ".csv" => "text/csv",
    ".json" => "application/json",
    ".html" => "text/html",
    ".bin" => "application/octet-stream",
)

function media_part(
    ::Type{T};
    data=nothing,
    url=nothing,
    file_id=nothing,
    path=nothing,
    media_type=nothing,
    kwargs...,
) where {T<:MediaPart}
    count(!isnothing, (data, url, file_id, path)) == 1 ||
        throw(ArgumentError("media requires exactly one of data, url, file_id, or path"))
    encoded = data isa AbstractVector{UInt8} ? base64encode(data) : data
    if media_type === nothing && path isa AbstractString
        media_type = get(MEDIA_PATH_TYPES, lowercase(last(splitext(path))), nothing)
    end
    if media_type === nothing
        return T(; data=encoded, url, file_id, path, kwargs...)
    end
    return T(; data=encoded, url, file_id, path, media_type, kwargs...)
end

"""
    stream(client, request) do response_stream
        for fragment in text_chunks(response_stream)
            print(fragment)
        end
        response(response_stream)
    end

The scoped form assembles the same canonical events as `stream(client, request)`
and closes the stream on every exit. It returns the block's result. Early exit
never submits another request and does not guarantee that provider billing stops.
"""
function stream(f, client::Union{ProviderLM,LMRouter}, request::Request)
    return ResponseStream(f, stream(client, request), request)
end

function live(f, router::LMRouter, config::LiveConfig)
    resolution = resolve(router, config.model)
    return live(f, lm(router, config.model), LiveConfig(config; model=resolution.model))
end

function explain_auth(resolution::Resolution; kwargs...)
    return explain_auth(resolution.provider; kwargs...)
end
function explain_auth(router::LMRouter, model::AbstractString)
    resolution = resolve(router, model)
    config = router.config
    config.auth === nothing || return explain_managed(resolution.provider; auth=config.auth,
        env=config.env === nothing ? ENV : config.env, api_keys=config.api_keys, credentials=config.credentials)
    return explain_auth(
        resolution.provider;
        env=config.env,
        api_keys=config.api_keys,
        settings=provider_setting(config.settings, resolution.provider, Dict{String,String}()),
        credential=provider_setting(config.credentials, resolution.provider),
        base_url=provider_setting(config.base_urls, resolution.provider),
    )
end

function Base.show(io::IO, part::MediaPart)
    source = if part.data !== nothing
        "inline data, $(ncodeunits(part.data)) characters"
    elseif part.path !== nothing
        "local file"
    elseif part.url !== nothing
        "URL"
    else
        "stored file"
    end
    return print(io, nameof(typeof(part)), "(", repr(part.media_type), ", ", source, ")")
end
function Base.show(io::IO, state::ContinuationState)
    return print(
        io, "ContinuationState(", repr(state.provider), ", ", repr(state.kind), ", <opaque data>)"
    )
end
function Base.show(io::IO, answer::Response)
    return print(
        io,
        "Response(",
        repr(answer.model),
        ", ",
        repr(answer.finish_reason),
        ", ",
        length(answer.message.parts),
        " parts)",
    )
end
function Base.show(io::IO, ::MIME"text/plain", answer::Response)
    show(io, answer)
    visible = text(answer)
    if visible !== nothing
        limit = get(io, :limit, false) ? 400 : 4000
        print(io, '\n', first(visible, limit))
        length(visible) > limit && print(io, "…")
    else
        for part in answer.message.parts
            print(io, "\n  ", kind(part))
            part isa ToolCallPart && print(io, ": ", part.name, " (", part.id, ")")
        end
    end
    usage = answer.usage
    if usage.input_tokens !== nothing || usage.output_tokens !== nothing
        print(
            io,
            "\nTokens: input ",
            something(usage.input_tokens, "unreported"),
            ", output ",
            something(usage.output_tokens, "unreported"),
        )
    end
end
