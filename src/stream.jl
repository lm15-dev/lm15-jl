struct SSEEvent
    event::Maybe{String}
    data::String
end
# A chunked HTTP stream can discover its terminal chunk during a read, after
# eof(io) returned false. Bulk reads represent that as an empty buffer; UInt8
# reads throw EOFError instead. Read currently available bytes with a strict
# allocation cap, never wait for an arbitrarily large provider chunk to fill.
function sse_chunk(io::IO)
    try
        while true
            eof(io) && return nothing
            chunk = read(io, clamp(bytesavailable(io), 1, 16 * 1024))
            isempty(chunk) || return chunk
            # HTTP can consume a chunk header before its payload arrives and
            # then return an empty read without being at EOF. The next eof
            # call waits for payload; an empty read alone is not a failure.
            yield()
        end
    catch error
        error isa InterruptException && rethrow()
        throw(TransportError("connection ended while reading the event stream"))
    end
end

function parse_sse(emit, io::IO; max_line_bytes=64*1024, max_event_bytes=1024*1024)
    integer_value(max_line_bytes) > 0 && integer_value(max_event_bytes) > 0 ||
        throw(ArgumentError("SSE limits must be positive"))
    name=nothing
    lines=String[]
    line=UInt8[]
    eventbytes=0
    previous_cr = false
    function accept(raw)
        s=String(copy(raw))
        isvalid(s) || (s=join(isvalid(c) ? string(c) : "\ufffd" for c in s))
        s=rstrip(s, ['\r', '\n'])
        if isempty(s)
            isempty(lines) || emit(SSEEvent(name, join(lines, "\n")))
            name=nothing
            empty!(lines)
            eventbytes=0
        elseif startswith(s, "event:")
            name=String(strip(s[7:end]))
        elseif startswith(s, "data:")
            push!(lines, String(lstrip(s[6:end])))
        end
    end
    while true
        chunk = sse_chunk(io)
        chunk === nothing && break
        for b in chunk
            if previous_cr && b == 0x0a
                previous_cr = false
                continue
            end
            push!(line, b)
            eventbytes+=1
            length(line)<=max_line_bytes ||
                throw(TransportError("SSE line exceeds configured limit"))
            eventbytes<=max_event_bytes ||
                throw(TransportError("SSE event exceeds configured limit"))
            previous_cr = b == 0x0d
            if b in (0x0a, 0x0d)
                accept(line)
                empty!(line)
            end
        end
    end
    isempty(line) || accept(line)
    isempty(lines) || emit(SSEEvent(name, join(lines, "\n")))
    return nothing
end
parse_sse(io::IO; kw...) = (out=SSEEvent[]; parse_sse(e->push!(out, e), io; kw...); out)

# Lazy, bounded task-backed iterator. Closing it closes the active connection;
# abandoning iteration without close is not sufficient to stop a blocked read.
mutable struct EventStream
    producer::Function
    channel::Union{Nothing,Channel{Any}}
    cancellation::Vector{Function}
    closed::Bool
    cleanup_errors::Vector{Exception}
end
EventStream(f::Function) = EventStream(f, nothing, Function[], false, Exception[])
function Base.isopen(s::EventStream)
    return !s.closed && (s.channel === nothing || isopen(s.channel) || isready(s.channel))
end
Base.IteratorSize(::Type{EventStream}) = Base.SizeUnknown()
Base.eltype(::Type{EventStream}) = StreamEvent
function Base.iterate(s::EventStream, state=nothing)
    s.closed && return nothing
    if s.channel===nothing
        s.channel=Channel{Any}(1) do ch
            try
                s.producer(event->put!(ch, event), s.cancellation)
            catch e
                if isopen(ch) && !s.closed
                    try
                        put!(ch, e)
                    catch
                    end
                end
            end
        end
    end
    item=try
        take!(s.channel)
    catch e
        e isa InvalidStateException && !isopen(s.channel) && return nothing
        rethrow()
    end
    if item isa Exception
        try
            close(s)
        catch cleanup
            item isa LM15Error && push!(item.cleanup_errors, cleanup)
        end
        throw(item)
    end
    return item, nothing
end
function Base.close(s::EventStream)
    s.closed && return nothing
    s.closed=true
    for cancel in s.cancellation
        try
            cancel()
        catch error
            push!(s.cleanup_errors, error)
        end
    end
    s.channel===nothing || !isopen(s.channel) || close(s.channel)
    isempty(s.cleanup_errors) || throw(first(s.cleanup_errors))
    return nothing
end

function source_closer(source)
    closed = false
    mutex = ReentrantLock()
    () -> lock(mutex) do
        closed && return nothing
        closed = true
        applicable(close, source) && close(source)
        return nothing
    end
end

function coalesce_stream(source; model=nothing)
    EventStream() do emit, cancel
        close_source = source_closer(source)
        push!(cancel, close_source)
        started=false
        saw_end=false
        finish=nothing
        usage=nothing
        data=nothing
        rank=-1
        primary = nothing
        completed = false
        try
            for event in source
                if event isa StreamStartEvent
                    started && continue
                    started=true
                    emit(event)
                elseif event isa StreamEndEvent
                    saw_end=true
                    event.finish_reason===nothing || (finish=event.finish_reason)
                    event.usage===nothing || (usage=event.usage)
                    newrank=if event.usage!==nothing
                        2
                    elseif event.finish_reason!==nothing
                        1
                    else
                        0
                    end
                    if event.provider_data!==nothing && newrank>=rank
                        data=event.provider_data
                        rank=newrank
                    end
                else
                    if !started && event isa StreamDeltaEvent
                        emit(StreamStartEvent(; model))
                        started=true
                    end
                    emit(event)
                end
            end
            if saw_end
                started || emit(StreamStartEvent(; model))
                emit(StreamEndEvent(; finish_reason=finish, usage, provider_data=data))
                completed = true
            end
        catch error
            primary = error
            rethrow()
        finally
            try
                close_source()
            catch cleanup
                if primary !== nothing
                    primary isa LM15Error && push!(primary.cleanup_errors, cleanup)
                elseif completed
                    @warn "Stream source cleanup failed after the end event; answer preserved"
                else
                    rethrow()
                end
            end
        end
    end
end
function stream(l::ProviderLM, r::Request)
    require_surface(l, :stream)
    raw=EventStream() do emit, cancel
        wire=build_request(l, r; stream=true)
        open_response(client_transport(l), wire) do head, io
            stop_read = source_closer(io)
            push!(cancel, stop_read)
            try
                if head.status >= 400
                    body = read(io)
                    throw(
                        attach_error_metadata(normalize_error(l, head.status, String(body)), head)
                    )
                end
                parse_sse(io) do frame
                    return foreach(emit, parse_stream_events(l, r, frame))
                end
            finally
                filter!(callback -> callback !== stop_read, cancel)
            end
        end
    end
    return coalesce_stream(raw; model=r.model)
end
function parse_stream_events(l, r, frame::SSEEvent)
    isempty(frame.data) && return StreamEvent[]
    frame.data=="[DONE]" && return StreamEvent[StreamEndEvent()]
    d=JSON.parse(frame.data)
    d isa AbstractDict || return StreamEvent[]
    l.dialect=="openai-responses" && return openai_stream_frame(l, r, d)
    l.dialect=="openai-chat" && return chat_stream_frame(l, r, d)
    l.dialect=="anthropic" && return anthropic_stream_frame(l, r, d)
    return gemini_stream_frame(l, r, d)
end
deltaevent(d::Delta) = StreamDeltaEvent(; delta=d)
function frame_error(l, d)
    inner=getobject(d, "error")
    code=wire_string(
        first_nonempty(
            get(inner, "code", nothing),
            get(inner, "type", nothing),
            get(inner, "status", nothing),
            get(d, "code", nothing),
            get(d, "error_type", nothing),
            "provider",
        ),
    )
    message=wire_string(first_nonempty(get(inner, "message", nothing), get(d, "message", nothing)))
    return StreamErrorEvent(; error=error_detail(l, code, message))
end
function openai_stream_frame(l, r, d)
    type=get(d, "type", nothing)
    i=Int(get(d, "output_index", 0))
    out=StreamEvent[]
    if type=="response.created"
        response=getobject(d, "response")
        push!(
            out,
            StreamStartEvent(;
                id=string_field(response, "id"),
                model=string(first_nonempty(get(response, "model", nothing), r.model)),
            ),
        )
    elseif type in ("response.output_item.added", "response.output_item.done")
        item=getobject(d, "item")
        tag=get(item, "type", nothing)
        if tag=="reasoning"
            if type=="response.output_item.added"
                push!(out, deltaevent(ThinkingDelta(; text="", part_index=i)))
            else
                state=obj(
                    (
                        key=>item[key] for key in ("id", "encrypted_content") if
                        !empty_optional(get(item, key, nothing))
                    )...,
                )
                isempty(state) || push!(
                    out,
                    deltaevent(
                        ContinuationDelta(;
                            provider="openai", kind="reasoning_item", data=state, part_index=i
                        ),
                    ),
                )
            end
        elseif tag=="function_call" && type=="response.output_item.added"
            push!(
                out,
                deltaevent(
                    ToolCallDelta(;
                        input=wire_string(get(item, "arguments", nothing)),
                        part_index=i,
                        id=first_nonempty(string_field(item, "call_id"), string_field(item, "id")),
                        name=string_field(item, "name"),
                    ),
                ),
            )
        end
    elseif type in ("response.output_text.delta", "response.refusal.delta")
        push!(
            out,
            deltaevent(
                TextDelta(;
                    text=wire_string(get(d, "delta", nothing)),
                    part_index=i,
                    logprobs=openai_logprobs(get(d, "logprobs", nothing)),
                ),
            ),
        )
    elseif type in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta")
        push!(
            out,
            deltaevent(ThinkingDelta(; text=wire_string(get(d, "delta", nothing)), part_index=i)),
        )
    elseif type=="response.function_call_arguments.delta"
        push!(
            out,
            deltaevent(
                ToolCallDelta(;
                    input=wire_string(get(d, "delta", nothing)),
                    part_index=i,
                    id=first_nonempty(string_field(d, "call_id"), string_field(d, "id")),
                    name=string_field(d, "name"),
                ),
            ),
        )
    elseif type=="response.output_text.annotation.added"
        part=citation_from(getobject(d, "annotation"))
        part===nothing || push!(
            out,
            deltaevent(
                CitationDelta(; text=part.text, url=part.url, title=part.title, part_index=i)
            ),
        )
    elseif type=="response.output_audio.delta"
        push!(
            out,
            deltaevent(
                AudioDelta(;
                    data=wire_string(get(d, "delta", nothing)), part_index=i, media_type="audio/wav"
                ),
            ),
        )
    elseif type in ("response.output_image.delta", "response.image.delta")
        push!(
            out,
            deltaevent(
                ImageDelta(;
                    data=wire_string(get(d, "delta", nothing)), part_index=i, media_type="image/png"
                ),
            ),
        )
    elseif type in ("response.completed", "response.incomplete")
        response=getobject(d, "response")
        has_tool=any(
            x->x isa AbstractDict && get(x, "type", nothing)=="function_call",
            getarray(response, "output"),
        )
        push!(
            out,
            StreamEndEvent(;
                finish_reason=response_finish(response; has_tool),
                usage=usage_from(l.dialect, get(response, "usage", nothing)),
                provider_data=response,
            ),
        )
    elseif type in ("error", "response.error", "response.failed")
        push!(out, frame_error(l, type=="response.failed" ? getobject(d, "response") : d))
    end
    return out
end
function chat_stream_frame(l, r, d)
    get(d, "error", nothing) isa AbstractDict && return StreamEvent[frame_error(l, d)]
    out=StreamEvent[]
    choices=getarray(d, "choices")
    choice=isempty(choices) ? obj() : asobject(first(choices))
    delta=getobject(choice, "delta")
    length(choices)>1 && unsupported(l.provider, "multiple choices in one stream")
    reason=first_nonempty(
        get(delta, "reasoning_content", nothing), get(delta, "reasoning", nothing)
    )
    reason===nothing || push!(out, deltaevent(ThinkingDelta(; text=string(reason))))
    content=string_field(delta, "content")
    content===nothing || push!(
        out,
        deltaevent(
            TextDelta(;
                text=content,
                logprobs=openai_logprobs(get(getobject(choice, "logprobs"), "content", nothing)),
            ),
        ),
    )
    for call in getarray(delta, "tool_calls")
        call isa AbstractDict || continue
        f=getobject(call, "function")
        push!(
            out,
            deltaevent(
                ToolCallDelta(;
                    input=wire_string(get(f, "arguments", nothing)),
                    part_index=Int(get(call, "index", 0)),
                    id=string_field(call, "id"),
                    name=string_field(f, "name"),
                ),
            ),
        )
    end
    finish=string_field(choice, "finish_reason")
    usage=get(d, "usage", nothing)
    if finish!==nothing || usage isa AbstractDict
        push!(
            out,
            StreamEndEvent(;
                finish_reason=finish===nothing ? nothing : get(CHAT_FINISH, finish, "stop"),
                usage=usage isa AbstractDict ? usage_from(l.dialect, usage) : nothing,
                provider_data=d,
            ),
        )
    end
    return out
end
function anthropic_stream_frame(l, r, d)
    type=get(d, "type", nothing)
    i=Int(get(d, "index", 0))
    out=StreamEvent[]
    if type=="message_start"
        m=getobject(d, "message")
        push!(
            out,
            StreamStartEvent(;
                id=string_field(m, "id"),
                model=string(first_nonempty(get(m, "model", nothing), r.model)),
            ),
        )
    elseif type=="content_block_start"
        b=getobject(d, "content_block")
        tag=get(b, "type", nothing)
        if tag=="tool_use"
            input=get(b, "input", nothing)
            fragment=if input isa AbstractDict
                (isempty(input) ? "" : JSON.serialize(input))
            else
                wire_string(input)
            end
            push!(
                out,
                deltaevent(
                    ToolCallDelta(;
                        input=fragment,
                        part_index=i,
                        id=string_field(b, "id"),
                        name=string_field(b, "name"),
                    ),
                ),
            )
        elseif tag=="redacted_thinking" && get(b, "data", nothing)!==nothing
            push!(out, deltaevent(ThinkingDelta(; text="", part_index=i)))
            push!(
                out,
                deltaevent(
                    ContinuationDelta(;
                        provider="anthropic",
                        kind="redacted_thinking",
                        data=obj("data"=>b["data"]),
                        part_index=i,
                    ),
                ),
            )
        end
    elseif type=="content_block_delta"
        delta=getobject(d, "delta")
        tag=get(delta, "type", nothing)
        if tag=="text_delta"
            push!(
                out,
                deltaevent(
                    TextDelta(; text=wire_string(get(delta, "text", nothing)), part_index=i)
                ),
            )
        elseif tag=="thinking_delta"
            push!(
                out,
                deltaevent(
                    ThinkingDelta(; text=wire_string(get(delta, "thinking", nothing)), part_index=i)
                ),
            )
        elseif tag=="input_json_delta"
            push!(
                out,
                deltaevent(
                    ToolCallDelta(;
                        input=wire_string(get(delta, "partial_json", nothing)), part_index=i
                    ),
                ),
            )
        elseif tag=="signature_delta" && string_field(delta, "signature")!==nothing
            push!(
                out,
                deltaevent(
                    ContinuationDelta(;
                        provider="anthropic",
                        kind="thinking_signature",
                        data=obj("signature"=>delta["signature"]),
                        part_index=i,
                    ),
                ),
            )
        elseif tag in ("citation_delta", "citations_delta")
            part=citation_from(get(delta, "citation", delta); anthropic=true)
            part===nothing || push!(
                out,
                deltaevent(
                    CitationDelta(; text=part.text, url=part.url, title=part.title, part_index=i),
                ),
            )
        end
    elseif type=="message_delta"
        delta=getobject(d, "delta")
        raw=getobject(d, "usage")
        finish=get(delta, "stop_reason", nothing)
        if finish!==nothing || !isempty(raw)
            push!(
                out,
                StreamEndEvent(;
                    finish_reason=finish===nothing ? nothing : finish_reason(l.dialect, finish),
                    usage=isempty(raw) ? nothing : usage_from(l.dialect, raw),
                    provider_data=d,
                ),
            )
        end
    elseif type=="message_stop"
        push!(out, StreamEndEvent())
    elseif type=="error"
        push!(out, frame_error(l, d))
    end
    return out
end
function gemini_stream_frame(l, r, d)
    haskey(d, "error") && return StreamEvent[frame_error(l, d)]
    candidate=try
        gemini_inband(l, d)
    catch e
        e isa LM15Error || rethrow()
        return StreamEvent[StreamErrorEvent(;
            error=ErrorDetail(;
                code=e.code, message=e.message, provider_code="inband_finish_reason"
            ),
        )]
    end
    out=StreamEvent[]
    has_tool=false
    logprobs=gemini_logprobs(get(candidate, "logprobsResult", nothing))
    for (pos, p) in enumerate(getarray(getobject(candidate, "content"), "parts"))
        p isa AbstractDict || continue
        i=pos-1
        if haskey(p, "text")
            if get(p, "thought", false)===true
                push!(out, deltaevent(ThinkingDelta(; text=wire_string(p["text"]), part_index=i)))
            else
                push!(
                    out,
                    deltaevent(TextDelta(; text=wire_string(p["text"]), part_index=i, logprobs)),
                )
                logprobs=()
            end
            for state in thought_state(p)
                push!(
                    out,
                    deltaevent(
                        ContinuationDelta(;
                            provider=state.provider, kind=state.kind, data=state.data, part_index=i
                        ),
                    ),
                )
            end
        elseif get(p, "functionCall", nothing) isa AbstractDict
            fc=p["functionCall"]
            has_tool=true
            push!(
                out,
                deltaevent(
                    ToolCallDelta(;
                        input=JSON.serialize(get(fc, "args", obj())),
                        part_index=i,
                        id=string_field(fc, "id"),
                        name=string_field(fc, "name"),
                    ),
                ),
            )
            signature=first_nonempty(
                get(p, "thoughtSignature", nothing), get(fc, "thoughtSignature", nothing)
            )
            signature===nothing || push!(
                out,
                deltaevent(
                    ContinuationDelta(;
                        provider="gemini",
                        kind="thought_signature",
                        data=obj("value"=>string(signature)),
                        part_index=i,
                    ),
                ),
            )
        elseif get(p, "inlineData", nothing) isa AbstractDict
            media=p["inlineData"]
            mime=wire_string(get(media, "mimeType", nothing))
            value=wire_string(get(media, "data", nothing))
            if startswith(mime, "audio/")
                push!(out, deltaevent(AudioDelta(; data=value, media_type=mime, part_index=i)))
            elseif startswith(mime, "image/")
                push!(out, deltaevent(ImageDelta(; data=value, media_type=mime, part_index=i)))
            else
                unsupported(l.provider, "non-streamable generated media")
            end
        end
    end
    finish=string_field(candidate, "finishReason")
    if finish!==nothing || (isempty(out) && haskey(d, "usageMetadata"))
        push!(
            out,
            StreamEndEvent(;
                finish_reason=finish_reason(l.dialect, finish; has_tool),
                usage=usage_from(l.dialect, get(d, "usageMetadata", nothing)),
                provider_data=d,
            ),
        )
    end
    return out
end

Base.@kwdef mutable struct StreamSlot
    text::Union{Nothing,IOBuffer} = nothing
    thinking::Union{Nothing,IOBuffer} = nothing
    image::Maybe{ImagePart} = nothing
    audio_chunks::Vector{String} = String[]
    audio_source::Maybe{Pair{Symbol,String}} = nothing
    audio_media_type::Maybe{String} = nothing
    citations::Vector{CitationPart} = CitationPart[]
    tool_input::Union{Nothing,IOBuffer} = nothing
    tool_id::Maybe{String} = nothing
    tool_name::Maybe{String} = nothing
    continuation::Vector{ContinuationState} = ContinuationState[]
end
Base.@kwdef mutable struct StreamAccumulator
    request::Request
    id::Maybe{String} = nothing
    model::Maybe{String} = nothing
    finish_reason::Maybe{String} = nothing
    usage::Usage = Usage()
    provider_data::Maybe{JsonObject} = nothing
    slots::Dict{Int,StreamSlot} = Dict{Int,StreamSlot}()
    continuation::Vector{ContinuationState} = ContinuationState[]
    logprobs::Vector{TokenLogprob} = TokenLogprob[]
end
StreamAccumulator(r::Request) = StreamAccumulator(; request=r)
function Base.push!(acc::StreamAccumulator, event::StreamEvent)
    validate(event)
    if event isa StreamStartEvent
        event.id===nothing || (acc.id=event.id)
        event.model===nothing || (acc.model=event.model)
    elseif event isa StreamEndEvent
        event.finish_reason===nothing || (acc.finish_reason=event.finish_reason)
        event.usage===nothing || (acc.usage=event.usage)
        event.provider_data===nothing || (acc.provider_data=event.provider_data)
    elseif event isa StreamDeltaEvent
        d=event.delta
        if d isa ContinuationDelta && d.part_index===nothing
            push!(
                acc.continuation, ContinuationState(; provider=d.provider, kind=d.kind, data=d.data)
            )
            return acc
        end
        slot=get!(acc.slots, d.part_index, StreamSlot())
        if d isa TextDelta
            slot.text===nothing && (slot.text=IOBuffer())
            write(slot.text, d.text)
            append!(acc.logprobs, d.logprobs)
        elseif d isa ThinkingDelta
            slot.thinking===nothing && (slot.thinking=IOBuffer())
            write(slot.thinking, d.text)
        elseif d isa ToolCallDelta
            slot.tool_input===nothing && (slot.tool_input=IOBuffer())
            write(slot.tool_input, d.input)
            d.id===nothing || (slot.tool_id=d.id)
            d.name===nothing || (slot.tool_name=d.name)
        elseif d isa ImageDelta
            kwargs=Dict{Symbol,Any}(:media_type=>something(d.media_type, "image/png"))
            for key in (:data, :url, :file_id)
                v=getfield(d, key)
                v===nothing || (kwargs[key]=v)
            end
            length(kwargs)>1 && (slot.image=ImagePart(; kwargs...))
        elseif d isa AudioDelta
            if d.data!==nothing
                slot.audio_source===nothing || throw(
                    StreamAssemblyError(
                        "audio source changed inside a stream"; part_index=d.part_index
                    ),
                )
                push!(slot.audio_chunks, d.data)
            elseif d.url!==nothing || d.file_id!==nothing
                isempty(slot.audio_chunks) || throw(
                    StreamAssemblyError(
                        "audio source changed inside a stream"; part_index=d.part_index
                    ),
                )
                slot.audio_source=d.url===nothing ? :file_id=>d.file_id : :url=>d.url
            end
            d.media_type===nothing || (slot.audio_media_type=d.media_type)
        elseif d isa CitationDelta
            push!(slot.citations, CitationPart(; text=d.text, url=d.url, title=d.title))
        elseif d isa ContinuationDelta
            push!(
                slot.continuation,
                ContinuationState(; provider=d.provider, kind=d.kind, data=d.data),
            )
        end
    end
    return acc
end
function buffer_text(io::IOBuffer)
    position=Base.position(io)
    seekstart(io)
    try
        read(io, String)
    finally
        seek(io, position)
    end
end
function pcm_wav(data; rate=24000, channels=1)
    io=IOBuffer()
    write(
        io,
        "RIFF",
        htol(UInt32(36+length(data))),
        "WAVEfmt ",
        htol(UInt32(16)),
        htol(UInt16(1)),
        htol(UInt16(channels)),
        htol(UInt32(rate)),
        htol(UInt32(rate*channels*2)),
        htol(UInt16(channels*2)),
        htol(UInt16(16)),
        "data",
        htol(UInt32(length(data))),
        data,
    )
    return take!(io)
end
function decode_audio_chunks(chunks)
    encoded=join(base64_payload(chunk) for chunk in chunks)
    isempty(encoded) && return UInt8[]
    occursin(r"^[A-Za-z0-9+/=]+$", encoded) ||
        throw(StreamAssemblyError("audio stream contains invalid base64"))
    raw=UInt8[]
    consumed=0
    # Padding delimits independently encoded packets. Unpadded fragments remain
    # together until a complete quartet exists, rather than guessing missing bits.
    for segment in eachmatch(r"[A-Za-z0-9+/]+={0,2}", encoded)
        text=segment.match
        ncodeunits(text)%4==0 ||
            throw(StreamAssemblyError("audio stream ended with an incomplete base64 packet"))
        append!(raw, base64decode(text))
        consumed+=ncodeunits(text)
    end
    consumed==ncodeunits(encoded) ||
        throw(StreamAssemblyError("audio stream has invalid base64 padding"))
    return raw
end
function assemble(acc; skip=Set{Int}())
    parts=Part[]
    for i in sort!(collect(keys(acc.slots)))
        slot=acc.slots[i]
        states=Tuple(slot.continuation)
        before=length(parts)
        slot.thinking===nothing ||
            push!(parts, ThinkingPart(buffer_text(slot.thinking); continuation=states))
        slot.text===nothing || push!(parts, TextPart(buffer_text(slot.text); continuation=states))
        slot.image===nothing || push!(parts, reconstruct(slot.image; continuation=states))
        if !isempty(slot.audio_chunks)
            raw=decode_audio_chunks(slot.audio_chunks)
            mime=slot.audio_media_type
            if mime in (nothing, "audio/pcm", "audio/pcm16")
                raw=pcm_wav(raw)
                mime="audio/wav"
            end
            push!(parts, AudioPart(; media_type=mime, data=base64encode(raw), continuation=states))
        elseif slot.audio_source!==nothing
            push!(
                parts,
                AudioPart(;
                    media_type=something(slot.audio_media_type, "audio/wav"),
                    continuation=states,
                    (slot.audio_source,)...,
                ),
            )
        end
        append!(parts, [reconstruct(c; continuation=states) for c in slot.citations])
        if slot.tool_input!==nothing && !(i in skip)
            push!(
                parts,
                ToolCallPart(;
                    id=something(slot.tool_id, "tool_call_$i"),
                    name=slot.tool_name,
                    input=json_object(buffer_text(slot.tool_input)),
                    continuation=states,
                ),
            )
        end
        length(parts)==before && !(i in skip) && push!(parts, TextPart(""; continuation=states))
    end
    isempty(parts) && push!(parts, TextPart(""))
    has_tool=any(p->p isa ToolCallPart, parts)
    finish=something(acc.finish_reason, has_tool ? "tool_call" : "stop")
    finish=="stop" && has_tool && (finish="tool_call")
    return Response(;
        id=acc.id,
        model=something(acc.model, acc.request.model),
        message=Message("assistant", Tuple(parts); continuation=Tuple(acc.continuation)),
        finish_reason=finish,
        usage=acc.usage,
        logprobs=isempty(acc.logprobs) ? nothing : Tuple(acc.logprobs),
        provider_data=acc.provider_data,
    )
end
function response(acc::StreamAccumulator)
    unnamed=sort!([i for (i, s) in acc.slots if s.tool_input!==nothing && s.tool_name===nothing])
    isempty(unnamed) || throw(
        StreamAssemblyError(
            "tool call arrived without a name";
            partial=assemble(acc; skip=Set(unnamed)),
            part_index=first(unnamed),
        ),
    )
    return assemble(acc)
end
function partial_response(acc)
    try
        response(acc)
    catch e
        e isa StreamAssemblyError ? e.partial : nothing
    end
end
"""
    ResponseStream(events, request)
    ResponseStream(f, events, request)

Lazily assemble canonical events. Iterate `events(rs)` or `text_chunks(rs)`, then
call `response(rs)` for the completed answer. The `do` form always closes its
source; leaving before the end event does not fabricate a completed answer.
"""
mutable struct ResponseStream{S}
    source::S
    accumulator::StreamAccumulator
    result::Maybe{Response}
    failure::Maybe{Exception}
    cleanup_errors::Vector{Exception}
    state::Any
    started::Bool
    done::Bool
    source_closed::Bool
end
function ResponseStream(source, r::Request)
    return ResponseStream(
        source, StreamAccumulator(r), nothing, nothing, Exception[], nothing, false, false, false
    )
end
Base.IteratorSize(::Type{<:ResponseStream}) = Base.SizeUnknown()
Base.eltype(::Type{<:ResponseStream}) = StreamEvent
Base.isopen(rs::ResponseStream) = !rs.done && rs.failure === nothing
events(rs::ResponseStream) = rs

function record_cleanup!(rs::ResponseStream, error)
    push!(rs.cleanup_errors, error)
    rs.failure isa LM15Error && push!(rs.failure.cleanup_errors, error)
    @warn "Stream cleanup failed; the completed answer or original failure is preserved" exception_type=string(
        nameof(typeof(error))
    )
    return nothing
end

function close_source!(rs::ResponseStream)
    rs.source_closed && return nothing
    rs.source_closed = true
    try
        applicable(close, rs.source) && close(rs.source)
    catch error
        record_cleanup!(rs, error)
    end
    return nothing
end

function fail_stream!(rs::ResponseStream, error)
    rs.failure = error
    rs.done = true
    close_source!(rs)
    return throw(error)
end

function Base.iterate(rs::ResponseStream, state=nothing)
    rs.failure === nothing || throw(rs.failure)
    rs.done && return nothing
    item = try
        rs.started ? iterate(rs.source, rs.state) : iterate(rs.source)
    catch error
        if rs.result !== nothing && !(error isa StreamAssemblyError)
            record_cleanup!(rs, error)
            rs.done = true
            close_source!(rs)
            return nothing
        end
        fail_stream!(rs, error)
    end
    rs.started = true
    if item === nothing
        if rs.result === nothing
            fail_stream!(
                rs,
                StreamAssemblyError(
                    "stream ended without an end event"; partial=partial_response(rs.accumulator)
                ),
            )
        end
        rs.done = true
        close_source!(rs)
        return nothing
    end
    event, rs.state = item
    rs.result === nothing || fail_stream!(
        rs, StreamAssemblyError("event arrived after the end event"; partial=rs.result)
    )
    if event isa StreamErrorEvent
        fail_stream!(
            rs,
            error_for_code(
                event.error.code, event.error.message; provider_code=event.error.provider_code
            ),
        )
    end
    try
        push!(rs.accumulator, event)
        event isa StreamEndEvent && (rs.result = response(rs.accumulator))
    catch error
        fail_stream!(rs, error)
    end
    return event, nothing
end

function response(rs::ResponseStream)
    rs.failure === nothing || throw(rs.failure)
    for _ in rs
    end
    return rs.result
end

function Base.close(rs::ResponseStream)
    if !rs.done && rs.result === nothing && rs.failure === nothing
        rs.failure = StreamAssemblyError(
            "stream closed before its end event"; partial=partial_response(rs.accumulator)
        )
    end
    rs.done = true
    return close_source!(rs)
end

function ResponseStream(f, source, request::Request)
    rs = ResponseStream(source, request)
    try
        f(rs)
    finally
        close(rs)
    end
end

materialize_response(source, request::Request) = ResponseStream(response, source, request)
function text_chunks(rs::ResponseStream)
    return (
        event.delta.text for event in rs if event isa StreamDeltaEvent && event.delta isa TextDelta
    )
end

"""Expose a complete response as canonical events without inventing missing deltas."""
function response_to_events(answer::Response)
    validate(answer)
    for p in answer.message.parts
        p isa Union{TextPart,ThinkingPart,ImagePart,AudioPart,CitationPart,ToolCallPart} ||
            throw(ArgumentError("$(kind(p)) has no stream delta representation"))
        p isa ImagePart &&
            p.path!==nothing &&
            throw(
                ArgumentError(
                    "path-addressed images must be loaded explicitly before converting to events"
                ),
            )
        p isa AudioPart &&
            p.data===nothing &&
            throw(ArgumentError("response-to-events audio must carry inline data"))
    end
    EventStream() do emit, _
        emit(StreamStartEvent(; id=answer.id, model=answer.model))
        logprobs=answer.logprobs===nothing ? () : answer.logprobs
        for (position, p) in enumerate(answer.message.parts)
            index=position-1
            delta=if p isa TextPart
                fragment=TextDelta(; text=p.text, part_index=index, logprobs)
                logprobs=()
                fragment
            elseif p isa ThinkingPart
                ThinkingDelta(; text=p.text, part_index=index)
            elseif p isa ToolCallPart
                ToolCallDelta(;
                    input=JSON.serialize(p.input), id=p.id, name=p.name, part_index=index
                )
            elseif p isa ImagePart
                ImageDelta(;
                    data=p.data,
                    url=p.url,
                    file_id=p.file_id,
                    media_type=p.media_type,
                    part_index=index,
                )
            elseif p isa AudioPart
                AudioDelta(; data=p.data, media_type=p.media_type, part_index=index)
            else
                CitationDelta(; text=p.text, url=p.url, title=p.title, part_index=index)
            end
            emit(deltaevent(delta))
            for state in p.continuation
                emit(
                    deltaevent(
                        ContinuationDelta(;
                            provider=state.provider,
                            kind=state.kind,
                            data=state.data,
                            part_index=index,
                        ),
                    ),
                )
            end
        end
        for state in answer.message.continuation
            emit(
                deltaevent(
                    ContinuationDelta(; provider=state.provider, kind=state.kind, data=state.data)
                ),
            )
        end
        return emit(
            StreamEndEvent(;
                finish_reason=answer.finish_reason,
                usage=answer.usage,
                provider_data=answer.provider_data,
            ),
        )
    end
end
