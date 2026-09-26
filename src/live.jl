function live_setup_frames(l::ProviderLM, c::LiveConfig)
    require_surface(l, :live)
    validate(c)
    all(t->t isa FunctionTool, c.tools) || unsupported(l.provider, "builtin tools in live sessions")
    if l.dialect=="gemini"
        c.input_format===nothing ||
            (c.input_format.encoding=="pcm16" && c.input_format.channels==1) ||
            unsupported(l.provider, "live input audio format")
        c.output_format===nothing ||
            (
                c.output_format.encoding=="pcm16" &&
                c.output_format.channels==1 &&
                c.output_format.sample_rate==24000
            ) ||
            unsupported(l.provider, "live output audio format")
        model=startswith(c.model, "models/") ? c.model : "models/"*c.model
        setup=obj("model"=>model)
        c.system===nothing || (
            setup["systemInstruction"]=obj(
                "parts"=>[obj("text"=>system_text(c.system; provider=l.provider))]
            )
        )
        isempty(c.tools) || (setup["tools"]=gemini_tools(c.tools))
        gen=obj()
        (c.output_format!==nothing || audio_native(c.model)) &&
            (gen["responseModalities"]=["AUDIO"])
        c.voice===nothing || (
            gen["speechConfig"]=obj(
                "voiceConfig"=>obj("prebuiltVoiceConfig"=>obj("voiceName"=>c.voice))
            )
        )
        isempty(gen) || (setup["generationConfig"]=gen)
        merge!(setup, something(c.extensions, obj()))
        audio_native(c.model) && (setup["outputAudioTranscription"]=obj())
        return [obj("setup"=>setup)]
    end
    session=obj("type"=>"realtime")
    c.system===nothing || (session["instructions"]=system_text(c.system; provider=l.provider))
    audio=obj()
    session["output_modalities"]=if c.output_format!==nothing || c.voice!==nothing
        ["audio"]
    else
        ["text"]
    end
    if c.output_format!==nothing || c.voice!==nothing
        output=obj()
        c.output_format===nothing || (output["format"]=live_audio_format(l, c.output_format))
        c.voice===nothing || (output["voice"]=c.voice)
        audio["output"]=output
    end
    c.input_format===nothing || (
        audio["input"]=obj(
            "format"=>live_audio_format(l, c.input_format), "turn_detection"=>nothing
        )
    )
    isempty(audio) || (session["audio"]=audio)
    isempty(c.tools) || (
        session["tools"]=[
            obj(
                "type"=>"function",
                "name"=>t.name,
                "description"=>t.description,
                "parameters"=>t.parameters,
            ) for t in c.tools
        ]
    )
    merge!(session, something(c.extensions, obj()))
    return [obj("type"=>"session.update", "session"=>session)]
end
function audio_native(model)
    return occursin("live-preview", lowercase(model)) || occursin("native-audio", lowercase(model))
end
function live_audio_format(l, f::AudioFormat)
    f.channels==1 || unsupported(l.provider, "multichannel live audio")
    return if f.encoding=="pcm16"
        obj("type"=>"audio/pcm", "rate"=>f.sample_rate)
    else
        obj("type"=>"audio/$(f.encoding)")
    end
end
function encode_live_event(l::ProviderLM, c::LiveConfig, e::LiveClientEvent)
    validate(e)
    if l.dialect=="gemini"
        e isa LiveClientAudioEvent && return [
            obj("realtimeInput"=>obj("audio"=>obj("mimeType"=>e.media_type, "data"=>e.data)))
        ]
        e isa LiveClientImageEvent && return [
            obj("realtimeInput"=>obj("video"=>obj("mimeType"=>e.media_type, "data"=>e.data)))
        ]
        e isa LiveClientEndAudioEvent && return [obj("realtimeInput"=>obj("audioStreamEnd"=>true))]
        e isa LiveClientInterruptEvent && return [obj("clientContent"=>obj("turnComplete"=>true))]
        if e isa LiveClientTextEvent && audio_native(c.model)
            return [obj("realtimeInput"=>obj("text"=>e.text))]
        elseif e isa LiveClientToolResultEvent
            output=[
                obj("text"=>parts_to_text(e.content; provider=l.provider, where="live tool output"))
            ]
            return [
                obj(
                    "toolResponse"=>obj(
                        "functionResponses"=>[obj("id"=>e.id, "response"=>obj("output"=>output))]
                    ),
                ),
            ]
        end
        parts=if e isa LiveClientTurnEvent
            [gemini_part(l, p, Dict{String,String}()) for p in e.parts]
        else
            [obj("text"=>e.text)]
        end
        return [
            obj(
                "clientContent"=>obj(
                    "turns"=>[obj("role"=>"user", "parts"=>parts)],
                    "turnComplete"=>e isa LiveClientTurnEvent ? e.turn_complete : true,
                ),
            ),
        ]
    end
    e isa LiveClientAudioEvent && return [obj("type"=>"input_audio_buffer.append", "audio"=>e.data)]
    e isa LiveClientEndAudioEvent &&
        return [obj("type"=>"input_audio_buffer.commit"), obj("type"=>"response.create")]
    e isa LiveClientInterruptEvent && return [obj("type"=>"response.cancel")]
    item=if e isa LiveClientToolResultEvent
        obj(
            "type"=>"function_call_output",
            "call_id"=>e.id,
            "output"=>parts_to_text(e.content; provider=l.provider, where="live tool output"),
        )
    else
        content=if e isa LiveClientTextEvent
            [obj("type"=>"input_text", "text"=>e.text)]
        elseif e isa LiveClientImageEvent
            [obj("type"=>"input_image", "image_url"=>"data:$(e.media_type);base64,$(e.data)")]
        else
            [openai_input(p, l.provider) for p in e.parts]
        end
        obj("type"=>"message", "role"=>"user", "content"=>content)
    end
    frames=[obj("type"=>"conversation.item.create", "item"=>item)]
    e isa LiveClientTurnEvent && !e.turn_complete || push!(frames, obj("type"=>"response.create"))
    return frames
end
function live_usage(l, d)
    if l.dialect=="gemini"
        raw=get(d, "usageMetadata", get(getobject(d, "serverContent"), "usageMetadata", nothing))
        return raw isa AbstractDict ? usage_from("gemini", raw; live=true) : nothing
    end
    raw=get(d, "usage", nothing)
    raw isa AbstractDict || return nothing
    normalized=copy(raw)
    for side in ("input", "output")
        singular=side*"_token_details"
        plural=side*"_tokens_details"
        haskey(raw, singular) && !haskey(raw, plural) && (normalized[plural]=raw[singular])
    end
    return usage_from("openai-responses", normalized)
end
function live_call(l, d; default_id=nothing)
    id=first_nonempty(string_field(d, "call_id"), string_field(d, "id"), default_id)
    name=string_field(d, "name")
    id===nothing && throw(GenericProviderError("live tool call carries no id"; provider=l.provider))
    name===nothing && unnamed_call(l.provider, "live tool call")
    input=l.dialect=="gemini" ? getobject(d, "args") : json_object(get(d, "arguments", nothing))
    return LiveServerToolCallEvent(; id, name, input)
end
function decode_live_frame(l::ProviderLM, raw)
    d=if raw isa AbstractDict
        raw
    else
        JSON.parse(raw isa AbstractVector{UInt8} ? String(copy(raw)) : raw)
    end
    d isa AbstractDict ||
        throw(GenericProviderError("live frame is not an object"; provider=l.provider))
    out=LiveServerEvent[]
    if l.dialect=="gemini"
        haskey(d, "error") &&
            return LiveServerEvent[LiveServerErrorEvent(; error=frame_error(l, d).error)]
        for fc in getarray(getobject(d, "toolCall"), "functionCalls")
            fc isa AbstractDict || continue
            push!(out, live_call(l, fc; default_id="fc_0"))
        end
        server=getobject(d, "serverContent")
        for p in getarray(getobject(server, "modelTurn"), "parts")
            p isa AbstractDict || continue
            if haskey(p, "text")
                push!(out, LiveServerTextEvent(; text=wire_string(p["text"])))
            elseif get(p, "inlineData", nothing) isa AbstractDict
                media=p["inlineData"]
                mime=wire_string(get(media, "mimeType", nothing))
                startswith(mime, "audio/") ||
                    unsupported(l.provider, "non-audio media in a live server frame")
                push!(
                    out,
                    LiveServerAudioEvent(;
                        data=wire_string(get(media, "data", nothing)), media_type=mime
                    ),
                )
            elseif get(p, "functionCall", nothing) isa AbstractDict
                push!(out, live_call(l, p["functionCall"]; default_id="fc_0"))
            end
        end
        transcript=string_field(getobject(server, "outputTranscription"), "text")
        transcript===nothing || push!(out, LiveServerTextEvent(; text=transcript))
        usage=live_usage(l, d)
        turn_complete=get(server, "turnComplete", false)===true
        usage===nothing || turn_complete || push!(out, LiveServerUsageEvent(; usage))
        get(server, "interrupted", false)===true && push!(out, LiveServerInterruptedEvent())
        turn_complete &&
            push!(out, LiveServerTurnEndEvent(; usage=usage===nothing ? Usage() : usage))
        return out
    end
    type=get(d, "type", nothing)
    if type in (
        "response.output_text.delta",
        "response.text.delta",
        "response.output_audio_transcript.delta",
        "response.audio_transcript.delta",
    )
        delta=first_nonempty(string_field(d, "delta"), string_field(d, "text"))
        delta===nothing || push!(out, LiveServerTextEvent(; text=delta))
    elseif type=="response.output_audio.delta"
        delta=string_field(d, "delta")
        delta===nothing || push!(out, LiveServerAudioEvent(; data=delta))
    elseif type=="response.function_call_arguments.delta"
        delta=string_field(d, "delta")
        delta===nothing || push!(
            out,
            LiveServerToolCallDeltaEvent(;
                input_delta=delta,
                id=first_nonempty(string_field(d, "call_id"), string_field(d, "id")),
                name=string_field(d, "name"),
            ),
        )
    elseif type=="response.output_item.done"
        item=getobject(d, "item")
        get(item, "type", nothing)=="function_call" && push!(out, live_call(l, item))
    elseif type in ("response.done", "response.completed")
        response=getobject(d, "response")
        usage=live_usage(l, response)
        if get(response, "status", nothing)=="cancelled"
            usage===nothing || push!(out, LiveServerUsageEvent(; usage))
            push!(out, LiveServerInterruptedEvent())
        elseif any(
            i->i isa AbstractDict && get(i, "type", nothing)=="function_call",
            getarray(response, "output"),
        )
            usage===nothing || push!(out, LiveServerUsageEvent(; usage))
        else
            push!(out, LiveServerTurnEndEvent(; usage=usage===nothing ? Usage() : usage))
        end
    elseif type in ("response.cancelled", "response.canceled")
        push!(out, LiveServerInterruptedEvent())
    elseif type in ("error", "response.error")
        error=frame_error(l, d).error
        error.provider_code=="response_cancel_not_active" ||
            push!(out, LiveServerErrorEvent(; error))
    end
    return out
end

mutable struct LiveSession{W,L}
    socket::W
    lm::L
    config::LiveConfig
    pending::Vector{LiveServerEvent}
    send_lock::ReentrantLock
    receive_lock::ReentrantLock
    closed::Bool
end
function LiveSession(ws, l, c)
    return LiveSession(ws, l, c, LiveServerEvent[], ReentrantLock(), ReentrantLock(), false)
end
function Base.show(io::IO, s::LiveSession)
    return print(io, "LiveSession(", repr(s.lm.provider), ", ", s.closed ? "closed" : "open", ")")
end
function send!(s::LiveSession, e::LiveClientEvent)
    frames=encode_live_event(s.lm, s.config, e)
    lock(s.send_lock) do
        s.closed && throw(ArgumentError("live session is closed"))
        for frame in frames
            try
                HTTP.WebSockets.send(s.socket, wire_json(frame))
            catch error
                error isa InterruptException && rethrow()
                throw(TransportError("live send failed"; provider=s.lm.provider))
            end
        end
    end
    return nothing
end
send_text!(s::LiveSession, value) = send!(s, LiveClientTextEvent(; text=value))
function send_turn!(s::LiveSession, value; turn_complete=true)
    return send!(s, LiveClientTurnEvent(; parts=normalize_content(value), turn_complete))
end
function send_audio!(s::LiveSession, data; media_type="audio/pcm;rate=16000")
    return send!(
        s,
        LiveClientAudioEvent(;
            data=data isa AbstractVector{UInt8} ? base64encode(data) : data, media_type
        ),
    )
end
function send_image!(s::LiveSession, data; media_type="image/jpeg")
    return send!(
        s,
        LiveClientImageEvent(;
            data=data isa AbstractVector{UInt8} ? base64encode(data) : data, media_type
        ),
    )
end
function send_tool_result!(s::LiveSession, id, content)
    return send!(s, LiveClientToolResultEvent(; id, content=normalize_content(content)))
end
interrupt!(s::LiveSession) = send!(s, LiveClientInterruptEvent())
end_audio!(s::LiveSession) = send!(s, LiveClientEndAudioEvent())
function recv(s::LiveSession)
    lock(s.receive_lock) do
        s.closed && throw(TransportError("live session is closed"; provider=s.lm.provider))
        while isempty(s.pending)
            raw=try
                HTTP.WebSockets.receive(s.socket)
            catch error
                error isa InterruptException && rethrow()
                throw(TransportError("live receive failed"; provider=s.lm.provider))
            end
            append!(s.pending, decode_live_frame(s.lm, raw))
        end
        return popfirst!(s.pending)
    end
end
function Base.close(s::LiveSession)
    s.closed && return nothing
    s.closed=true
    close(s.socket)
    return nothing
end
"""Open a scoped live session: `live(lm, config) do session ... end`."""
function live(f, l::ProviderLM, c::LiveConfig)
    frames=live_setup_frames(l, c)
    credential=resolve_credential(l.credential)
    scheme=select_scheme(l.access, credential)
    uri=HTTP.URI(l.base_url)
    host=uri.host*(isempty(uri.port) ? "" : ":"*uri.port)
    protocol=uri.scheme=="https" ? "wss" : "ws"
    headers=Dict{String,String}(lowercase(k)=>String(v) for (k, v) in l.access.headers)
    if l.dialect=="gemini"
        credential isa ApiKey || unsupported(l.provider, "non-key Gemini Live credentials")
        url=with_query(
            "$protocol://$host/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent",
            obj("key"=>credential.value),
        )
    else
        url=with_query("$protocol://$host$(rstrip(uri.path,'/'))/realtime", obj("model"=>c.model))
        scheme in ("bearer", "api-key", "x-api-key") ||
            unsupported(l.provider, "live credential scheme")
        merge!(headers, credential_headers(l.access, credential))
    end
    returned = nothing
    primary = nothing
    completed = false
    try
        websocket_open(
            url;
            headers=collect(headers),
            retry=false,
            redirect=false,
            readtimeout=120,
            suppress_close_error=true,
        ) do ws
            session=LiveSession(ws, l, c)
            try
                for frame in frames
                    HTTP.WebSockets.send(ws, wire_json(frame))
                end
                if l.dialect=="gemini"
                    while true
                        raw=HTTP.WebSockets.receive(ws)
                        d=JSON.parse(raw isa AbstractVector{UInt8} ? String(copy(raw)) : raw)
                        haskey(d, "setupComplete") && break
                        haskey(d, "error") && throw(
                            error_for_code(
                                frame_error(l, d).error.code,
                                "live setup failed";
                                provider=l.provider,
                            ),
                        )
                        append!(session.pending, decode_live_frame(l, d))
                    end
                end
                returned = f(session)
                completed = true
            catch error
                primary = error
                rethrow()
            finally
                # WebSockets.open owns the socket and closes it on exit. Do not
                # close twice or replace an application exception with cleanup.
                session.closed = true
            end
        end
    catch error
        primary === nothing || throw(primary)
        if completed
            @warn "Live connection cleanup failed after the block completed; result preserved"
            return returned
        end
        error isa InterruptException && rethrow()
        throw(TransportError("could not open the live connection"; provider=l.provider))
    end
    return returned
end

Base.@kwdef struct Turn
    ended_by::String
    text::String = ""
    audio::Vector{UInt8} = UInt8[]
    audio_media_type::Maybe{String} = nothing
    tool_calls::Tuple = ()
    usage::Maybe{Usage} = nothing
    error::Maybe{ErrorDetail} = nothing
    events::Tuple = ()
end
ok(t::Turn) = t.ended_by=="turn_end"
function sum_usage(a::Maybe{Usage}, b::Usage)
    a===nothing && return b
    fields=Dict{Symbol,Any}()
    for key in fieldnames(Usage)
        x=getfield(a, key)
        y=getfield(b, key)
        fields[key]=x===nothing || y===nothing ? nothing : checked_token_sum(x, y)
    end
    return Usage(; fields...)
end
function materialize_turn(events)
    words=IOBuffer()
    audio=UInt8[]
    mime=nothing
    calls=ToolCallInfo[]
    usage=nothing
    error=nothing
    for e in events
        if e isa LiveServerTextEvent
            write(words, e.text)
        elseif e isa LiveServerAudioEvent
            mime===nothing ||
                e.media_type===nothing ||
                mime==e.media_type ||
                throw(ArgumentError("turn contains different audio formats; consume raw events"))
            mime===nothing && (mime=e.media_type)
            append!(audio, base64decode(base64_payload(e.data)))
        elseif e isa LiveServerToolCallEvent
            push!(calls, ToolCallInfo(; id=e.id, name=e.name, input=e.input))
        elseif e isa Union{LiveServerUsageEvent,LiveServerTurnEndEvent}
            usage=sum_usage(usage, e.usage)
        elseif e isa LiveServerErrorEvent
            error=e.error
        end
    end
    boundary=isempty(events) ? "incomplete" : kind(last(events))
    boundary in ("turn_end", "interrupted", "error", "tool_call") || (boundary="incomplete")
    return Turn(;
        ended_by=boundary,
        text=String(take!(words)),
        audio,
        audio_media_type=mime,
        tool_calls=Tuple(calls),
        usage,
        error,
        events=Tuple(events),
    )
end
# Bounded turn collection (changes/2026-09-15-live-collection-limits.md): each view
# retains at most max_bytes of accepted events (their compact ASCII canonical JSON)
# and max_events events; past either it fails with CollectionLimitError, keeping
# what it accepted, and stays sealed. The session stays open under the caller's control.
const DEFAULT_TURN_MAX_BYTES = 16 * 1024 * 1024
const DEFAULT_TURN_MAX_EVENTS = 10_000
ascii_json_size(x::Nothing) = 4
ascii_json_size(x::Bool) = x ? 4 : 5
ascii_json_size(x::Integer) = ncodeunits(string(x))
ascii_json_size(x::AbstractFloat) = ncodeunits(JSON.serialize(x))
function ascii_json_size(s::AbstractString)
    n = 2
    for c in s
        if c == '"' || c == '\\' || c in ('\b', '\f', '\n', '\r', '\t')
            n += 2
        elseif ' ' <= c <= '~'
            n += 1
        else
            n += isvalid(c) && codepoint(c) > 0xffff ? 12 : 6
        end
    end
    return n
end
ascii_json_size(d::AbstractDict) = 2 + max(length(d) - 1, 0) + sum((ascii_json_size(String(k)) + 1 + ascii_json_size(v) for (k, v) in d); init=0)
ascii_json_size(v::Union{AbstractVector,Tuple}) = 2 + max(length(v) - 1, 0) + sum((ascii_json_size(x) for x in v); init=0)
"""The byte charge of one live server event: its canonical JSON, compact, ASCII-escaped."""
live_event_size(e::LiveServerEvent) = ascii_json_size(to_dict(e))
function check_budget(name, value)
    value isa Integer && !(value isa Bool) && value > 0 ||
        throw(ArgumentError("$name must be a positive integer"))
    return Int(value)
end
mutable struct TurnView{S}
    session::S
    events::Vector{LiveServerEvent}
    closed::Bool
    terminal::Bool
    max_bytes::Int
    max_events::Int
    retained_bytes::Int
    failure::Union{Nothing,CollectionLimitError}
end
"""
    turn(session; max_bytes=16 MiB, max_events=10_000) -> TurnView

A view over the next turn: iterate its events, `snapshot` it, or `result` it. The view
retains every event it yields, within its budgets; past one it throws
`CollectionLimitError` (which keeps the accepted events and, for a byte overflow, the
event that did not fit). Raw `recv(session)` collects nothing.
"""
turn(s::LiveSession; kw...) = turn_view(s; kw...)
# Any source with a `recv` method (the live session; a scripted session in tests).
turn_view(s; max_bytes=DEFAULT_TURN_MAX_BYTES, max_events=DEFAULT_TURN_MAX_EVENTS) =
    TurnView(s, LiveServerEvent[], false, false, check_budget("max_bytes", max_bytes),
        check_budget("max_events", max_events), 0, nothing)
Base.IteratorSize(::Type{<:TurnView}) = Base.SizeUnknown()
Base.eltype(::Type{<:TurnView}) = LiveServerEvent
function limit_failure(view::TurnView, limit, maximum, rejected=nothing)
    what = limit == "max_bytes" ? "$(maximum) bytes" : "$(maximum) events"
    view.failure = CollectionLimitError(ErrorMetadata(;
        message="Live turn collection reached its configured budget ($limit = $what); the session is still open",
        limit, maximum, retained_bytes=view.retained_bytes, partial_events=Tuple(view.events), rejected_event=rejected))
    return view.failure
end
function Base.iterate(view::TurnView, state=nothing)
    view.failure === nothing || throw(view.failure)
    (view.closed || view.terminal) && return nothing
    # A cap is a cap: at the event limit, nothing more is read.
    length(view.events) >= view.max_events && throw(limit_failure(view, "max_events", view.max_events))
    event=recv(view.session)
    size=live_event_size(event)
    size > view.max_bytes - view.retained_bytes && throw(limit_failure(view, "max_bytes", view.max_bytes, event))
    push!(view.events, event)
    view.retained_bytes += size
    event isa Union{LiveServerTurnEndEvent,LiveServerInterruptedEvent,LiveServerErrorEvent} &&
        (view.terminal=true)
    return event, nothing
end
function snapshot(view::TurnView)
    view.failure === nothing && return materialize_turn(view.events)
    t=materialize_turn(view.events)
    return Turn(; (n=>getfield(t, n) for n in fieldnames(Turn))..., ended_by="incomplete")
end
Base.close(view::TurnView) = (view.closed=true; nothing)
function result(view::TurnView)
    view.failure === nothing || throw(view.failure)
    if !isempty(view.events) && last(view.events) isa LiveServerToolCallEvent
        return snapshot(view)
    end
    for event in view
        event isa LiveServerToolCallEvent && break
    end
    result=snapshot(view)
    result.ended_by=="incomplete" &&
        throw(TransportError("turn closed before a boundary; inspect snapshot"))
    return result
end
