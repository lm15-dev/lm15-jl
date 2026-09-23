@doc """
    stream(client_or_router, request::Request)
    stream(f, client_or_router, request::Request)

Create a lazy stream of canonical events, or run a scoped block with a
ResponseStream. The unscoped stream starts its producer when consumed; close it
when leaving early. The scoped form closes on every exit and returns the block's
value. Events use bounded channels rather than buffering the full network stream.

A ResponseStream still accumulates content needed for its final Response. Streaming
therefore limits transport buffering, not the eventual size of a retained answer.
Breaking a loop alone does not release a blocked read. Closing or encountering a
transport error does not guarantee that remote work or billing stops.

Use text_chunks for text fragments and response to drain and assemble the result.
See also [`ResponseStream`](@ref), [`StreamAccumulator`](@ref), [`complete`](@ref).
""" stream

for (name, signature, explanation) in (
    (:StreamAccumulator, "StreamAccumulator(request::Request)", "Collect canonical events locally with push! and assemble them using response. This low-level object is not a network connection or a stream-lifecycle validator. Use ResponseStream to require a proper end event and manage source closing."),
    (:response, "response(accumulator::StreamAccumulator)\n    response(stream::ResponseStream)", "Assemble the accumulator's current data, or drain a ResponseStream and return its completed Response. The stream form requires an end event; the low-level accumulator form assembles the available state. Invalid tool fragments or inconsistent content can raise StreamAssemblyError with a partial answer."),
    (:events, "events(stream::ResponseStream)", "Return the same ResponseStream as a typed event iterator. It shares consumption state with text_chunks and response; do not independently consume it from multiple tasks."),
    (:text_chunks, "text_chunks(stream::ResponseStream)", "Iterate only the text strings from TextDelta events while advancing and assembling all events. Reasoning, citations and tool fragments are not returned as text chunks; inspect the final Response or use events when needed."),
    (:materialize_response, "materialize_response(source, request)", "Own a source through ResponseStream, consume it to a completed Response and close it on every exit. With a network source this performs I/O; with synthetic events it is local. A source that ends without a terminal event fails."),
    (:coalesce_stream, "coalesce_stream(source; model=nothing)", "Normalize repeated stream starts/ends into one public lifecycle, collecting final metadata without inventing an end when none was received. Returns a lazy, closeable event source; source closing is owned by the wrapper."),
    (:SSEEvent, "SSEEvent(event, data)", "Hold an optional SSE event name and its joined data lines. It is framing data, not yet a canonical StreamEvent."),
    (:parse_sse, "parse_sse(emit, io::IO; max_line_bytes=65536, max_event_bytes=1048576)\n    parse_sse(io::IO; kwargs...)", "Read server-sent-event framing, emitting SSEEvent values through a callback or collecting them into a vector. Accept LF, CRLF and CR line endings, enforce configured limits and handle split UTF-8 input. The caller owns io; this parser does not open a provider connection or interpret its JSON."),
    (:AudioFormat, "AudioFormat(; encoding, sample_rate, channels=1)", "Declare a live audio format. encoding is pcm16, opus, mp3 or aac; rate and channel count must be positive. Provider-supported combinations are narrower than the canonical vocabulary."),
    (:LiveConfig, "LiveConfig(; model, system=nothing, tools=(), voice=nothing, input_format=nothing, output_format=nothing, extensions=nothing)", "Configure a live session. Tools are FunctionTool or BuiltinTool values. Provider support determines voices, formats and extensions. Creating the value neither opens a socket nor runs tools."),
    (:LiveSession, "LiveSession", "Represent an open live connection owned by live's do block. Use send!/recv or turn instead of manipulating socket/pending fields. Leaving the block closes the socket. Reads and sends can block; media data may be billable."),
    (:Turn, "Turn(; ended_by, text=\"\", audio=UInt8[], audio_media_type=nothing, tool_calls=(), usage=nothing, error=nothing, events=())", "Hold a materialized live-turn snapshot. ended_by is turn_end, interrupted, error, tool_call or incomplete. audio contains decoded bytes; tool_calls contains ToolCallInfo. Unknown usage counters remain unknown when summed. Use ok to check for ordinary completion."),
    (:TurnView, "TurnView", "Iterate one live turn from turn(session). Iteration continues across tool calls until an end, interruption or error; result(view) instead returns at a tool call so your application can respond. Closing a view does not close its session."),
    (:turn, "turn(session::LiveSession)", "Create a TurnView without receiving data yet. Its iterator and result perform receives; snapshot only inspects events already collected."),
    (:snapshot, "snapshot(view::TurnView)", "Materialize the events already collected without receiving another frame. Can return an incomplete Turn. Incompatible mixed audio formats can require consuming raw events instead."),
    (:ok, "ok(turn::Turn)", "Return true only when ended_by is turn_end. A tool call, interruption, error or incomplete snapshot is not an ordinarily finished turn."),
    (:send!, "send!(session::LiveSession, event::LiveClientEvent)", "Encode and send one canonical live event, possibly producing multiple frames. Mutates connection state, contacts the provider and returns nothing. Does not automatically receive an answer."),
    (:send_text!, "send_text!(session, text)", "Send a LiveClientTextEvent and return nothing. Provider semantics differ from sending an explicitly completed turn; use send_turn! when that is the intended operation."),
    (:send_turn!, "send_turn!(session, content; turn_complete=true)", "Send prompt content as a live turn. content accepts ordinary text and prompt parts. The default marks the turn complete; returns nothing after sending."),
    (:send_audio!, "send_audio!(session, data; media_type=\"audio/pcm;rate=16000\")", "Send live audio bytes or base64 text. The MIME declaration must describe the actual bytes; this function performs no recording, resampling or codec conversion."),
    (:send_image!, "send_image!(session, data; media_type=\"image/jpeg\")", "Send encoded image bytes or base64 text to a live session. This is not an image-matrix encoder and does not infer dimensions or format from pixels."),
    (:send_tool_result!, "send_tool_result!(session, id, content)", "Send presentational content answering a live tool call. Use the call's ID. Pass tool_content(value) for a Julia value; do not nest ToolResultPart as live content."),
    (:interrupt!, "interrupt!(session)", "Send the provider's supported interruption event and return nothing. This does not promise immediate cancellation or reversal of charges."),
    (:end_audio!, "end_audio!(session)", "Signal that the current input-audio stream has ended where supported. It does not close the whole live session."),
    (:recv, "recv(session::LiveSession)", "Receive the next canonical LiveServerEvent, reading more WebSocket frames if necessary. May block and may return an error event rather than throwing for every provider-level failure. Transport failure raises TransportError."),
)
    doc = "    " * signature * "\n\n" * explanation
    @eval @doc $doc $name
end

@doc """
    live(f, client_or_router, config::LiveConfig)

Open a live session, run `f(session)`, and close the WebSocket on every exit.
Return the block's value. Setup resolves credentials and contacts the provider;
there is no unscoped `live(client, config)` socket constructor. Live calls can
incur charges and do not automatically execute tools or stop remote billing.

Use `result(turn(session))` to receive a response or yield at a tool call.
Use `recv` for full-duplex event handling. The live path uses its own WebSocket
connection and a 120-second underlying read timeout; `HTTPTransport` settings do
not configure it. Check `ok(turn)` or `ended_by` before treating it as finished.
""" live

for (name, signature, explanation) in (
    (:LiveClientTurnEvent, "LiveClientTurnEvent(; parts, turn_complete=true)", "Send nonempty prompt parts as a turn."),
    (:LiveClientAudioEvent, "LiveClientAudioEvent(; data, media_type=\"audio/pcm;rate=16000\")", "Send base64 audio with an explicit matching MIME format."),
    (:LiveClientImageEvent, "LiveClientImageEvent(; data, media_type=\"image/jpeg\")", "Send base64-encoded image data, not a Julia pixel array."),
    (:LiveClientTextEvent, "LiveClientTextEvent(; text=\"\")", "Send text through the provider's live text mapping."),
    (:LiveClientToolResultEvent, "LiveClientToolResultEvent(; id, content)", "Send nonempty presentational Parts answering one tool call ID."),
    (:LiveClientInterruptEvent, "LiveClientInterruptEvent()", "Request interruption where the provider supports it."),
    (:LiveClientEndAudioEvent, "LiveClientEndAudioEvent()", "Signal the end of an input-audio segment."),
    (:LiveServerAudioEvent, "LiveServerAudioEvent(; data, media_type=nothing)", "Receive base64 audio; Turn assembly decodes it to bytes."),
    (:LiveServerTextEvent, "LiveServerTextEvent(; text=\"\")", "Receive a live text fragment."),
    (:LiveServerToolCallEvent, "LiveServerToolCallEvent(; id, name, input=Dict{String,Any}())", "Receive a complete proposed tool invocation. It does not execute anything."),
    (:LiveServerToolCallDeltaEvent, "LiveServerToolCallDeltaEvent(; input_delta=\"\", id=nothing, name=nothing)", "Receive an incomplete tool-argument fragment; do not execute it as a complete call."),
    (:LiveServerInterruptedEvent, "LiveServerInterruptedEvent()", "Observe provider-reported interruption, distinct from ordinary completion."),
    (:LiveServerTurnEndEvent, "LiveServerTurnEndEvent(; usage=Usage())", "Observe an ordinary live-turn boundary and optional usage."),
    (:LiveServerUsageEvent, "LiveServerUsageEvent(; usage=Usage())", "Receive reported usage. Missing counters stay nothing rather than zero."),
    (:LiveServerErrorEvent, "LiveServerErrorEvent(; error)", "Receive canonical ErrorDetail. A materialized Turn records this as an error boundary."),
)
    doc = "    " * signature * "\n\n" * explanation * " Constructing an event performs no I/O; send! is explicit for client events."
    @eval @doc $doc $name
end

@doc """
    push!(acc::StreamAccumulator, event::StreamEvent)

Validate and append one canonical event to a local accumulator. Part indices are
zero-based. This does not receive from a socket or enforce the whole stream
lifecycle; ResponseStream adds ownership and terminal-event checking.
""" Base.push!(::StreamAccumulator, ::StreamEvent)

@doc """
    close(stream::ResponseStream)

Close the owned source. Closing before the end event records StreamAssemblyError;
a partial answer must not masquerade as completed. Closing the source can unblock
network I/O but does not guarantee that provider billing stops.
""" Base.close(::ResponseStream)

@doc """
    close(view::TurnView)

Stop consuming this view without closing its live session. Use live's scoped form
to own the connection; closing a view alone is not remote cancellation.
""" Base.close(::TurnView)

@doc """
    close(session::LiveSession)

Explicitly close a live WebSocket. Prefer the owning live do block for normal
cleanup. This changes local connection state, not the provider's billing history.
""" Base.close(::LiveSession)

@doc """
    close(router::LMRouter)

Close cached client handles and empty the router's client cache. This does not
cancel already-created response streams or remote jobs; own those explicitly.
""" Base.close(::LMRouter)

@doc """
    close(client::ProviderLM)

Release the provider handle (currently a no-op). A client does not own previously
returned streams or jobs. Close each stream through its scoped API or explicitly.
""" Base.close(::ProviderLM)
