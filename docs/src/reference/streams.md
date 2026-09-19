# Streams and live sessions

```@meta
CurrentModule = LM15
```

## Response streams

```@docs
stream
ResponseStream
StreamAccumulator
push!(::StreamAccumulator, ::StreamEvent)
response
events
text_chunks
materialize_response
response_to_events
coalesce_stream
SSEEvent
parse_sse
```

## Canonical events and deltas

Part indices in this layer are zero-based. Do not use a partial tool-argument delta
as an executable invocation.

```@docs
StreamEvent
StreamStartEvent
StreamDeltaEvent
StreamEndEvent
StreamErrorEvent
Delta
TextDelta
ThinkingDelta
AudioDelta
ImageDelta
ToolCallDelta
CitationDelta
ContinuationDelta
```

## Live ownership and turn access

[`result`](@ref) has both a VideoJob form and a TurnView form; its reference is
shared with the resource operations rather than duplicated here.

```@docs
AudioFormat
LiveConfig
LiveSession
live
Turn
TurnView
turn
snapshot
ok
send!
send_text!
send_turn!
send_audio!
send_image!
send_tool_result!
interrupt!
end_audio!
recv
```

## Live event types

```@docs
LiveClientEvent
LiveClientTurnEvent
LiveClientAudioEvent
LiveClientImageEvent
LiveClientTextEvent
LiveClientToolResultEvent
LiveClientInterruptEvent
LiveClientEndAudioEvent
LiveServerEvent
LiveServerAudioEvent
LiveServerTextEvent
LiveServerToolCallEvent
LiveServerToolCallDeltaEvent
LiveServerInterruptedEvent
LiveServerTurnEndEvent
LiveServerUsageEvent
LiveServerErrorEvent
```

## Closing owned objects

```@docs
close(::ResponseStream)
close(::TurnView)
close(::LiveSession)
close(::LMRouter)
close(::ProviderLM)
```
