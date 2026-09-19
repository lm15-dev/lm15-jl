# Use a live session

Live sessions exchange events over a WebSocket. They require a supported model and
account and can incur charges while connected. The following is a live application
example, not an executable documentation block.

```julia
using LM15
client = OpenAILM(api_key=ENV["OPENAI_API_KEY"])
config = LiveConfig(model=ENV["LM15_LIVE_MODEL"])
answer = live(client, config) do session
    send_text!(session, "Explain a simple pendulum briefly.")
    result(turn(session))
end
if ok(answer)
    println(answer.text)
else
    println("Turn stopped at: ", answer.ended_by)
end
```

The block owns the socket and returns its own result. There is no unscoped
socket-returning `live(client, config)` constructor. Closing the block is not a
promise that every provider-side charge stops immediately.

## A turn can stop for different reasons

`ended_by` distinguishes `turn_end`, `tool_call`, `interrupted`, `error` and
`incomplete`. `ok` is true only for an ordinary `turn_end`. `snapshot(view)` reads
already-collected events; `result(view)` receives more data as needed.

Ordinary iteration over `turn(session)` continues across tool calls. In contrast,
`result(view)` returns at a complete tool call so the application can approve and
answer it. A call in `Turn.tool_calls` is a `ToolCallInfo`: convert it to
`ToolCallPart(info)`, check and execute the chosen binding, then send
`send_tool_result!(session, output.id, output.content)`. Do not send the whole
ToolResultPart as nested live content. Continue the same view after replying, with
an explicit application call/turn limit.

## Raw events and media

Use `recv(session)` for full-duplex applications that handle events themselves.
Do not race multiple consumers of the same turn. Send canonical events with `send!`
or helpers such as `send_turn!`, `send_audio!`, `send_image!`, `interrupt!` and
`end_audio!`. A tool-argument delta is not a complete call to execute.

Audio MIME declarations must match the actual encoding, rate and channel layout.
The library does not record, resample or transcode audio. A Turn materializes audio
bytes; incompatible format changes require consuming raw events. Reported usage
preserves unknown counters instead of converting them to zero.

## Timeouts and ownership

The live path has its own WebSocket setup and a 120-second underlying read timeout.
`HTTPTransport` settings configure ordinary HTTP calls, not this connection.
Closing a TurnView only stops that view; it does not close its session. The scoped
`live` form is the normal connection owner.

See [stream/live reference](../reference/streams.md), [privacy](privacy.md), and
[the support boundaries](../project/status.md).
