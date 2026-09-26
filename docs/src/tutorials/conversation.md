# Keep a conversation explicit

Build a first Request, obtain an assistant message, then append it and a follow-up
question to a new Request. Retain the whole assistant message rather than copying
only text and losing continuation state.

This lesson uses two clearly fabricated, in-memory HTTP replies: `8` and `64`.
It makes no network request and does not prove how an online model will answer.
The first Request stays unchanged; the follow-up carries its own full history.

[Read the complete annotated Julia lesson](../../../examples/tutorials/conversation.jl).
The build renders this source into the tutorial page; the script itself was run on
2026-09-26.

Next: [streaming](streaming.md) and [how LM15 works](../guides/model.md).
