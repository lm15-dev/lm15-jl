# Read a response as it arrives

Print text fragments while a ResponseStream assembles their complete Response.
Learn why a terminal event matters, why the scoped block owns cleanup, and why an
exhausted source or early close is not a completed answer.

This lesson uses synthetic local events. It does not contact a provider. The
separate [live stream script](../../../examples/tutorials/live-stream.jl) shows the
potentially paid version and requires explicit authorization before running.

[Read the complete annotated Julia lesson](../../../examples/tutorials/streaming.jl).
The documentation build renders it; the script and its live counterpart
(`examples/tutorials/live-stream.jl`, run against OpenAI on 2026-09-26) were run.

Next: [tool calls](tools.md) and [resource ownership](../guides/lifecycle.md).
