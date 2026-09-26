# Check, approve and answer a tool call

Follow the request → proposed call → checked arguments → explicit execution → tool
result → follow-up request sequence. A received call is not permission to execute
code. A follow-up request is not submitted merely by being constructed.

The local lesson uses a synthetic provider call and a real Julia calculation. For
the actual two-request exchange and final provider response, the separate
[live tool script](../../../examples/tutorials/live-tools.jl) includes credential
setup, a confirmation prompt and a strict stopping rule. It can incur charges and
is never executed by the documentation generator.

[Read the complete annotated local lesson](../../../examples/tutorials/tools.jl).
The local lesson and the live one (`examples/tutorials/live-tools.jl`, run against
OpenAI on 2026-09-26: the model asked for `square(19)` and explained 361) were run.

Next: [conversion rules](../guides/conversions.md), [custom interfaces](../guides/extensions.md)
and [privacy](../guides/privacy.md).
