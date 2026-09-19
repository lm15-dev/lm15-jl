# Migrate Chat-style code deliberately

Importing a request is different from configuring a client. LM15 rejects
unrepresentable fields rather than silently dropping them. The examples here are
local data conversion unless an explicit provider call is shown.

```julia
using LM15
body = Dict("model" => "example-model", "messages" => [
    Dict("role" => "user", "content" => "Hello"),
])
req = request_from_openai_chat(body)
```

That returns a canonical Request. Use its messages/configuration with the intended
provider client, not an assumed global account.

`request_from_openai_chat(client, body)` uses client context. The router overload
`request_from_openai_chat(router, model, messages; kwargs...)` returns a pair
`(request, client)` and can resolve client credentials. It does not submit generation.

Transport headers, keys, endpoints, timeouts and retry options belong on RouterConfig
or the client. They are rejected if mixed into the migration request's keyword
arguments. `drop_params` is not a supported way to make a request succeed by losing
its intent.

## Model names

`openai_chat_model_string` translates known foreign provider/model prefixes.
Unknown slash prefixes are errors. `resolve_openai_chat` routes an implicit OpenAI
family match through Chat; an explicit provider prefix keeps its meaning. Prefer
clear `provider:model` names when moving an application.

## Responses and execution

`response_from_openai_chat(body; model=..., choice=...)` parses an already obtained
Chat-style response. This is local parsing, not another provider call.
`complete_from_openai_chat` and `stream_from_openai_chat` perform the actual
submission and therefore can incur charges. Stream ownership still applies.

Keep approval, tool execution, retries and conversation history explicit. Loading
serialized requests restores data, not callable bindings or permissions. Rebind
trusted local functions separately and do not evaluate response text as Julia code.

See [advanced API forms](../reference/advanced.md) and [release notes](../project/releases.md).
