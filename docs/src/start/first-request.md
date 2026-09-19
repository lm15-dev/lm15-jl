# Make your first request

**Goal:** obtain text and reported token usage from one model call.

**Requires:** LM15 in your project, a provider account, an allowed model and its key.
**Effect:** sends the prompt to the chosen provider and may incur charges.
This live example is not executed by the documentation build and was not run during
this writing pass.

## Configure the account outside your source file

Set `OPENAI_API_KEY` in the environment of the Julia process using your normal secret
manager or shell setup. Do not commit a key, paste it into a shared notebook, or
print the environment to debug it. Shell syntax varies by operating system.

Set `LM15_MODEL` to a model your account can use, or review the example model below.
Model names and availability can change independently of LM15.

## The complete call

```julia
using LM15

client = OpenAILM(api_key=ENV["OPENAI_API_KEY"])
model = get(ENV, "LM15_MODEL", "gpt-4.1-mini")
req = Request(model, user("Explain why the sky looks blue in two sentences.");
    config=Config(max_tokens=100))

answer = complete(client, req)              # The provider request happens here.
println("Finish reason: ", answer.finish_reason)
visible = text(answer)
if visible === nothing
    println("The response is not plain text; inspect answer.message.parts.")
else
    println(visible)
end
println("Input tokens: ", answer.usage.input_tokens)
println("Output tokens: ", answer.usage.output_tokens)
```

The reply's wording and usage vary. A missing usage counter prints as `nothing`;
it does not mean zero. `max_tokens` limits requested output, not your total account
bill, input cost or all provider-side work.

## What each object does

- `client` selects the provider and the explicit credential. It does not hold an
  invisible conversation for you.
- `req` holds the model, messages and generation settings. Constructing it does not
  send a request.
- `complete` submits it and returns a `Response`. It does not run Julia tools or retry
  on failure.
- `text` reads a text-representable answer. Mixed content, a refusal or a tool call
  may need inspection of the actual parts instead.

For direct clients, use the provider's model name. With `LMRouter`, use an explicit
`provider:model` to select both. Read [routing](../guides/providers.md).

## If it fails

A missing environment key is local configuration. Authentication errors may mean
an expired token or the wrong account. An unsupported model may mean the model
name, provider or account permissions are wrong. Do not solve these by printing
secrets or silently selecting another account.

Use the [credential guide](../guides/credentials.md) and
[error-handling guide](../guides/errors.md). A failed connection does not prove that
the provider did no work; avoid blind retries.

## Next

[Continue the conversation](../tutorials/conversation.md), then
[read a streamed answer](../tutorials/streaming.md).
