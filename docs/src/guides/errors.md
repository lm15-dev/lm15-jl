# Handle failures without losing control

There are three different failures to consider: an invalid local request, a failed
provider operation, and a failure in your application tool. They need different fixes.

## Catch the right family

The following is an application pattern for a potentially paid call, not an
executed documentation example. It assumes `client` and `req` are already configured.

```julia
try
    answer = complete(client, req)
    # Inspect finish_reason and content before using the answer.
catch error
    if error isa AuthError
        println("Check the selected account and credential source.")
    elseif error isa ContextLengthError
        println("Reduce or deliberately summarize the request history.")
    elseif error isa LM15Error
        println("Request failed with code: ", error.code)
    else
        rethrow()
    end
end
```

`ContextLengthError` and `UnsupportedModelError` belong to `InvalidRequestError`.
Device-login expiry belongs to `AuthError`. Errors may have status, provider code,
request ID, retry delay or credential hint; absent fields remain `nothing`.

Not every exception is an LM15Error. `ArgumentError` covers many invalid local
constructions. `WaitTimeout` and `UnknownProviderError` have their
own meanings. Your function can also throw its own exception, which is not disguised
as a successful tool result.

## Retry is an application decision

`retryable(error)` classifies likely transient transport, lock, rate-limit, timeout
and server errors. LM15 does not repeat the operation automatically. A connection
failure may happen after the provider started work; repeating a job submission or
side-effectful function can duplicate that work. Keep returned IDs and inspect
existing jobs when possible.

Never fix a failed request by silently dropping unsupported settings, changing the
provider account, discarding conversation parts or weakening type conversion.

## Incomplete streams

A `StreamAssemblyError` may contain `partial`. That can be useful for inspection or
UI recovery, but it is not a completed answer. The primary error wins over cleanup
failures. An early close cannot turn incomplete data into a success.

## Polling and transport timeouts

`WaitTimeout.snapshot` contains the last file/job snapshot. Remote work may still be
running. `HTTPTransport(connect_timeout=10, read_timeout=120)` configures positive
whole-second HTTP limits. These do not bound an entire multi-request tool loop,
interrupt a numerical computation, or configure the separate live WebSocket path.

See [ownership](lifecycle.md), [jobs](jobs.md), [live sessions](live.md), and
[error reference](../reference/errors.md).
