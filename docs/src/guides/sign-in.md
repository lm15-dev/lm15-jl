# Sign in once, use everywhere

Some providers can be used through an account you already pay for (an xAI, Claude,
ChatGPT, GitHub Copilot, Kimi Code or Meta subscription), or through a key you would
rather save once than export in every shell. [`Auth`](@ref) keeps those connections in a
private file that LM15 shares across languages: a login made from Python or Go is usable
from Julia, and the other way round.

## At a terminal

```julia
using LM15, LM15.Interactive
connect() do lm                  # choose or make a connection, then a model
    answer = complete(lm, "Explain drought stress in two sentences.")
    println(text(answer))
end
```

`connect` shows where connections are saved, offers the saved ones first (subscriptions
before keys), runs a sign-in if you choose one, lists the models your account can use and
returns a client bound to that one connection and model. It never sends a prompt, never
sets a process default, and a completed sign-in stays saved even if you cancel the model
choice. Without a terminal it refuses rather than reading input it cannot get.

## In a program

```julia
using LM15
auth = local_auth()                          # reads nothing yet
login(auth, "xai", "device"; ui=TerminalUI())
router = LMRouter(RouterConfig(; auth))
answer = complete(router, Request("xai:grok-4.7", user("hello")))
```

With an `Auth` attached, a router takes, in order: an explicit `api_keys` entry, an
explicit named cloud identity (`credentials`), the saved connection. It does **not** fall
back to an environment key or the machine's cloud identity: a signed-out or failed
subscription stops the request instead of charging a key you did not choose. To use a key
deliberately, pass it in `api_keys`, or save it: `set_api_key(auth, "openai", key)`, or
`configure(auth, "gemini"; method="env", answers=Dict("name" => "GEMINI_API_KEY"))` to
read a variable at request time.

## What each operation touches

| Operation | Effect |
|---|---|
| `login_providers()`, `login_methods(provider)` | definitions only |
| `connections(auth)`, `status(auth, provider)` | reads the store; no network, no renewal |
| `login(auth, provider, method; ui)` | the sign-in flow, then an atomic save |
| `set_api_key`, `configure` | an atomic save; nothing is verified |
| `verify(auth, provider)` | lists the route's models once (may be metered) |
| `logout(auth, provider_or_id)` | forgets locally and keeps a marker so no ambient key replaces it; revokes nothing at the provider |
| `cancel_login(auth, provider)` | durable cancel: `"cancelled"`, `"complete"` or `"none"` |

A saved connection has an identity generation (changes on every new sign-in and logout)
and a credential revision (changes on every renewal). A client from `connect` pins the
connection id and generation: after a replacement or logout it fails with
`AuthOperationError` (`connection_changed`, `login_required`) instead of silently using a
different account.

## What is verified

Methods say how far they have been checked: `supported` (live receipts exist),
`unverified` (the code exists; pass `allow_unverified=true`), `unavailable` (not here). An
account sign-in is not a promise of included usage or of the provider's permission; check
your plan. The file is private (mode 0600 on Unix) but not encrypted.

Failures are [`AuthOperationError`](@ref)s with a `reason` a program can match on.
