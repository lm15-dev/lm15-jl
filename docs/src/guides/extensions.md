# Extend a Julia data or transport interface

Prefer a small explicit interface to reflecting every field of an application
object. These examples illustrate extension code; they are not run by the documentation build.

## An application-owned result type

```julia
using LM15
struct Label
    value::String
end
LM15.tool_value(value::Label) = value.value
```

| Method | Required when | Agreement |
|---|---|---|
| `tool_value(value::T)` | T can be a result | Return an explicit JSON-shaped representation |

With that method, `tool_content(Label("oak"))` answers a tool call with `"oak"`. Do
not add new canonical Part/Tool variants for ordinary application data; the shared
wire representation is deliberately closed.

Extend LM15 functions for your own types. Avoid changing unrelated packages'
functions on unrelated types. Make conversion policy explicit: precision, units,
dimensions and missing values are not details to discard.

Tool *inputs* are not extended here: LM15 does not convert them. Your code checks
and converts a call's JSON input (see [conversions](conversions.md)).

## A custom transport

A streaming transport implements one method:

```julia
using LM15
struct FixedTransport <: AbstractTransport
    response::HttpResponse
end
function LM15.open_response(f, transport::FixedTransport, wire::WireRequest)
    body = IOBuffer(transport.response.body)
    head = HttpResponse(status=transport.response.status,
        headers=transport.response.headers)
    try
        return f(head, body)
    finally
        close(body)
    end
end
```

The callback receives `head::HttpResponse` and an open `IO`. The transport owns that
IO and closes it on every exit, including callback exceptions. The same interface
supports complete and streamed requests. A callable `wire -> HttpResponse(...)`
is a simpler buffered alternative for fixtures, not a promise of streaming.

A replacement provider transport does not intercept separate credential HTTP or CLI
acquisition. For isolated examples use explicit synthetic credentials and controlled
settings, not a real ambient cloud chain. Do not log wire headers or bodies.

## Protocol adapters are a different layer

The exported wire builders, parsers and live codecs support integration and
conformance work. They do not provide an unrestricted registry for arbitrary new
router backends. Provider selection currently remains data-driven and protocol
selection uses string checks. See [advanced reference](../reference/advanced.md)
and [the pinned contract](../project/contract.md) before changing mappings.
