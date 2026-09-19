# Extend a Julia data or transport interface

Prefer a small explicit interface to reflecting every field of an application
object. These examples illustrate extension code; they were not executed in the
writing pass.

## An application-owned type

```julia
using LM15
struct Label
    value::String
end
LM15.tool_schema(::Type{Label}) = tool_schema(String)
LM15.tool_decode(::Type{Label}, value; path="input") =
    Label(tool_decode(String, value; path))
LM15.tool_value(value::Label) = value.value

label_tool = @tool "Count characters in a label" label_length(label::Label) = length(label.value)
```

| Method | Required when | Agreement |
|---|---|---|
| `tool_schema(T)` | T is a tool argument | Describe the JSON the decoder actually accepts |
| `tool_decode(T, value; path)` | T is a tool argument | Validate it, return a value of T, and preserve useful error paths |
| `tool_value(value::T)` | T can be a result | Return an explicit JSON-shaped representation |
| Binding `output` callback | A special result presentation is wanted | Return text or supported Parts, not an arbitrary raw object |

A decoder must not advertise one shape and accept a different one silently. LM15
also checks that custom decoders return the declared type, including inside supported
containers. Do not add new canonical Part/Tool variants for ordinary application
data; the shared wire representation is deliberately closed.

Extend LM15 functions for your own types. Avoid changing unrelated packages'
functions on unrelated types. Keep schema construction free of unexpected I/O and
make conversion policy explicit: precision, units, dimensions and missing values
are not details to discard.

## Existing methods, closures and callable objects

For an existing method, select the interface without redefining or calling it:

```julia
average(values::AbstractVector{Float64}) = sum(values) / length(values)
average_tool = @tool "Average a list" average(values::AbstractVector{Float64})
```

Execution uses Julia's `invoke` for the declared positional signature rather than
selecting a more specific method merely because decoding produced a concrete Vector.
If positional defaults are omitted, Julia's shorter default method is selected;
dispatch inside that method remains ordinary Julia behavior.

Closures and callable structs can supply the names and types explicitly:

```julia
increment = tool(x -> x + 1, NamedTuple{(:x,),Tuple{Int}}; name="increment")
struct Multiplier
    factor::Int
end
(m::Multiplier)(x::Int) = m.factor * x
multiply = tool(Multiplier(3), NamedTuple{(:x,),Tuple{Int}}; name="multiply")
```

For an existing optional keyword, pass `keywords=(:scale,)` and
`optional=(:scale,)` alongside the named-tuple argument type. The function owns its
actual default expression. Prefer the macro definition form when possible so names,
types and optional status come from the definition itself.

Descriptions remain explicit; the macro does not merge unrelated method docstrings
or infer domain constraints from the function body.

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
