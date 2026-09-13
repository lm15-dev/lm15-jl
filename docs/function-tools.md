# Turn a Julia function into a model tool

A tool has two jobs that should not be confused:

1. **Describe an operation** to the model: its name, inputs and purpose.
2. **Run the operation** locally, if your application decides to allow it.

LM15 keeps these separate. Defining a tool, adding it to a request, building a
request and checking its arguments do not run the function.

## Start with an ordinary function

```julia
using LM15

square_tool = @tool "Square an integer" function square(n::Int; scale::Float64=1.0)
    return (answer=n^2 * scale,)
end

square(3) # Still an ordinary Julia function: (answer=9.0,)

req = Request("openai:gpt-4.1-mini", user("What is 3 squared?"); tools=square_tool)
println(to_json(FunctionTool(square_tool))) # Inspect the generated specification.
```

`@tool` reads the argument names and types in this definition. Here it describes
`n` as a required integer and `scale` as an optional number. It does not infer
limits from the function's body: domain rules still belong in your function.

The returned `ToolBinding` keeps the callable separate from the ordinary
`FunctionTool` put into `req.tools`. Only the specification is serialized. Requests
can mix bindings, handwritten `FunctionTool`s and provider-built-in tools.

## Check, approve, then execute

This entire example runs without a provider or credentials:

```julia
call = from_json(ToolCallPart, """
{"type":"tool_call","id":"call-1","name":"square","input":{"n":3}}
""")

arguments = tool_arguments(square_tool, call) # (n=3,), with n converted to Int
# Your application can ask for approval here.
output = execute_tool(square_tool, call)     # Explicitly runs exactly one function.

continued = Request(req; messages=[
    req.messages...,
    assistant(call),
    tool_message(output),
])
```

Checking rejects a wrong tool name, unknown arguments, missing required arguments,
wrong types, overflow and precision-losing conversions. Booleans are not accepted
as numbers. `tool_arguments` never runs defaults or the function body.

`execute_tool` uses the same checks, invokes the selected method and wraps its
output in a `ToolResultPart`. There is no automatic tool loop or retry. Function
exceptions propagate to your application rather than being disguised as success.
**Output conversion can fail after the function has already had side effects. Do
not retry an execution blindly.** A process or container sandbox is still your
application's responsibility; type checking is not a sandbox.

## Existing functions and multiple methods

```julia
average(values::AbstractVector{Float64}) = sum(values) / length(values)
average_tool = @tool "Average a list of numbers" average(values::AbstractVector{Float64})
```

The signature-only form neither defines nor calls the function. The explicit
signature chooses a method using Julia's `invoke`; it does not select a more
specific overload just because an input array became a concrete `Vector`. When
positional defaults are omitted, Julia's ordinary shorter default method is used;
any dispatch performed inside that method remains ordinary Julia behavior.
An operation such as SciML's
`solve` can have many methods and options: write one small, bounded wrapper for the
operation you actually want to expose.

For closures and callable structs, names/types can be supplied as a named-tuple
type. No inspection of Julia's compiler internals is needed:

```julia
increment = tool(x -> x + 1, NamedTuple{(:x,),Tuple{Int}};
    name="increment", description="Add one to an integer")
```

Use `tool(binding; name=..., description=..., output=...)` to change presentation
or output conversion without redefining the function. An `output` function must
return text, a supported Part, or a sequence of Parts; `tool_content`,
`table_content` and `array_content` already do this. Descriptions are explicit:
`@tool "description" ...` also works in notebooks. The macro does not guess
parameter descriptions from prose or combine documentation from unrelated methods.

### Defaults are not nulls

An argument with `Union{Nothing,Int}` permits JSON `null` but is **still required**
unless it has a default. Default expressions belong to Julia; they run only when
an explicit execution omits that argument. They are not evaluated while generating
the schema and are not copied into JSON as guessed constants.

Optional positional arguments must be omitted as a trailing group. You cannot
omit `y` but supply `z` in `f(x, y=2, z=3)`; that call is refused rather than
misinterpreted. Use keyword arguments for independent optional settings.

For an existing function with defaults, use the explicit interface:

```julia
existing = tool(square, NamedTuple{(:n,:scale),Tuple{Int,Float64}};
    keywords=(:scale,), optional=(:scale,), name="square")
```

Varargs, destructuring, `where` parameters, untyped arguments and general unions
are deliberately not guessed. Give the model a small typed wrapper instead.
Handwritten `FunctionTool(parameters=...)` remains available for any provider
schema beyond this derivation subset.

## Supported values

| Julia input type | JSON representation |
|---|---|
| `String`, `Bool` | String, boolean |
| Concrete integers, including `BigInt` | Integer; bounded integer types advertise their range |
| `Float16`, `Float32`, `Float64` | Finite number; conversion must not lose precision |
| `Nothing`, `Union{Nothing,T}` | Null, or null/the representation of T |
| `Vector{T}`, `AbstractVector{T}` | Array of checked T values |
| Concrete named-tuple types | Object with named, required fields |
| `Dict{String,T}` | Object with values of T |
| Julia enums | Enum names as strings; never evaluated as code |
| `Complex{T}` | Object with `real` and `imag` fields |
| Concrete Unitful quantities, when Unitful is loaded | Object with `value` and the exact declared `unit` |

Input floats are checked *after JSON parsing*: the JSON parser reads decimals as
Float64. A decimal such as `0.1` is therefore refused for `Float32` rather than
silently rounded. When rounding is intended, expose a Float64 argument and perform
the deliberate conversion in your function. Use an explicit string-based codec
when decimal or arbitrary precision matters; this layer cannot recover digits
already rounded by a JSON parser. Huge integer JSON literals remain exact here,
but this does not guarantee exact arithmetic inside a remote model.

Ordinary outputs become JSON text: numbers, booleans, null, vectors, tuples,
string-keyed dictionaries and named tuples are supported. Complex outputs retain
both components. Float16/32 values are promoted exactly before encoding so their
short display spelling does not change their value in a Float64 JSON reader.
Strings remain plain text. Supported content Parts, or sequences of Parts, pass
through as content. The canonical `tool_result` factory itself stays unchanged.

**`missing` is not silently turned into null.** Choose what it means in your
application. NaN, infinity and arbitrary-precision floats also require an explicit
representation. Dictionaries with non-string keys are refused, not renamed.

## Scientific packages

Tables and Unitful support uses Julia's optional package extensions. Neither
package is a mandatory LM15 dependency. There is no automatic serialization of an
entire DataFrame, arbitrary scientific object or solver solution.

### Tables and DataFrames

When Tables.jl is loaded, `table_content(table)` writes column names and rows. It
refuses more than 100 rows by default rather than quietly dropping the rest.
Select columns/rows first, or explicitly increase `max_rows`.

```julia
using DataFrames, Tables

measurements = DataFrame(trial=[1,2], height=[1.5,2.0])
content = table_content(measurements)
# A function returning a table can use:
# table_tool = tool(binding; output=table_content)

with_missing = DataFrame(x=[1,missing])
content = table_content(with_missing;
    cell=x -> ismissing(x) ? nothing : tool_value(x)) # Explicit missing→null policy.
```

### Physical units

```julia
using Unitful

speed_tool = @tool "Speed from metres and seconds" function measured_speed(
    distance::typeof(1.0u"m"), duration::typeof(1.0u"s"))
    duration > 0u"s" || throw(ArgumentError("duration must be positive"))
    distance / duration
end
```

The model supplies `{"value":12,"unit":"m"}` and
`{"value":3,"unit":"s"}`. The local function receives actual Unitful quantities.
The answer retains its value **and** unit. Wrong units are refused, not stripped
or guessed; incoming unit strings are never evaluated as Julia code. Support is
for concrete `Unitful.Quantity` types; logarithmic and other special unit wrappers
need an application-owned conversion.

### Matrices and simulations

`array_content(matrix)` explicitly includes its dimensions, column-major order and
values. It refuses offset axes; write an application representation for custom
indices. This materializes the array, so summarize large results first.

Keep solver objects local. Return a small named tuple containing the quantities the
model actually needs. The runnable [scientific examples](../examples/scientific/README.md)
exercise a real SciML differential-equation solve, DataFrames and Unitful together.

## Your own types

Extend the functions owned by LM15 for a type owned by your application. Keep
`tool_schema` and `tool_decode` consistent: a decoder must check the advertised
rules and return the declared Julia type. `tool_value` defines a deliberate output
representation. LM15 does not reflect every struct's internal fields into a schema.

```julia
struct Label
    value::String
end
LM15.tool_schema(::Type{Label}) = tool_schema(String)
LM15.tool_decode(::Type{Label}, value; path="input") =
    Label(tool_decode(String, value; path))
LM15.tool_value(value::Label) = value.value
```

These are ordinary multiple-dispatch methods. They add no fields or variants to
the language-neutral LM15 contract.
