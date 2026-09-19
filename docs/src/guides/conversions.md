# Convert data without changing its meaning

A model supplies JSON. Julia functions may expect types that JSON cannot describe
by itself. LM15 converts the deliberately selected tool interface, not every Julia
object it encounters.

## Input types

| Selected Julia type | Representation and rule |
|---|---|
| `String`, `Bool` | String or boolean; no string-to-number guessing |
| Concrete integers, including `BigInt` | Integral value; reject booleans, fractions and overflow |
| `Float16`, `Float32`, `Float64` | Finite number; reject precision loss during conversion |
| `Nothing`, `Union{Nothing,T}` | Null, or the representation of T |
| `Vector{T}`, `AbstractVector{T}` | Array of checked T elements; an explicit interface chooses dispatch |
| Concrete named tuple | Object with named required fields |
| `Dict{String,T}` | Object whose values each decode as T |
| Enum | An existing enum name, never evaluated as code |
| `Complex{T}` | Object with `real` and `imag` fields |
| Concrete Unitful quantity | Object with `value` and the exact declared `unit` |

Matrices, untyped `Any`, general unions, arbitrary structs and unspecified array
representations require a wrapper or custom conversion. A static/vector package
can define a codec for its own type rather than being silently converted to a
particular storage layout.

## Optional does not mean nullable

`x::Union{Nothing,Int}` permits null but remains required unless a default exists.
Defaults run as ordinary Julia expressions when an explicit execution omits an
argument. They are not evaluated to populate schema metadata.

Omit optional positional arguments only as a trailing group. Independent optional
settings should be keywords. Varargs, `where` interfaces and untyped argument names
need a small explicit wrapper; `@tool` does not guess an entire method table.

## Precision

The JSON parser reads decimal numbers as Float64. Converting its `0.1` to Float32
would change the represented value, so the default decoder refuses. When rounding
is intended, expose Float64 and perform a deliberate Float32 conversion in the
function. No decoder can recover decimal digits already rounded by a JSON parser;
use a checked decimal-string representation when that matters.

Integer parsing retains BigInt precision, then checks the selected target type.
That preserves the local data representation, not a guarantee that a remote model
performs exact arithmetic. Let Julia do numerical work and validate the result.

## Output types

`tool_content` keeps strings as text, passes presentational parts through, and writes
ordinary numbers, booleans, null, vectors, tuples, named tuples and string-keyed
objects as JSON text. Complex values retain both components. Float16/32 output is
promoted exactly before JSON encoding; BigFloat requires an explicit representation.
NaN and infinity are not strict JSON numbers.

`missing` is not automatically null. Choose an application policy. Tables support
an explicit `cell` converter. `array_content` records shape, column-major order and
values; it refuses offset axes rather than erasing their meaning. Large arrays and
tables should usually be summarized before encoding.

A binding's `output` callback must return text or supported parts, not a raw arbitrary
object. Use `tool_content`, `table_content`, or `array_content` as appropriate.
Output conversion happens after execution and can fail after a side effect.

See [custom codecs](extensions.md), [units](../tutorials/units.md), and
[tool reference](../reference/tools.md).
