# Convert data without changing its meaning

A model supplies JSON. LM15 hands a tool call's input to you as that JSON, and
turns your function's result into a tool answer. Converting the input into the
Julia types your function takes is your code's job: LM15 does not derive tools from
Julia methods or convert their arguments (removed before 1.0, 2026-09-23).

## Input values

A call's `input` holds what the JSON parser produced: `String`, `Bool`, `nothing`,
`BigInt` for whole numbers, `Float64` for decimals, and string-keyed dictionaries
and vectors of those. Check what you received against what your tool's schema
offered, then convert deliberately; the [units tutorial](../tutorials/units.md)
shows a checked quantity, and the [tool tutorial](../tutorials/tools.md) a checked
integer. A string is never evaluated as code.

## Precision

The JSON parser reads decimal numbers as Float64. Converting its `0.1` to Float32
changes the represented value; when rounding is intended, do it deliberately in the
function. Nothing can recover decimal digits already rounded by a JSON parser; use a
checked decimal-string representation when that matters.

Integer parsing retains BigInt precision; convert to your function's type yourself.
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

Use `tool_content`, `table_content`, or `array_content` as appropriate. Output
conversion happens after your function ran, and can fail after a side effect.

See [custom result types](extensions.md), [units](../tutorials/units.md), and
[tool reference](../reference/tools.md).
