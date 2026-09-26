# Keep physical units through a tool call

Use a typed distance and duration to calculate speed. Preserve value and unit in
both directions; reject mismatched units instead of stripping them or interpreting
incoming strings as Julia expressions.

[Read the complete annotated Julia lesson](../../../examples/scientific/units.jl).
It uses real Unitful quantities and explicitly authored local calls. There is no
provider request. The documentation build runs it with the scientific assertions.

The build renders the source in the scientific environment and includes its plain
Markdown in the final manual. Large arrays still need an explicit representation
and selection policy even when their elements carry units.

Next: [a bounded simulation](simulation.md) and [conversion rules](../guides/conversions.md).
