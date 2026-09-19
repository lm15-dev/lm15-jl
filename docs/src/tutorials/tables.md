# Summarize a table without uploading it all

Keep a DataFrame local. Offer a bounded summary operation or an explicit selection
of rows, encode only the result, and choose what missing values mean rather than
silently replacing them.

[Read the complete annotated Julia lesson](../../../examples/scientific/tables.jl).
It uses real DataFrames, Tables and Statistics packages with tiny synthetic data;
it does not contact a provider. The existing scientific assertion runner now uses
these same operation definitions, but this extraction has not been executed yet.

The docs pipeline renders this lesson in its separate scientific environment and
includes its captured Markdown without rerunning it in the final site build.
See [documentation work](../project/documentation.md).

Next: [physical units](units.md) and [data privacy](../guides/privacy.md).
