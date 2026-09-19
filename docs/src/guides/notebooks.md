# Use LM15 in a notebook or scientific project

LM15 uses ordinary Julia values and functions. A notebook is a useful place to
inspect a request, approve a proposed operation, and plot a local result—but it
also makes it easy to reveal data or accidentally repeat work.

## Keep the environment explicit

Use the notebook's package environment. IJulia kernels and a terminal session may
have different active projects; inspect `Base.active_project()` when imports fail.
Pluto has its own dependency management. Follow the notebook's workflow rather than
assuming a terminal installation is automatically visible.

The `.jl` tutorial sources are the primary examples. Running them does not require
a notebook. The new notebook guidance is written guidance, not a tested IJulia or
Pluto integration claim.

## Inspect before executing

Display `Request`, `Response`, `FunctionTool(binding)` and `tool_arguments(binding,
call)` to understand what would happen. Run `execute_tool` only after approval.
A cell re-run can submit a second provider request or execute a function again;
there is no automatic deduplication or undo.

Keep provider submissions in clearly labeled cells. Do not put a paid call in a
reactive definition that will rerun whenever unrelated input changes. Saving a
notebook may save displayed outputs: review them before sharing it.

## Keep the calculation local

Use [table summaries](../tutorials/tables.md), [Unitful quantities](../tutorials/units.md)
and [bounded simulations](../tutorials/simulation.md). Return small named records
instead of raw solver objects. A model explanation is not a substitute for numerical
checks, units or a successful solver return code.

Encode plots deliberately as small image files or bytes before creating ImageParts.
Do not assume every array is an image or that the provider should receive the full
dataset. Avoid putting model calls inside a numerical solver's inner loop or treating
a remote response as an automatically differentiable scientific operation.

See [privacy](privacy.md), [ownership](lifecycle.md), and [conversion rules](conversions.md).
