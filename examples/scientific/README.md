# Scientific workflows in ordinary Julia files

These reader-facing sources use real packages and tiny synthetic data. They do local
computation, not provider inference:

- [`tables.jl`](tables.jl): DataFrames summaries, selected rows and explicit missing-value policy.
- [`units.jl`](units.jl): checked Unitful arguments and results that retain their units.
- [`simulation.jl`](simulation.jl): a bounded SciML/Tsit5 solve and a concise result record.

`runtests.jl` includes these same operation definitions and retains the prior
assertions, including provider request-building and both extension load orders.
**The extraction and new tutorial demonstrations were written without running them.**
Earlier successful checks belong to the previous source snapshot, not this refactor.

## Run later when authorized

From this directory, install dependencies first (downloads packages, not provider data):

```sh
julia --project=. -e 'using Pkg; Pkg.develop(path="../.."); Pkg.instantiate()'
```

Run one lesson or the assertions:

```sh
julia --startup-file=no --project=. tables.jl
julia --startup-file=no --project=. units.jl
julia --startup-file=no --project=. simulation.jl
julia --startup-file=no --project=. runtests.jl
```

This environment targets Julia 1.12 and has separate dependency choices from the
LM15 library. Literate is a tutorial-rendering dependency here, not a runtime
requirement for applications using LM15.

The documentation build renders these sources in an isolated scientific stage,
records their source/dependency context, and includes ordinary Markdown in the final
site without rerunning the calculations there. See [`docs/README.md`](../../docs/README.md).

Large datasets, solver objects and private identifiers should stay local unless your
application explicitly decides otherwise. A local tool result can be sent later;
that later request is a separate operation with its own privacy and cost implications.
