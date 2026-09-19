# Julia documentation study

Reviewed 2026-09-13. This is a comparison of useful documentation patterns, not a
ranking of package quality. I inspected published pages and selected build sources;
I did not run these libraries' tutorials or perform a visual/accessibility audit.

## What to borrow—and what not to copy

| Library | Pages inspected | Useful pattern | Application to LM15 |
|---|---|---|---|
| **JuMP** | [Introduction](https://jump.dev/JuMP.jl/stable/), [getting started](https://jump.dev/JuMP.jl/stable/tutorials/getting_started/getting_started_with_JuMP/), [contributing](https://jump.dev/JuMP.jl/stable/developers/contributing/) | Clearly separates worked tutorials, task-oriented manual, function reference, background and developer material. The tutorial states learning goals, shows a complete solution, then explains the steps. It checks solver status before reading the result. | Use this as the main organizational model. Show a complete request/result workflow and check the response before using it. Keep implementation details off the beginner path. Do not copy the length of its introductory tutorial or its full solver logs. |
| **DataFrames** | [Getting started](https://dataframes.juliadata.org/stable/man/getting_started/), [missing data](https://dataframes.juliadata.org/stable/man/missing/), [build source](https://github.com/JuliaData/DataFrames.jl/blob/main/docs/make.jl) | Concrete inputs and displayed results make copying, mutation and missing values observable. Practical topics have their own pages. The build enables doctests and separates the guide from the API. | Give `missing`, precision, units and shallow copies small worked examples, including a failure and its fix. Do not turn the first LM15 lesson into an exhaustive list of constructor spellings. |
| **Makie** | [Getting started](https://docs.makie.org/stable/tutorials/getting-started) | Shows the intended result first. Builds one example in stages, explains Julia environments, chooses a broadly usable CPU backend, and finishes by saving a useful artifact. | Start each tutorial with its outcome. Grow one request into a usable workflow rather than showing disconnected fragments. Explain setup clearly, but link longer Julia setup instructions instead of repeating them on every page. |
| **Flux** | [One-minute example](https://fluxml.ai/Flux.jl/stable/guide/models/quickstart/), [home](https://fluxml.ai/Flux.jl/stable/) | Offers a compact complete example for experienced users and points beginners to a slower introduction. Explains how ordinary Julia functions, callable objects and explicit loops fit together. | Provide a short quick start and a separate guided tutorial. Keep tool execution and the application loop visible. Do not inherit GPU dependencies or promise a one-minute first run when installation and compilation may take longer. |
| **SciML / DifferentialEquations** | [Overview, migration notice and installation guidance](https://docs.sciml.ai/DiffEqDocs/stable/) | Makes the ecosystem's scope and version changes explicit; points users needing fewer dependencies toward smaller packages. | Organize scientific examples around the actual calculation, use the narrow solver package already tested here, and distinguish LM15 support from a provider/model/account's support. Do not reproduce the ecosystem-sized navigation tree. |
| **Tables** | [Implementing the interface](https://tables.juliadata.org/stable/implementing-the-interface/) | Lists required methods, optional methods, defaults and the behavior an implementation must satisfy. Explains that implementing an interface need not mean subtyping its abstract type. | Document `tool_schema`, `tool_decode`, `tool_value` and `open_response` as explicit extension agreements: required methods, expected return values, failure behavior and a complete example. Do not encourage users to invent new canonical wire variants. |
| **HTTP** | [Current introduction](https://juliaweb.github.io/HTTP.jl/stable/), [current client guide](https://juliaweb.github.io/HTTP.jl/stable/guides/client/), [1.11.0 documentation](https://juliaweb.github.io/HTTP.jl/v1.11.0/) | Separates simple calls from streaming, connection ownership and operational controls. Current examples use a local server, making the behavior reproducible. | Test network-shaped examples against local fixtures, explain who closes streams, and document when an operation contacts a provider. **LM15 depends on HTTP 1.x; current HTTP stable is 2.0. Do not copy its new timeout names or return behavior into LM15 examples.** |

The common lesson is not “use the same theme.” It is **give readers a clear task,
a working example, an observable result, and a route to deeper details**.

## Documentation machinery worth adopting

- [Julia's documentation conventions](https://docs.julialang.org/en/v1/manual/documentation/):
  useful help in the REPL, editor and notebook; signature first, a short summary,
  runnable examples and links to related functions. Document special methods when
  their behavior differs. Keep extension-author details separate from ordinary use.
- [Documenter's package guide](https://documenter.juliadocs.org/stable/man/guide/):
  combine Markdown and source docstrings, with explicit navigation and checked
  cross-references. Use a separate documentation environment; do not require the
  Julia 1.12 workspace feature from users of a library targeting Julia 1.10.
- [Documenter's doctests](https://documenter.juliadocs.org/stable/man/doctests/):
  execute examples and compare their output. Named blocks share state within a
  page. Failures should fail the build; never suppress mismatches merely to publish.
- [Documenter's build checks](https://documenter.juliadocs.org/stable/lib/public/#Documenter.makedocs):
  `checkdocs=:exports` checks that **existing docstrings** appear in the manual.
  It does not prove that every export has a docstring. LM15 needs a separate
  missing-docstring check as well as a review of important method variants.
- [Literate's output formats](https://fredrikekre.github.io/Literate.jl/v2/outputformats/):
  generate tutorial pages from runnable `.jl` sources. Its Documenter output uses
  `@example` blocks; let Documenter execute those once rather than executing the
  same page in both generators. JuMP's [build script](https://github.com/jump-dev/JuMP.jl/blob/master/docs/make.jl)
  provides a real example of a Literate-backed documentation pipeline.
- [Documenter's hosting guide](https://documenter.juliadocs.org/stable/man/hosting/):
  distinguish released documentation from development documentation. Publication
  credentials belong only in trusted deployment jobs, not example execution or
  pull-request code. Generated output does not belong on the source branch.

These sources move as packages release new versions. They inform presentation and
maintenance; LM15 behavior must come from its own code, tests and pinned contract.
The legacy SciML ODE URL returned an HTML redirect, and its destination was not
retrievable during this study. It was not treated as a reviewed tutorial.

## LM15's starting point

Audited commit: `d6758fdffb0256b95735f5ec8f727cb35fac56da`.

- `README.md`: 354 lines mixing introduction, verification evidence, installation,
  examples, security behavior, limitations and contributor instructions.
- `docs/function-tools.md`: a substantial, useful guide, but it combines a first
  tutorial, detailed conversion rules, scientific integrations and extension work.
- `examples/scientific/runtests.jl`: real executable workflows, currently presented
  as tests rather than a set of reader-facing tutorials.
- No documentation project, `make.jl`, generated reference site or docs build job.
- **312 exported bindings; 290 have no docstring according to `Docs.hasdoc`.**
  Examples include `complete`, `Response`, `Config`, `Usage`, `FunctionTool`,
  `LMRouter`, `OpenAILM`, `wait!`, `parse_json` and `estimate`.
- A positive docstring check is only a presence check. It does not establish
  completeness, correct examples or coverage of all method variants. Some exported
  Base functions also inherit documentation that does not explain LM15 behavior.
- The tool example builds a follow-up request but does not show the model's final
  answer. A full tutorial should close that loop while keeping execution explicit.

The help audit was a local inspection, not a test-suite run:

```julia
using LM15, REPL
public_names = filter(!=(:LM15), names(LM15))
missing_docs = filter(n -> !Docs.hasdoc(LM15, n), public_names)
length(public_names), length(missing_docs) # (312, 290), on Julia 1.12.5
```

The sibling `website/README.md` and `website/astro.config.mjs` describe a shared
LM15 documentation hub using Astro/Starlight, with migration still pending. That
is a reason to keep Julia sources in this repository and plan a clean publishing
handoff—not create a second independently maintained copy of the same manual.

See [the documentation plan](documentation-plan.md) for the proposed work.
