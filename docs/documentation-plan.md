# LM15.jl documentation plan

**Status: the manual, source help, tutorial sources and build setup are written.**
They have not been tested, built or deployed; the execution/publication acceptance
criteria below remain pending. See [the authoring entry point](README.md).
Based on the [Julia documentation study](documentation-research.md), 2026-09-13,
and LM15 commit `d6758fd`.

## 1. Direction

Build a **task-first Julia manual**, not a larger README or a directory of function
names. Borrow JuMP's separation of tutorials and reference, Makie's step-by-step
teaching, DataFrames' precise examples, and Tables' clear extension agreements.

Use **Documenter.jl** to build the manual and source-code help together. Use
**Literate.jl** for substantial runnable tutorials, not for every paragraph.

The reader's first experience should be:

1. Understand what LM15 does and what stays under their control.
2. Install it using a command verified for the version being documented.
3. Get one useful result.
4. Find the next task without searching a long list of implementation details.

A scientific user should quickly see the central idea: **Julia does the calculation;
the model can request it and explain the result. Large datasets and solver objects
can stay local.**

## 2. The immediate gaps

- The README leads with extensive implementation and verification information
  before the first example. Keep a visible status summary, but move the detailed
  evidence and limitations to a dedicated page.
- Most public names lack built-in help: the audit found **290 of 312** without a
  docstring. A generated website alone will not fix that.
- Examples often rely on variables introduced elsewhere. Each tutorial needs a
  complete runnable source, including imports and setup.
- The tool tutorial needs to finish the story: receive a call, check it, approve
  it, execute it, send its result, and obtain the final response.
- Scientific examples are currently embedded in a test file. They need a readable
  narrative without losing their existing assertions.

## 3. Proposed navigation

These are content groups, not a requirement to publish dozens of empty pages.

| Section | Reader's question | Planned content |
|---|---|---|
| **Start here** | “Can I use this, and how do I begin?” | What LM15 is; installation; one real request; a clearly labeled no-key path using a local function tool. |
| **Tutorials** | “Walk me through a complete task.” | Continue a conversation; stream an answer; complete an approved tool round-trip; work with a DataFrame; preserve physical units; expose a bounded SciML calculation. |
| **How-to guides** | “How do I do this particular thing?” | Choose a provider/model; configure credentials; send images/documents/audio; request JSON; configure generation and inspect usage; handle errors and timeouts; convert unusual data; use files/caches/batches/generation/live sessions. |
| **Reference** | “What exactly does this function accept and return?” | Grouped types, functions, defaults, errors, compatibility rules and extension methods, drawn from source docstrings. |
| **Project and development** | “What is verified, and how can I contribute?” | Support/status; releases and migration; development setup; tests and docs builds; pinned contract and verification evidence. |

A short **How LM15 works** guide explains the shared model once:

- Messages and configuration form a `Request`.
- A provider returns a `Response`, or a stream that can be assembled into one.
- A tool call is a request to your application—not permission to execute code.
- Your application owns approval, execution, follow-up requests and stopping rules.

Link to this explanation when needed. Do not make readers learn every canonical
part type or provider protocol before making a request.

### Reference groups

Organize by purpose, with a complete searchable index as a secondary entry point:

1. **Requests and results:** messages, parts, `Request`, `Response`, `Config`,
   `Usage`, accessors and JSON readers/writers.
2. **Function tools:** `@tool`, `ToolBinding`, `FunctionTool`, argument checking,
   execution, data conversion and optional scientific extensions.
3. **Providers and credentials:** direct clients, router, catalogs, compatibility,
   credentials and authentication diagnostics.
4. **Streams and live sessions:** iteration, accumulation, closing, partial answers
   and ownership of connections.
5. **Resources and jobs:** files, caches, batch/media jobs, waiting and cancellation.
6. **Advanced interfaces:** custom transports and codecs, wire builders/parsers,
   ingestion and authentication helpers.

All exports remain findable. Mark specialist APIs as advanced rather than removing
exports or hiding undocumented functions to make a coverage check pass.

## 4. The first tutorials

| Tutorial | Useful result | Checks and limits to teach |
|---|---|---|
| **Make your first request** | Read text and usage from one response. | Key setup, selected provider/model, possible charges, and variable model output. |
| **Keep the conversation going** | Ask a follow-up using the earlier answer. | Explicit history and copy-and-change constructors; no hidden conversation state. |
| **Read an answer as it arrives** | Print fragments and obtain the assembled response. | Use the `do` form, explain its return value, and close on every exit. |
| **Let a model use a Julia function** | Complete one approved tool exchange. | Typed interface, checked arguments, explicit execution, no automatic retries or unlimited loops. |
| **Ask about a table without uploading it all** | Return a selected summary or bounded rows. | Explicit columns, row limits, missing-value policy and what leaves the machine. |
| **Keep the units** | Calculate a speed using Unitful quantities. | Expected units, output units and refusal of mismatched inputs. |
| **Explain a simulation result** | Run a small SciML solve and return a checked summary. | Bounded inputs, solver success, iteration limit, numerical checks; keep the solution object local. |

Give new users one recommended spelling first. Put alternative constructors,
manual stream ownership and provider-specific options in the linked reference.

For every tutorial:

- Start with its goal, prerequisites, packages and whether it makes paid requests.
- Show the finished result, then a complete short script, then explain the steps.
- Use a small, deterministic dataset; no unannounced downloads or hidden setup.
- Show one likely failure and a useful fix.
- End with the next task and relevant function links.
- Offer the exact `.jl` source used to produce the page.

Live examples show **illustrative output**, not a promised exact answer. Offline
examples label fabricated provider responses explicitly; a fixture is not evidence
of a successful live call. A no-key tutorial must still do something useful, such
as executing a real local calculation, rather than pretending to contact a model.

## 5. Built-in help is part of the product

First document the main entry points: `complete`, `Request`, `Response`, `Config`,
`Usage`, `LMRouter`, direct provider constructors, `FunctionTool`, `ToolBinding`,
`stream`, `response`, `text`, tool execution and result helpers. Then cover the
remaining public families without leaving an unfinished “complete reference.”

Each useful docstring needs:

- A readable Julia signature and a one-sentence purpose.
- Inputs/defaults and the return value, including when `nothing` is possible.
- A short runnable example and related functions.
- Relevant errors and side effects—not a boilerplate warning on every accessor.

For operations involving I/O, also explain:

- Whether it contacts a provider, reads/writes a local file, or may do so through
  credential resolution. `build_request` is not automatically safe to call offline
  with arbitrary credential providers.
- Whether it can incur charges or repeat an action.
- Who closes a stream or session, and whether timeout/cancellation stops remote work.

Cover distinct method behavior explicitly. A docstring for scoped `stream` is not
by itself documentation of every `stream` form. Test that help attaches correctly
to macro-generated types and that `?complete`, `?Config` and `?@tool` are useful.

Extension pages follow Tables' example: list required methods, optional methods,
return types, failure rules and a complete application-owned type. Explain why
schema generation and decoding must agree, without assuming readers know compiler
internals or changing how unrelated packages behave.

## 6. Authoring and build layout

Proposed files, to be created during implementation:

```text
docs/
  Project.toml          # Documentation-only dependencies
  make.jl               # Explicit page order and strict build checks
  src/
    index.md
    start/
    tutorials/          # Handwritten overview + generated tutorial pages
    guides/
    reference/
    project/
    assets/
examples/
  tutorials/            # Reader-facing Literate .jl sources
  scientific/           # Separate scientific environment and runnable sources
```

- Source docstrings own function-level facts; the website includes them with
  `@docs`. Do not maintain competing copies of defaults in Markdown tables.
- Tutorial `.jl` files own runnable workflows. Refactor the existing scientific
  test runner to exercise those sources and retain its assertions, rather than
  maintaining a second set of example calculations.
- Plain Markdown remains appropriate for explanation and short task guides.
- Keep Documenter and Literate out of runtime dependencies. Keep the heavy
  scientific environment separate from the basic docs environment. Core tutorial
  pages use Documenter's executable blocks. Scientific pages are executed with
  Literate in the scientific environment and supplied as ordinary Markdown with
  captured results/assets; the final site build must not execute them a second time.
  Check their source revision before including them, so cached outputs cannot hide
  an untested example change.
- Use the conventional separate docs project; do not require Julia 1.12 workspaces
  from library users. Build the site on a recorded Julia version and check core
  example compatibility separately on the supported runtime versions.
- Do not commit generated HTML or generated notebooks to `main`.

Preserve current README and `docs/function-tools.md` links until replacements
exist. Split the current tool guide across tutorial, conversion reference and
extension guide; leave a useful forwarding page once the new destinations work.

## 7. Keep examples honest

Three different checks serve three different purposes:

1. **Core documentation checks:** execute deterministic examples with Documenter,
   fail on broken internal links and incorrect output, and check docstring inclusion.
   Also inventory missing exported docstrings: `checkdocs=:exports` alone is not
   enough. During migration, use an explicit remaining-work list, not blanket
   `warnonly=true` or `doctest=false`.
2. **Scientific checks:** run real DataFrames, Tables, Unitful and SciML examples in
   their own environment. Verify numerical results with appropriate tolerances,
   inspect representations, and check optional-extension loading. Generated pages
   must come from those tested sources.
3. **Live checks:** separately authorized, budgeted and dated. Never run them as a
   side effect of building documentation or use their changing prose as a doctest.

Install dependencies before isolating example execution. Run executable docs with
no real credentials or login stores and no outbound provider access; permit
loopback only where a local fixture server is needed. Use the existing transport
extension point, not changes to production behavior, for fixture-backed examples.

Check external links separately from offline example execution. Internal link and
example failures block publication; transient external-site failures get an explicit
report instead of making unrelated documentation fixes impossible.

Every code block must be classified: executed example, tested source excerpt,
authorized live example, or clearly marked illustrative configuration. Do not
silently downgrade broken runnable examples to untested code fences. Review any
updated expected output against intended behavior; leave contract fixtures intact.

## 8. Publishing and the shared website

The Julia repository owns this manual, its reference and its runnable examples.
The existing Astro/Starlight website remains the shared LM15 entry point.

**Recommendation:** have Documenter render the Julia documentation and publish it
as a versioned static artifact. The shared website can link to that artifact or
host it under an agreed Julia path. Do not teach Astro to interpret Julia docstrings
or maintain a hand-copied second manual there.

The first delivery is a local/CI preview. Choose the final URL and artifact handoff
with the website migration before enabling publication. Initially this may mean
separate Julia navigation/search; that is preferable to building a custom rendering
bridge before the writing and examples are good. No website workflow changes are
part of this planning task.

Use `dev` for unreleased source and `stable` only for an actual release. Verify
registry availability before recommending `Pkg.add("LM15")`; do not invent an
available package version or tag. Link canonical rules to the package's
`CONTRACT_PIN`, not to a moving contract branch.

Keep publication credentials out of example jobs and untrusted pull requests.
Use the existing LM15 branding with minimal styling. No credential-entry widget,
live “run this” button, analytics, or elaborate custom theme is needed for launch.

## 9. Delivery order and acceptance

| Stage | Deliverable | Done when |
|---|---|---|
| **1. Foundations** | Docs project, navigation, a reviewed reference grouping, core docstrings and an export-coverage backlog. | The site builds locally; core REPL help works; example checks can run without provider access. |
| **2. First complete journey** | Installation, first request, conversation, streaming and approved tools. Shorten the README into a useful entry point. | A reader can follow a complete workflow from a clean environment; every dependency and paid action is visible. |
| **3. Scientific workflows and full reference** | Table, unit and simulation tutorials; advanced task guides; all public API families. | Real scientific examples pass; every export is documented or explicitly linked to its documented alias; important overloads and effects are reviewed. |
| **4. Publication and upkeep** | Versioned preview/release artifacts, website handoff, support page and contributor instructions. | Links work at the real hosting prefix; releases identify package/Julia versions; deployment uses only verified artifacts. |

Before calling the result ready, ask a new reader to:

- Find installation and run the no-key example without guessing missing setup.
- Make a paid request knowing what is sent and which account is used.
- Stop streaming without leaking a connection.
- Find out why a tool input, missing value or unit was rejected and fix it.
- Look up a function both on the site and through Julia help.
- Tell the difference between implemented support, locally tested behavior and
  behavior verified against a live provider.

Do not bury the current limitations, but do not require reading the full verification
report before using the package. A brief visible status label links to evidence and
known gaps. Check keyboard navigation, narrow screens, code copying, search and
source links before publishing; this study did not test those visual behaviors.
