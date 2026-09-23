# LM15.jl

Call model providers directly from Julia. Build a `Request`, receive a typed
`Response` or stream, and decide explicitly which Julia tools may run.
There is no LM15 relay and no hidden tool-execution loop.

**Documentation draft:** the new manual, source help, build setup and refactored
examples were written without running tests or building the site. Earlier checks
are retained as [version-specific evidence](verification/function-tools/README.md),
not proof of this new revision.

## Start here

- [Install](docs/src/start/installation.md)
- [Make your first provider request](docs/src/start/first-request.md)
- [Try a local tool without a key](examples/tutorials/no-key.jl)
- [Read the manual](docs/src/index.md)
- [Find a function](docs/src/reference/index.md), or use `?complete` and `?FunctionTool` in Julia

From a checkout, in your application's Julia environment:

```julia
using Pkg
Pkg.develop(path="/absolute/path/to/lm15-jl")
using LM15
```

See the installation page for repository installation and version selection.
Registry publication is not assumed.

## One local function tool

```julia
using LM15

square(n::Integer) = big(n)^2
call = tool_call("example-1", "square", Dict("n" => 19))
output = tool_result(call, tool_content(square(call.input["n"])))
println(only(output.content).text) # Expected: 361
```

This call is authored locally; no model was contacted. A real provider can return
that call description, but your application still decides whether to run it.
See the [complete tool exchange](docs/src/tutorials/tools.md), including the
separately authorized live example.

## Scientific workflows

Use real Julia calculations and return deliberate summaries, not arbitrary internal
objects. Tables and Unitful integrations load with those optional packages; neither
is a mandatory LM15 dependency.

- [DataFrames and selected rows](examples/scientific/tables.jl)
- [Physical units](examples/scientific/units.jl)
- [A bounded SciML simulation](examples/scientific/simulation.jl)
- [Conversion rules](docs/src/guides/conversions.md): precision, dimensions and missing values

## Important boundaries

The package targets Julia 1.10+ on native Linux, macOS and Windows. Earlier recorded
execution used Julia 1.12.5 on Linux. There is no supported browser-WASM build.
Provider/model/account support can be narrower than the canonical type vocabulary.

Calls, media, storage and jobs can incur charges. A timeout does not necessarily
stop remote work. Keep keys and private data out of logs; [read the privacy guide](docs/src/guides/privacy.md).
The [status page](docs/src/project/status.md) records platform, protocol and uncommon
credential-chain gaps without hiding them behind a passing fixture count.

## Development

The shared [contract](docs/src/project/contract.md) is pinned by `CONTRACT_PIN`.
[Contributor instructions](docs/src/project/contributing.md) describe the package,
scientific and conformance commands. [Documentation instructions](docs/README.md)
describe the authored, manual-only preview workflow and its isolation requirements.
No new test run, site build or deployment was performed in the writing pass.
