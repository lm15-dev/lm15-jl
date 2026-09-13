# Function tools and scientific integrations: verification

Verified on **Julia 1.12.5, Linux x86-64**. All test runs were in network
namespaces; only loopback was enabled for local HTTP/WebSocket tests. Fresh
HOME directories kept real login stores out. No provider calls were made.

| Check | Result |
|---|---|
| Normal `Pkg.test(; allow_reresolve=false)` | **874 assertions, 34 test sets, all pass** |
| Real scientific-package workflows | **48 assertions, 4 test sets, all pass** |
| Unchanged shared contract, all 16 directions | **1,380 pass, 0 fail, 2 existing skips** |
| Method ambiguities in LM15 and the two extension modules | **0** |
| Optional packages loaded before and after LM15 | **Both orders pass** |

The existing skips are `openai.computer_use`: no canonical request and no response
golden. No fixture, expected result, comparator or contract file was changed.

## What was checked

- Defining/describing/serializing a tool does not execute its body or defaults.
- JSON integers become checked Julia integers; booleans, fractional integers,
  overflow and precision-losing conversions are refused before execution.
- Required versus nullable arguments, keyword defaults and trailing positional
  defaults; explicit method selection, closures and callable structs.
- Caller-owned type extensions, including a broken nested decoder that must fail
  rather than have its wrong result silently converted.
- Scalar/record/media outputs, exact Float32 promotion before JSON encoding,
  cycles, unsupported values and offset axes. Execution is never retried after
  an error, including an output conversion error following a side effect.
- Actual DataFrames summaries and selected table rows, explicit missing-value
  policy, and refusal instead of silent row truncation.
- Actual Unitful input/output, quantities nested in tables/arrays, and rejection
  of wrong or executable-looking unit strings.
- An actual SciML exponential-decay solve using Tsit5, checked against its analytic
  result. Only a small result summary crosses the model boundary.
- Building tool declarations and result replay for four provider protocols without
  sending the requests.

## Reproduce

From `lm15-jl/` with dependencies installed:

```sh
julia --startup-file=no --project=. -e 'using Pkg; Pkg.test(; allow_reresolve=false)'
python3 tools/check_contract.py --direction all --report-dir /tmp/lm15-contract-check

cd examples/scientific
julia --project=. -e 'using Pkg; Pkg.develop(path="../.."); Pkg.instantiate()'
julia --startup-file=no --project=. runtests.jl
```

The install step needs network access for package downloads, not provider access.
The recorded native runs additionally used `unshare -rn`, enabled `lo`, set HOME
to a new temporary directory, and selected the isolated dependency depot at
`/tmp/lm15-julia-clean-depot`. The contract launcher supplies its own sandbox.

`summary.json` records the source digest, its recipe, the contract pin and the
scientific package versions. The logs and per-direction JSON reports are retained
beside it. Initial review runs exposed a non-exported SciML helper in the example
and quoting in the reverse-load subprocess test; both were corrected and the full
scientific suite rerun. No assertion was skipped to obtain these results.

## Limits and choices

This is not live-provider schema-acceptance evidence. Julia 1.10, macOS and Windows
were not executed here; the native CI matrix covers those targets when run.

`@tool` captures a chosen typed interface, not every method behind a Julia
function. Descriptions are explicit. Defaults remain Julia expressions and run
only on execution. General schemas still use handwritten FunctionTools. Unknown
scientific representations need a wrapper or a type-specific conversion.

Tables and Unitful use optional extensions, not mandatory dependencies. Table
output is explicitly selected and bounded; units are preserved, never parsed as
code. Provider routing was not rewritten: the new multiple-dispatch interfaces
are for type schemas, argument conversion and result conversion. The canonical
request/response contract is unchanged.
