# Work on LM15.jl

Keep changes scoped to the Julia repository unless the task explicitly calls for
contract or shared-website changes. Read the pinned contract's authority rules before
changing mappings. The commands below are the ones the 1.0.0 release was checked with.

## Environments

From the repository root, the ordinary package environment is selected by
`julia --project=.`. The documentation environment is `docs/`; scientific tutorials
use `examples/scientific/`. Documentation and scientific dependencies do not become
mandatory runtime dependencies of LM15.

Install dependencies before entering a no-network execution environment:

```sh
julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --startup-file=no --project=examples/scientific -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

These commands can download packages and update development manifests. Review those
changes. Restart Julia after editing source unless using an appropriate development
reload workflow.

## Run checks only when authorized

```sh
julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()'
julia --startup-file=no --project=examples/scientific examples/scientific/runtests.jl
python3 tools/check_contract.py --direction all
```

Package tests use synthetic credentials and local fixture servers. The conformance
launcher requires the sibling `lm15-contract` checkout and its pin. Do not supply
real keys or real login stores to fixture runs. A network namespace with only
loopback enabled provides stronger isolation than merely choosing a fake key.

`tools/probe_requests.py` records additional comparisons with the reference and can
exit nonzero for known differences. Read the report; do not erase differences to
obtain a green count.

## Formatting and documentation

JuliaFormatter uses the development-only environment under `tools/format`. Formatting
is separate from writing and was not run in this pass. The documentation workflow
is described on [the next page](documentation.md).

Source help for generated types and provider factories lives in `src/docstrings/`.
It is included after the bindings exist, without altering their constructors. This
keeps the large documentation-only change reviewable, but requires checking help
alongside the implementation when behavior changes. Existing method-specific help
remains next to its methods. Do not add generic empty help just to satisfy coverage.

## Report problems safely

Include the package/source version, Julia version, operation, provider identity and
a minimal synthetic example. Remove real prompts, credentials, headers, token bodies
and personal files. A normalized error code and reviewed request ID are usually
more useful than a raw HTTP dump. Include whether the issue is local, a fixture
failure, or an observed live-provider failure; do not blur those evidence levels.
