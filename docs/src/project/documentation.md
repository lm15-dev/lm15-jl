# Write and build the documentation

This page describes the authored workflow, **not an executed build result**.
The new sources, checker and workflow still need their first authorized run.

## One owner for each kind of information

- Public API facts belong in source docstrings; reference pages include them with
  `@docs`. Explain special overloads, optional values, effects and ownership.
- Complete tutorials live in annotated `.jl` files. Literate renders their narrative
  and code. The scientific assertion runner includes those same source files.
- Task guides are ordinary Markdown with clearly classified snippets. A live
  configuration block is not secretly an executed doctest.
- Verification records remain tied to a source digest and actual commands. Never
  turn an expected result in a manuscript into a claim that it was observed.

## Execution stages

1. Install dependencies with network access, as described in [contributing](contributing.md).
2. `docs/build.sh` starts a Linux network namespace with only loopback enabled and a
   temporary home directory. It does not fall back to unrestricted execution if
   namespace creation fails.
3. `docs/scientific.jl` runs the retained scientific assertions and renders the
   scientific sources to ordinary Markdown in `docs/.science/`. It writes a source
   digest and dependency-version receipt.
4. `docs/make.jl` requires that matching receipt, stages the manual, generates core
   tutorial `@example` blocks, checks public help/reference coverage and runs
   Documenter with strict checks. It does not rerun the rendered scientific pages.
5. The final HTML and receipt go to `docs/build/`. Nothing publishes automatically.

Core examples run once in Documenter's generator. Science calculations run for the
assertion suite and again for rendering; the final HTML stage does not execute them
again. This extra local computation keeps checks and displayed outputs connected
without bringing the solver stack into the basic documentation environment.

## Future build commands

From the repository root, after an explicit decision to execute the examples:

```sh
bash docs/build.sh
```

The wrapper is Linux-only and requires `unshare` and `ip`. On another platform use
an equivalent isolated container/VM; do not merely set `LM15_DOCS_ISOLATED=1` and
call that isolation. The environment flag is only an accidental-use guard.

Serve the built directory with a local static HTTP server. For example, if Python
is available:

```sh
python3 -m http.server 8000 --bind 127.0.0.1 --directory docs/build
```

Documenter uses the same pretty URLs locally and when hosted. Opening a directory
URL directly as a local file is not an equivalent preview.

## Strict checks

`check_public_help` checks exported names and explicit reference entries, allowing
only the two documented aliases to reuse their targets. Documenter's
`checkdocs=:exports` then checks inclusion of existing exported docstrings;
`doctest=true` checks runnable examples. They are complementary checks.

Do not turn off doctests, suppress all warnings or reclassify a broken executable
block as illustrative code just to publish. Review expected-output changes against
intended behavior. Never alter contract fixtures to agree with an implementation bug.

External links are deliberately separate from the offline example build. Review
links to moving providers/packages and record transient failures rather than using
them as permission to weaken internal-link or example checks.

## Live examples

Only downloadable files such as `live-tools.jl` and `live-stream.jl` use actual model
calls. The generator copies them as text and never includes them. They require
`LM15_RUN_LIVE_EXAMPLES=yes` and a separately supplied account key when a user
explicitly runs them. That flag is consent to run an example, not a total-spend cap
or a replacement for tool approval.

## Review before publication

Check the generated site on a narrow screen and with keyboard navigation. Try code
copying, source/download links, Julia help and search. Confirm actual hosting-prefix
links, version labels and a readable status banner. None of those behaviors was
tested during this writing pass. See [publishing](publishing.md).
