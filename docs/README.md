# LM15.jl documentation sources

The complete manual is under [`src/`](src/index.md). Tutorials are authored as
annotated Julia files under `examples/tutorials/` and `examples/scientific/`.
Source help is attached from `src/docstrings/` after generated bindings exist.

**Written only:** no tests, example execution, formatter, link checker, site build
or deployment was run in this documentation pass. The new source-help attachment
also needs a future package-load check. Existing verification logs remain evidence
for their original source, not this revision.

## Read without building

- [Start](src/start/installation.md), [first request](src/start/first-request.md)
- [Tutorials](src/tutorials/no-key.md)
- [How LM15 works](src/guides/model.md)
- [Reference groups](src/reference/index.md)
- [Status](src/project/status.md), [contributing](src/project/contributing.md)
- [Build details](src/project/documentation.md), [publishing](src/project/publishing.md)

The source reference pages contain Documenter `@docs` blocks; their full help is in
Julia source and will be rendered by Documenter. Tutorial landing pages link to the
complete readable `.jl` lesson; the build renders that same lesson in place.

## Build later, only when execution is authorized

From the Julia repository root, install dependencies first. This setup downloads
packages but should never receive provider credentials:

```sh
julia --startup-file=no --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --startup-file=no --project=examples/scientific -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

Then, on a Linux host supporting network namespaces:

```sh
bash docs/build.sh
```

This **executes** documentation examples and scientific assertions. It requires
Julia 1.12, `unshare` and `ip`, uses a temporary home and only loopback networking,
and refuses an unrestricted fallback. A credential-environment guard adds protection
against accidental use; it is not itself a security sandbox. Other platforms need
an equivalent isolated container/VM before setting the internal execution flag.

The scientific stage writes Markdown and a source receipt to `.science/`. The final
stage rejects stale output, renders core examples once, checks public help/reference
coverage and builds HTML into `build/`. Live scripts are copied for download, never
included. Source/edit links point back to tracked files, not staging directories.

Generated files, documentation manifests and preview output are ignored by Git.
To preview with an optional local Python installation:

```sh
python3 -m http.server 8000 --bind 127.0.0.1 --directory docs/build
```

Use HTTP rather than opening pretty URLs as local files.

## Publishing remains explicit

`.github/workflows/documentation.yml` is manual-only and uploads a preview artifact;
it does not deploy. `deploy.jl` is a separate gated entry point for reviewed project
pages. The shared website can instead consume the same reviewed artifact. No final
hosting path, registry release or stable documentation version is presumed here.

See [publishing](src/project/publishing.md). Preserve the draft banner until the
first execution and human review provide evidence for changing its wording.

## Design record

[Research](documentation-research.md) explains the Julia documentation patterns.
[The plan](documentation-plan.md) retains the intended milestones; written sources
do not mean its execution and publication acceptance criteria have been met.
