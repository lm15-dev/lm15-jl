# Install and choose an environment

Use a project environment so your experiment does not change unrelated Julia work.
These are installation commands, not documentation-build examples: installing
packages needs network access. They have not been executed in this writing pass.

## From a checkout

In a Julia session opened in your own application directory:

```julia
using Pkg
Pkg.activate(".")
Pkg.develop(path="/absolute/path/to/lm15-jl")
using LM15
```

Replace the path with your checkout. `Pkg.develop` keeps that checkout live: edits
there affect the next fresh Julia session. Restart Julia after changing source.

## From the repository

For an application that does not need an editable checkout:

```julia
using Pkg
Pkg.activate(".")
Pkg.add(url="https://github.com/lm15-dev/lm15-jl.git")
using LM15
```

The repository command selects its current source; it is not a promise of a
registered or released version. Review and commit your application's `Project.toml`
and `Manifest.toml` for reproducibility. Use an explicit reviewed `rev` if your
project needs a fixed source revision. Do not assume `Pkg.add("LM15")` is available
until registry publication has been confirmed.

## Optional scientific packages

LM15 alone does not install a data-frame package, Unitful or a solver. Add only what
you need to your application's environment:

```julia
using Pkg
Pkg.add(["DataFrames", "Tables", "Unitful"])
```

Loading Tables or Unitful enables the corresponding LM15 extension. A DataFrame
still needs the deliberate `table_content` representation; it is not uploaded
simply because a package was loaded.

The [scientific example environment](../project/contributing.md) declares the
solver dependencies used by the tutorials. Those examples target Julia 1.12.

## Notebook and editor use

Make the notebook kernel or editor use the environment in which LM15 was installed.
In IJulia, check `Base.active_project()` if `using LM15` cannot find the package.
Pluto manages notebook environments separately; follow its package-management
workflow instead of assuming a terminal environment is shared.

Plain source files remain the common example format. No notebook runtime is required
for the library. New notebook/editor integrations in this manual have not been tested.

## Next

[Make a provider request](first-request.md), or
[try a local function tool without credentials](../tutorials/no-key.md).
