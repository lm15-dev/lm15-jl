# Run only when documentation execution is explicitly authorized. This script was
# written but not executed in the documentation-writing pass.
using Literate, TOML
include("build_support.jl")
using .DocsBuildSupport
DocsBuildSupport.require_isolation()

root = DocsBuildSupport.ROOT
output = joinpath(@__DIR__, ".science")
isdir(output) && rm(output; recursive=true)
mkpath(output)

# Exercise the same source files and retained assertions, not a second calculation
# maintained only for the website. Rendering below is a separate execution, but the
# final Documenter build consumes its plain Markdown without rerunning science.
module ScientificDocumentationRegressions
include(joinpath(@__DIR__, "..", "examples", "scientific", "runtests.jl"))
end

for name in DocsBuildSupport.SCIENCE_TUTORIALS
    source = joinpath(root, "examples", "scientific", name * ".jl")
    page = Literate.markdown(source, output;
        flavor=Literate.CommonMarkFlavor(), execute=true, credit=false)
    DocsBuildSupport.add_download_link(page, name * ".jl")
end

using LM15, DataFrames, Tables, Unitful, SciMLBase, OrdinaryDiffEqTsit5
packages = Dict(string(nameof(m)) => string(Base.pkgversion(m))
    for m in (LM15, DataFrames, Tables, Unitful, SciMLBase, OrdinaryDiffEqTsit5, Literate))
DocsBuildSupport.write_receipt(joinpath(output, "receipt.toml");
    stage="scientific", packages=packages)
