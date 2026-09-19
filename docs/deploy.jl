# Explicit publication only, after a reviewed isolated build. This file is not
# included by make.jl and is not called by the preview workflow.
using Documenter
include("build_support.jl")
using .DocsBuildSupport

get(ENV, "LM15_DOCS_PUBLISH", "no") == "yes" || error("Publication is not authorized")
get(ENV, "LM15_DOCS_REVIEWED", "no") == "yes" ||
    error("Review the build, examples and hosting configuration before publishing")
DocsBuildSupport.require_matching_receipt(
    joinpath(@__DIR__, "build", "build-receipt.toml"), "manual")
isfile(joinpath(@__DIR__, "build", "index.html")) || error("No built manual to publish")

# Configure a trusted CI deployment environment before enabling this. Do not run
# publication with a token exposed to pull-request code. Shared-site hosting can
# consume the same artifact instead of using this project-pages deployment path.
deploydocs(root=@__DIR__, repo="github.com/lm15-dev/lm15-jl.git",
    devbranch="main", devurl="dev", target="build", push_preview=false)
