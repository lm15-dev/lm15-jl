using Documenter, Literate, LM15
include("build_support.jl")
using .DocsBuildSupport
DocsBuildSupport.require_isolation()
include("check_help.jl")

root = DocsBuildSupport.ROOT
science = joinpath(@__DIR__, ".science")
DocsBuildSupport.require_matching_receipt(joinpath(science, "receipt.toml"), "scientific")

work = joinpath(@__DIR__, ".work", "src")
isdir(dirname(work)) && rm(dirname(work); recursive=true)
mkpath(dirname(work))
cp(joinpath(@__DIR__, "src"), work)
for (directory, _, files) in walkdir(work)
    for file in files
        endswith(file, ".md") || continue
        path = joinpath(directory, file)
        DocsBuildSupport.set_edit_url(path, joinpath("docs", "src", relpath(path, work)))
    end
end
mkpath(joinpath(work, "downloads"))
mkpath(joinpath(work, "tutorials"))

for name in DocsBuildSupport.CORE_TUTORIALS
    file = joinpath(root, "examples", "tutorials", name * ".jl")
    page = Literate.markdown(file, joinpath(work, "tutorials");
        flavor=Literate.DocumenterFlavor(), execute=false, credit=false)
    DocsBuildSupport.add_download_link(page, name * ".jl")
    cp(file, joinpath(work, "downloads", name * ".jl"); force=true)
end
for name in DocsBuildSupport.SCIENCE_TUTORIALS
    cp(joinpath(root, "examples", "scientific", name * ".jl"),
        joinpath(work, "downloads", name * ".jl"); force=true)
end
for file in readdir(science)
    file == "receipt.toml" && continue
    cp(joinpath(science, file), joinpath(work, "tutorials", file); force=true)
end
for file in DocsBuildSupport.LIVE_SOURCES
    # Downloadable text only. Never include/evaluate a live script here.
    cp(joinpath(root, "examples", "tutorials", file), joinpath(work, "downloads", file); force=true)
end

# Documentation facts link to exactly the contract pin of this source revision.
pin = strip(read(joinpath(root, "CONTRACT_PIN"), String))
for (directory, _, files) in walkdir(work)
    for file in files
        endswith(file, ".md") || continue
        path = joinpath(directory, file)
        write(path, replace(read(path, String), "@CONTRACT_PIN@"=>pin))
    end
end

check_public_help(work)
DocMeta.setdocmeta!(LM15, :DocTestSetup, :(using LM15); recursive=true)

makedocs(
    root=@__DIR__, source=joinpath(".work", "src"), build="build",
    sitename="LM15.jl", modules=[LM15], checkdocs=:exports,
    doctest=true, warnonly=false, linkcheck=false,
    format=Documenter.HTML(prettyurls=true, edit_link="main",
        canonical=get(ENV, "LM15_DOCS_CANONICAL", nothing), assets=["assets/lm15.css"]),
    pages=[
        "Home"=>"index.md",
        "Start here"=>[
            "Installation"=>"start/installation.md",
            "First request"=>"start/first-request.md",
        ],
        "Tutorials"=>[
            "No-key tool"=>"tutorials/no-key.md",
            "Conversation"=>"tutorials/conversation.md",
            "Streaming"=>"tutorials/streaming.md",
            "Approved tools"=>"tutorials/tools.md",
            "Tables"=>"tutorials/tables.md",
            "Physical units"=>"tutorials/units.md",
            "Simulation"=>"tutorials/simulation.md",
        ],
        "How-to guides"=>[
            "How LM15 works"=>"guides/model.md",
            "Providers and models"=>"guides/providers.md",
            "Credentials"=>"guides/credentials.md",
            "Sign-in (subscriptions and saved keys)"=>"guides/sign-in.md",
            "Configuration and usage"=>"guides/configuration.md",
            "Images, audio and documents"=>"guides/content.md",
            "JSON output"=>"guides/json.md",
            "Judgments"=>"guides/judgments.md",
            "Data conversion"=>"guides/conversions.md",
            "Failures and timeouts"=>"guides/errors.md",
            "Ownership"=>"guides/lifecycle.md",
            "Privacy and costs"=>"guides/privacy.md",
            "Files and caching"=>"guides/files-caches.md",
            "Background jobs"=>"guides/jobs.md",
            "Media generation"=>"guides/generation.md",
            "Live sessions"=>"guides/live.md",
            "Custom interfaces"=>"guides/extensions.md",
            "Chat-style migration"=>"guides/migration.md",
            "Notebooks"=>"guides/notebooks.md",
        ],
        "Reference"=>[
            "Index"=>"reference/index.md", "Content"=>"reference/content.md",
            "Tools"=>"reference/tools.md", "Providers"=>"reference/providers.md",
            "Authentication"=>"reference/auth.md", "Managed sign-in"=>"reference/sign-in.md", "Streams and live"=>"reference/streams.md",
            "Resources and jobs"=>"reference/resources.md", "Errors"=>"reference/errors.md",
            "Advanced interfaces"=>"reference/advanced.md",
        ],
        "Project"=>[
            "Status and evidence"=>"project/status.md", "Releases"=>"project/releases.md",
            "Contract"=>"project/contract.md", "Contributing"=>"project/contributing.md",
            "Documentation work"=>"project/documentation.md", "Publishing"=>"project/publishing.md",
        ],
    ],
)
DocsBuildSupport.write_receipt(joinpath(@__DIR__, "build", "build-receipt.toml");
    stage="manual", packages=Dict("LM15"=>string(Base.pkgversion(LM15)),
        "Documenter"=>string(Base.pkgversion(Documenter)), "Literate"=>string(Base.pkgversion(Literate))))
