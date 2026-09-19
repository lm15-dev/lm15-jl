module DocsBuildSupport

using SHA, TOML

const ROOT = normpath(joinpath(@__DIR__, ".."))
const CORE_TUTORIALS = ("no-key", "conversation", "streaming", "tools")
const SCIENCE_TUTORIALS = ("tables", "units", "simulation")
const LIVE_SOURCES = ("live-tools.jl", "live-stream.jl")
const BANNER = """
!!! warning "Written, not yet verified"
    This documentation revision was written without running its examples, tests or
    site build. Earlier verification records describe an earlier source revision.
    Read [status](../project/status.md) before treating examples as verified behavior.

"""

function require_isolation()
    get(ENV, "LM15_DOCS_ISOLATED", "") == "1" || error(
        "Run through docs/build.sh in an isolated environment; see docs/README.md")
    get(ENV, "LM15_RUN_LIVE_EXAMPLES", "no") == "no" ||
        error("Live examples must remain disabled during documentation generation")
    # Presence checks only; never print values. This is an accidental-use guard,
    # not a replacement for the network namespace and fresh home directory.
    for name in keys(ENV)
        if occursin(r"(?:API_KEY|ACCESS_TOKEN|REFRESH_TOKEN|SECRET_ACCESS_KEY)$", name) ||
           name in ("AWS_ACCESS_KEY_ID", "AWS_SESSION_TOKEN", "GOOGLE_APPLICATION_CREDENTIALS",
                    "AZURE_CLIENT_SECRET", "AZURE_FEDERATED_TOKEN_FILE", "AWS_PROFILE")
            error("Remove credential environment variable $name before building documentation")
        end
    end
end

function source_digest()
    files = String[]
    for folder in ("src", "ext", "test", "docs/src", "examples/tutorials", "examples/scientific")
        for (directory, _, names) in walkdir(joinpath(ROOT, folder))
            for name in names
                name == "Manifest.toml" && continue
                any(suffix -> endswith(name, suffix),
                    (".jl", ".md", ".toml", ".css", ".svg", ".json", ".png", ".ico")) || continue
                push!(files, relpath(joinpath(directory, name), ROOT))
            end
        end
    end
    append!(files, ["Project.toml", "CONTRACT_PIN", "docs/Project.toml", "docs/make.jl",
        "docs/scientific.jl", "docs/build_support.jl", "docs/check_help.jl",
        "docs/build.sh", "docs/deploy.jl", ".github/workflows/documentation.yml"])
    buffer = IOBuffer()
    for relative in sort!(unique(files))
        write(buffer, replace(relative, '\\'=>'/'), '\0', read(joinpath(ROOT, relative)), '\0')
    end
    return bytes2hex(sha256(take!(buffer)))
end

function write_receipt(path; stage, packages=Dict{String,String}())
    open(path, "w") do io
        TOML.print(io, Dict(
            "stage"=>stage, "source_sha256"=>source_digest(),
            "julia"=>string(VERSION), "packages"=>packages,
            "live_provider_calls"=>false, "human_reviewed"=>false))
    end
end

function require_matching_receipt(path, stage)
    isfile(path) || error("Missing $stage output: run the documented isolated build first")
    receipt = TOML.parsefile(path)
    receipt["stage"] == stage || error("Wrong documentation artifact stage")
    receipt["source_sha256"] == source_digest() ||
        error("Documentation artifact is stale; regenerate it from the current sources")
    return receipt
end

function set_edit_url(path, relative_source)
    ref = get(ENV, "LM15_DOCS_SOURCE_REF", "main")
    url = "https://github.com/lm15-dev/lm15-jl/edit/$ref/" * replace(relative_source, '\\'=>'/')
    text = read(path, String)
    assignment = "EditURL = " * repr(url)
    if occursin(r"(?m)^EditURL\s*=", text)
        text = replace(text, r"(?m)^EditURL\s*=.*$" => assignment)
    else
        text = "```@meta\n" * assignment * "\n```\n\n" * text
    end
    write(path, text)
end

function add_download_link(path, source_name)
    stem = first(splitext(source_name))
    folder = stem in SCIENCE_TUTORIALS ? "scientific" : "tutorials"
    set_edit_url(path, "examples/$folder/$source_name")
    text = read(path, String)
    # Literate's first heading stays the page title. The source download is local
    # to the rendered artifact, so no generated branch/tag is assumed to exist.
    lines = split(text, '\n'; keepempty=true)
    heading = findfirst(line -> startswith(line, "# "), lines)
    insertion = "\n" * BANNER * "[Download the Julia source](../downloads/$source_name).\n"
    if heading === nothing
        write(path, insertion * text)
    else
        insert!(lines, heading + 1, insertion)
        write(path, join(lines, '\n'))
    end
end

end
