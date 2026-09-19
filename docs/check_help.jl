# These checks are authored for the future docs build. They have not been run in
# the writing pass. An alias must point to a documented object, not waive coverage.
using REPL

function check_public_help(source)
    aliases = Dict(:StaticCredential => :ApiKey, :RequestTimeoutError => :TimeoutError)
    public_names = filter(!=(:LM15), names(LM15))
    missing = Symbol[]
    for name in public_names
        target = get(aliases, name, name)
        Docs.hasdoc(LM15, target) || push!(missing, name)
    end
    isempty(missing) || error("Missing public help: " * join(string.(missing), ", "))

    listed = Set{Symbol}(keys(aliases))
    for (directory, _, files) in walkdir(joinpath(source, "reference"))
        for file in files
            endswith(file, ".md") || continue
            text = read(joinpath(directory, file), String)
            for block in eachmatch(r"```@docs\s*\n(.*?)```"s, text)
                for line in split(block.captures[1], '\n')
                    entry = strip(line)
                    isempty(entry) && continue
                    name = first(split(entry, '('; limit=2))
                    push!(listed, Symbol(replace(name, "LM15."=>"", "Base."=>"")))
                end
            end
        end
    end
    omitted = filter(name -> !(name in listed), public_names)
    isempty(omitted) || error("Exports omitted from reference: " * join(string.(omitted), ", "))
end
