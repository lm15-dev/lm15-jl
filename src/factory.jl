# Factory — build a configured UniversalLM.

const ADAPTER_DEFS = [
    ("openai", ["OPENAI_API_KEY"], (key, ) -> OpenAIAdapter(key)),
    ("anthropic", ["ANTHROPIC_API_KEY"], (key, ) -> AnthropicAdapter(key)),
    ("gemini", ["GEMINI_API_KEY", "GOOGLE_API_KEY"], (key, ) -> GeminiAdapter(key)),
]

function providers()
    Dict(name => keys for (name, keys, _) in ADAPTER_DEFS)
end

function build_default(; api_key=nothing, env=nothing)
    client = UniversalLM()

    # Build env key map
    env_key_map = Dict{String,String}()
    for (provider, keys, _) in ADAPTER_DEFS
        for k in keys
            env_key_map[k] = provider
        end
    end

    # Resolve explicit keys
    explicit = Dict{String,String}()
    if api_key isa String
        for (provider, _, _) in ADAPTER_DEFS
            explicit[provider] = api_key
        end
    elseif api_key isa Dict
        merge!(explicit, api_key)
    end

    # Parse env file
    file_keys = Dict{String,String}()
    if env !== nothing
        file_keys = parse_env_file(env, env_key_map)
    end

    # Register adapters
    for (provider, env_vars, create) in ADAPTER_DEFS
        key = get(explicit, provider, nothing)
        if key === nothing
            key = get(file_keys, provider, nothing)
        end
        if key === nothing
            for var in env_vars
                val = get(ENV, var, nothing)
                if val !== nothing
                    key = val
                    break
                end
            end
        end
        key !== nothing && register!(client, create(key))
    end

    client
end

function parse_env_file(path::String, env_key_map::Dict{String,String})
    result = Dict{String,String}()
    expanded = startswith(path, "~/") ? joinpath(homedir(), path[3:end]) : path
    isfile(expanded) || return result

    for line in readlines(expanded)
        line = strip(line)
        (isempty(line) || startswith(line, '#')) && continue
        startswith(line, "export ") && (line = line[8:end])
        idx = findfirst('=', line)
        idx === nothing && continue
        key = strip(line[1:idx-1])
        value = strip(line[idx+1:end])
        # Strip quotes
        if length(value) >= 2 && ((value[1] == '"' && value[end] == '"') || (value[1] == '\'' && value[end] == '\''))
            value = value[2:end-1]
        end
        if haskey(env_key_map, key) && !isempty(value)
            result[env_key_map[key]] = value
            !haskey(ENV, key) && (ENV[key] = value)
        end
    end
    result
end
