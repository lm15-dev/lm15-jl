# Model discovery — live provider APIs + models.dev fallback.

function fetch_openai_models(api_key::String; timeout=5.0)
    io = IOBuffer()
    Downloads.request("https://api.openai.com/v1/models";
        method="GET", output=io, timeout=timeout,
        headers=["Authorization" => "Bearer $api_key"])
    data = JSON.parse(String(take!(io)))
    [ModelSpec(string(get(item, "id", "")), "openai", nothing, nothing, String[], String[], false, false, false, item)
     for item in get(data, "data", Any[]) if get(item, "id", "") != ""]
end

function fetch_anthropic_models(api_key::String; timeout=5.0)
    io = IOBuffer()
    Downloads.request("https://api.anthropic.com/v1/models";
        method="GET", output=io, timeout=timeout,
        headers=["x-api-key" => api_key, "anthropic-version" => "2023-06-01"])
    data = JSON.parse(String(take!(io)))
    [ModelSpec(string(get(item, "id", "")), "anthropic", nothing, nothing, String[], String[], false, false, false, item)
     for item in get(data, "data", Any[]) if get(item, "id", "") != ""]
end

function fetch_gemini_models(api_key::String; timeout=5.0)
    io = IOBuffer()
    Downloads.request("https://generativelanguage.googleapis.com/v1beta/models?key=$api_key";
        method="GET", output=io, timeout=timeout)
    data = JSON.parse(String(take!(io)))
    specs = ModelSpec[]
    for item in get(data, "models", Any[])
        name = string(get(item, "name", ""))
        isempty(name) && continue
        id = startswith(name, "models/") ? name[8:end] : name
        push!(specs, ModelSpec(id, "gemini",
            get(item, "inputTokenLimit", nothing) |> x -> x isa Number ? Int(x) : nothing,
            get(item, "outputTokenLimit", nothing) |> x -> x isa Number ? Int(x) : nothing,
            String[], String[], false, false, false, item))
    end
    specs
end

const LIVE_FETCHERS = Dict{String,Function}(
    "openai" => fetch_openai_models,
    "anthropic" => fetch_anthropic_models,
    "gemini" => fetch_gemini_models,
)

function _resolve_api_keys(; api_key=nothing)
    env_map = providers()
    resolved = Dict{String,String}()
    if api_key isa String
        for p in keys(env_map); resolved[p] = api_key; end
    elseif api_key isa Dict
        merge!(resolved, api_key)
    end
    for (p, vars) in env_map
        haskey(resolved, p) && continue
        for v in vars
            val = get(ENV, v, nothing)
            if val !== nothing; resolved[p] = val; break; end
        end
    end
    resolved
end

"""Discover available models from live provider APIs and models.dev."""
function models(; provider=nothing, timeout=5.0, api_key=nothing,
                  supports=nothing, input_modalities=nothing, output_modalities=nothing)
    env_map = providers()
    selected = provider !== nothing ? [provider] : collect(keys(env_map))
    keys_map = _resolve_api_keys(api_key=api_key)

    live_specs = ModelSpec[]
    for p in selected
        key = get(keys_map, p, nothing)
        key === nothing && continue
        fetcher = get(LIVE_FETCHERS, p, nothing)
        fetcher === nothing && continue
        try
            append!(live_specs, fetcher(key; timeout=timeout))
        catch; end
    end

    fallback = ModelSpec[]
    try
        all = fetch_models_dev(timeout=timeout)
        fallback = filter(s -> s.provider in selected, all)
    catch; end

    merged = _merge_specs(live_specs, fallback)
    _filter_specs(merged; supports=supports, input_modalities=input_modalities, output_modalities=output_modalities)
end

"""Get provider status info."""
function providers_info(; api_key=nothing, timeout=5.0)
    env_map = providers()
    keys_map = _resolve_api_keys(api_key=api_key)
    specs = try models(api_key=api_key, timeout=timeout) catch; ModelSpec[] end

    counts = Dict{String,Int}()
    for s in specs; counts[s.provider] = get(counts, s.provider, 0) + 1; end

    Dict(p => Dict("env_keys" => vars, "configured" => haskey(keys_map, p), "model_count" => get(counts, p, 0))
         for (p, vars) in env_map)
end

function _merge_specs(primary, fallback)
    merged = Dict{String,ModelSpec}()
    for s in primary; merged["$(s.provider):$(s.id)"] = s; end
    for f in fallback
        key = "$(f.provider):$(f.id)"
        if !haskey(merged, key)
            merged[key] = f
        else
            existing = merged[key]
            merged[key] = ModelSpec(
                existing.id, existing.provider,
                something(existing.context_window, f.context_window),
                something(existing.max_output, f.max_output),
                isempty(existing.input_modalities) ? f.input_modalities : existing.input_modalities,
                isempty(existing.output_modalities) ? f.output_modalities : existing.output_modalities,
                existing.tool_call || f.tool_call,
                existing.structured_output || f.structured_output,
                existing.reasoning || f.reasoning,
                merge(f.raw, existing.raw),
            )
        end
    end
    sort(collect(values(merged)), by=s -> (s.provider, s.id))
end

function _filter_specs(specs; supports=nothing, input_modalities=nothing, output_modalities=nothing)
    filter(specs) do s
        if supports !== nothing
            features = Set{String}()
            s.tool_call && push!(features, "tools")
            s.structured_output && push!(features, "json_output")
            s.reasoning && push!(features, "reasoning")
            issubset(supports, features) || return false
        end
        if input_modalities !== nothing
            issubset(input_modalities, Set(s.input_modalities)) || return false
        end
        if output_modalities !== nothing
            issubset(output_modalities, Set(s.output_modalities)) || return false
        end
        true
    end
end
