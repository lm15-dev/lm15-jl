# Model catalog — fetch model specs from models.dev.

struct ModelSpec
    id::String
    provider::String
    context_window::Union{Int,Nothing}
    max_output::Union{Int,Nothing}
    input_modalities::Vector{String}
    output_modalities::Vector{String}
    tool_call::Bool
    structured_output::Bool
    reasoning::Bool
    raw::Dict{String,Any}
end

function fetch_models_dev(; timeout::Float64=20.0)
    io = IOBuffer()
    Downloads.request("https://models.dev/api.json";
        method="GET", output=io, timeout=timeout,
        headers=["User-Agent" => "lm15"])
    data = JSON.parse(String(take!(io)))

    providers_data = get(data, "providers", data)
    specs = ModelSpec[]

    for (provider_id, provider_payload) in providers_data
        provider_payload isa Dict || continue
        models = get(provider_payload, "models", nothing)
        models isa Dict || continue
        for (model_id, m) in models
            m isa Dict || continue
            limit = get(m, "limit", Dict())
            modalities = get(m, "modalities", Dict())
            push!(specs, ModelSpec(
                model_id, string(provider_id),
                get(limit, "context", nothing) |> x -> x isa Number ? Int(x) : nothing,
                get(limit, "output", nothing) |> x -> x isa Number ? Int(x) : nothing,
                String[string(x) for x in get(modalities, "input", [])],
                String[string(x) for x in get(modalities, "output", [])],
                Bool(get(m, "tool_call", false)),
                Bool(get(m, "structured_output", false)),
                Bool(get(m, "reasoning", false)),
                m,
            ))
        end
    end
    specs
end

function build_provider_model_index(specs::Vector{ModelSpec})
    out = Dict{String,Dict{String,ModelSpec}}()
    for s in specs
        provider_map = get!(out, s.provider, Dict{String,ModelSpec}())
        provider_map[s.id] = s
    end
    out
end
