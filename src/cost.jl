# Cost estimation from Usage + pricing data.

struct CostBreakdown
    input::Float64
    output::Float64
    cache_read::Float64
    cache_write::Float64
    reasoning::Float64
    input_audio::Float64
    output_audio::Float64
    total::Float64
end

CostBreakdown() = CostBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

const ADDITIVE_CACHE_PROVIDERS = Set(["anthropic"])
const SEPARATE_REASONING_PROVIDERS = Set(["gemini", "google"])
const _cost_index_ref = Ref{Union{Dict{String,ModelSpec},Nothing}}(nothing)

function estimate_cost(usage::Usage, rates::Dict{String,<:Real}, provider::String)
    per(r) = Float64(get(rates, r, 0.0)) / 1_000_000

    cr = something(usage.cache_read_tokens, 0)
    cw = something(usage.cache_write_tokens, 0)
    reason = something(usage.reasoning_tokens, 0)
    ia = something(usage.input_audio_tokens, 0)
    oa = something(usage.output_audio_tokens, 0)

    ti = provider in ADDITIVE_CACHE_PROVIDERS ? (usage.input_tokens - ia) : (usage.input_tokens - cr - cw - ia)
    ti = max(ti, 0)
    to_ = provider in SEPARATE_REASONING_PROVIDERS ? (usage.output_tokens - oa) : (usage.output_tokens - reason - oa)
    to_ = max(to_, 0)

    ci = ti * per("input"); co = to_ * per("output")
    ccr = cr * per("cache_read"); ccw = cw * per("cache_write")
    crr = reason * per("reasoning")
    cia = ia * per("input_audio"); coa = oa * per("output_audio")

    CostBreakdown(ci, co, ccr, ccw, crr, cia, coa, ci+co+ccr+ccw+crr+cia+coa)
end

function estimate_cost(usage::Usage, spec::ModelSpec)
    rates_any = get(spec.raw, "cost", Dict{String,Any}())
    rates = Dict{String,Float64}(string(k) => Float64(v) for (k, v) in rates_any if v isa Real)
    estimate_cost(usage, rates, spec.provider)
end

function enable_cost_tracking!()
    specs = fetch_models_dev()
    _cost_index_ref[] = Dict(s.id => s for s in specs if haskey(s.raw, "cost"))
    nothing
end

function disable_cost_tracking!()
    _cost_index_ref[] = nothing
    nothing
end

cost_index() = _cost_index_ref[]

function set_cost_index!(index::Union{Dict{String,ModelSpec},Nothing})
    _cost_index_ref[] = index
    nothing
end

function lookup_cost(model::String, usage::Usage)
    index = _cost_index_ref[]
    index === nothing && return nothing
    spec = get(index, model, nothing)
    spec === nothing && return nothing
    estimate_cost(usage, spec)
end

function sum_costs(costs)
    total = CostBreakdown()
    for cost in costs
        total = CostBreakdown(
            total.input + cost.input,
            total.output + cost.output,
            total.cache_read + cost.cache_read,
            total.cache_write + cost.cache_write,
            total.reasoning + cost.reasoning,
            total.input_audio + cost.input_audio,
            total.output_audio + cost.output_audio,
            total.total + cost.total,
        )
    end
    total
end
