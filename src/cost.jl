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

const ADDITIVE_CACHE_PROVIDERS = Set(["anthropic"])
const SEPARATE_REASONING_PROVIDERS = Set(["gemini", "google"])

function estimate_cost(usage::Usage, rates::Dict{String,Float64}, provider::String)
    per(r) = get(rates, r, 0.0) / 1_000_000

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
