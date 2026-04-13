# UniversalLM — routes requests to the correct provider adapter.

const AdapterType = Union{OpenAIAdapter, AnthropicAdapter, GeminiAdapter}

mutable struct UniversalLM
    adapters::Dict{String, AdapterType}
end

UniversalLM() = UniversalLM(Dict{String, AdapterType}())

function register!(lm::UniversalLM, adapter::AdapterType)
    lm.adapters[provider_name(adapter)] = adapter
end

function resolve_adapter(lm::UniversalLM, model::String, provider::String="")
    p = isempty(provider) ? resolve_provider(model) : provider
    haskey(lm.adapters, p) || throw(ProviderError("no adapter registered for provider '$p'"))
    lm.adapters[p]
end

function complete(lm::UniversalLM, request::LMRequest; provider::String="")
    adapter = resolve_adapter(lm, request.model, provider)
    req = build_request(adapter, request, false)
    resp = http_request(req)
    resp.status >= 400 && throw(normalize_error(adapter, resp.status, text(resp)))
    parse_response(adapter, request, resp)
end

function stream_events(lm::UniversalLM, request::LMRequest; provider::String="")
    adapter = resolve_adapter(lm, request.model, provider)
    req = build_request(adapter, request, true)
    resp_io = http_stream(req)
    sse_events = parse_sse(resp_io)
    events = StreamEvent[]
    for raw in sse_events
        evt = parse_stream_event(adapter, request, raw)
        evt !== nothing && push!(events, evt)
    end
    events
end

function do_embeddings(lm::UniversalLM, request::EmbeddingRequest; provider::String="")
    adapter = resolve_adapter(lm, request.model, provider)
    do_embeddings(adapter, request)
end
