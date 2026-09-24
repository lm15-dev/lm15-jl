struct RouteRule
    prefix::String
    provider::String
    note::String
end
RouteRule(prefix, provider; note="") = RouteRule(prefix, canonical_provider(provider), note)
const DEFAULT_RULES=Tuple(
    RouteRule(prefix, provider) for (prefix, provider) in (
        ("claude-", "anthropic"),
        ("gpt-", "openai"),
        ("o1", "openai"),
        ("o3", "openai"),
        ("o4", "openai"),
        ("gemini-", "gemini"),
        ("gemma-", "gemini"),
        ("nano-banana", "gemini"),
        ("grok-", "xai"),
        ("sora-", "openai"),
        ("veo-", "gemini"),
        ("chat-latest", "openai"),
    )
)
mutable struct ModelRegistry
    models::OrderedDict{Tuple{String,String},ModelInfo}
    lock::ReentrantLock
end
ModelRegistry() = ModelRegistry(OrderedDict{Tuple{String,String},ModelInfo}(), ReentrantLock())
function register!(registry::ModelRegistry, model::ModelInfo; replace=true)
    validate(model)
    lock(registry.lock) do
        key=(canonical_provider(model.provider), model.id)
        !replace && haskey(registry.models, key) && throw(ArgumentError("model already registered"))
        return registry.models[key]=model
    end
    return registry
end
function ModelRegistry(models)
    registry=ModelRegistry()
    for model in models
        register!(
            registry, model isa ModelInfo ? model : from_dict(ModelInfo, model); replace=false
        )
    end
    return registry
end
function list_models(registry::ModelRegistry; provider=nothing)
    lock(registry.lock) do
        return [
            m for m in values(registry.models) if
            provider===nothing || canonical_provider(m.provider)==canonical_provider(provider)
        ]
    end
end
Base.@kwdef struct RouterConfig
    registry::Maybe{ModelRegistry} = nothing
    rules::Tuple = DEFAULT_RULES
    env::Union{Nothing,AbstractDict} = nothing
    api_keys::AbstractDict = Dict{String,Any}()
    base_urls::AbstractDict = Dict{String,String}()
    settings::AbstractDict = Dict{String,Any}()
    transport::Any = nothing
    adaptations::String = "note"
end
function Base.show(io::IO, ::RouterConfig)
    return print(io, "RouterConfig(<credentials and environment withheld>)")
end
Base.@kwdef struct Resolution
    requested::String
    model::String
    provider::String
    source::String
    rule::Maybe{RouteRule} = nothing
    model_info::Maybe{ModelInfo} = nothing
end
function describe(r::Resolution)
    return "$(repr(r.requested)) → $(r.provider):$(r.model) via $(r.source)"
end
Base.show(io::IO, r::Resolution) = print(io, describe(r))
mutable struct LMRouter
    config::RouterConfig
    clients::Dict{String,ProviderLM}
    lock::ReentrantLock
end
function LMRouter(config::RouterConfig=RouterConfig())
    for (label, mapping) in
        ((:api_keys, config.api_keys), (:base_urls, config.base_urls), (:settings, config.settings))
        seen=Set{String}()
        for key in keys(mapping)
            key isa AbstractString ||
                throw(NotConfiguredError("$label keys must be provider names"))
            canonical=canonical_provider(key)
            haskey(PROVIDERS, canonical) ||
                throw(NotConfiguredError("$label contains unknown provider $key"))
            canonical in seen &&
                throw(NotConfiguredError("$label has duplicate spellings for $canonical"))
            push!(seen, canonical)
        end
    end
    return LMRouter(config, Dict{String,ProviderLM}(), ReentrantLock())
end
function resolve(router::LMRouter, model::AbstractString)
    requested=String(model)
    isempty(requested) && throw(UnknownModelError("model must not be empty"; model=requested))
    bits=split(requested, ':'; limit=2)
    if length(bits)==2 && !isempty(bits[2]) && haskey(PROVIDERS, canonical_provider(bits[1]))
        return Resolution(;
            requested, model=String(bits[2]), provider=canonical_provider(bits[1]), source="prefix"
        )
    end
    registry=router.config.registry
    if registry!==nothing
        matches=[m for m in list_models(registry) if m.id==requested || requested in m.aliases]
        providers=unique([canonical_provider(m.provider) for m in matches])
        length(providers)>1 && throw(
            AmbiguousModelError(
                "model is offered by multiple providers; use provider:model";
                model=requested,
                providers=Tuple(providers),
            ),
        )
        if !isempty(matches)
            exact=[m for m in matches if m.id==requested]
            narrowed=isempty(exact) ? matches : exact
            length(narrowed)>1 && throw(
                AmbiguousModelError(
                    "model matches multiple catalog entries; use a canonical id";
                    model=requested,
                    providers=Tuple(providers),
                ),
            )
            info=only(narrowed)
            provider=canonical_provider(info.provider)
            haskey(PROVIDERS, provider) ||
                throw(UnknownModelError("catalog provider is not routable"; model=requested))
            return Resolution(;
                requested, model=info.id, provider, source="catalog", model_info=info
            )
        end
    end
    for rule in router.config.rules
        startswith(requested, rule.prefix) || continue
        provider=canonical_provider(rule.provider)
        haskey(PROVIDERS, provider) ||
            throw(UnknownModelError("routing rule names an unknown provider"; model=requested))
        return Resolution(; requested, model=requested, provider, source="rule", rule)
    end
    return throw(
        UnknownModelError(
            "no provider prefix, catalog entry, or routing rule matched; use provider:model";
            model=requested,
        ),
    )
end
function provider_setting(mapping, provider, default=nothing)
    for (key, value) in mapping
        canonical_provider(key)==provider && return value
    end
    return default
end
function lm(router::LMRouter, model::AbstractString)
    resolution=resolve(router, model)
    provider=resolution.provider
    config=router.config
    lock(router.lock) do
        haskey(router.clients, provider) && return router.clients[provider]
        definition=provider_definition(provider)
        source=if definition.access.credential_policy=="oauth"
            nothing
        else
            explicit_source(provider, config.api_keys)
        end
        key=source===nothing ? nothing : config.api_keys[source]
        base=provider_setting(config.base_urls, provider)
        definition.access.host!==nothing &&
            base!==nothing &&
            throw(
                NotConfiguredError(
                    "cloud URLs are derived from host settings; configure region/resource instead";
                    provider,
                ),
            )
        settings=provider_setting(config.settings, provider, Dict{String,String}())
        client=ProviderLM(
            provider;
            api_key=key,
            base_url=base,
            settings,
            env=config.env===nothing ? ENV : config.env,
            transport=config.transport,
            adaptations=config.adaptations,
        )
        return router.clients[provider]=client
    end
end
function complete(router::LMRouter, request::Request)
    resolution=resolve(router, request.model)
    return complete(lm(router, request.model), reconstruct(request; model=resolution.model))
end
function plan(router::LMRouter, request::Request; stream=false)
    resolution=resolve(router, request.model)
    # Like resolve, plan is offline: a route with no key still plans; the
    # stand-in credential is never read, because nothing is sent.
    client=try
        lm(router, request.model)
    catch e
        e isa NotConfiguredError || rethrow()
        ProviderLM(resolution.provider; api_key="unused: plan sends nothing", adaptations=router.config.adaptations)
    end
    return plan(client, reconstruct(request; model=resolution.model); stream)
end
function stream(router::LMRouter, request::Request)
    resolution=resolve(router, request.model)
    return stream(lm(router, request.model), reconstruct(request; model=resolution.model))
end
function Base.close(router::LMRouter)
    lock(router.lock) do
        foreach(close, values(router.clients))
        return empty!(router.clients)
    end
    return nothing
end
function estimate(pricing::InferencePricing, usage::Usage)
    return sum((
        if getfield(pricing, rate)===nothing || getfield(usage, count)===nothing
            0.0
        else
            getfield(pricing, rate)*getfield(usage, count)/1_000_000
        end for (rate, count) in (
            (:input_per_million, :input_tokens),
            (:output_per_million, :output_tokens),
            (:cache_read_per_million, :cache_read_tokens),
            (:cache_write_per_million, :cache_write_tokens),
        )
    ))
end
