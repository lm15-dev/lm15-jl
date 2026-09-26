struct RouteRule
    prefix::String
    provider::String
    note::String
end
RouteRule(prefix, provider; note="") = RouteRule(prefix, canonical_provider(provider), note)
const ROUTING_DATA=JSON.parse(read(joinpath(@__DIR__, "data", "routing.json"), String))
const DEFAULT_RULES=Tuple(
    RouteRule(r["prefix"], r["provider"]; note=get(r, "note", "")) for r in ROUTING_DATA["DEFAULT_RULES"]
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
"""
    RouterConfig(; registry=nothing, rules=DEFAULT_RULES, env=nothing, api_keys=Dict(),
                 base_urls=Dict(), settings=Dict(), credentials=Dict(), providers=(),
                 auth=nothing, transport=nothing, adaptations="note")

How an `LMRouter` routes and authenticates. `api_keys` are explicit credentials (a string,
a credential value, or a zero-argument function) per provider; `credentials` names a
cloud identity per cloud door (`"platform"`, `"workload"`, `"environment"`, `"cli"`);
`base_urls` are server roots (a cloud door's endpoint root); `settings` are host settings;
`providers` declares providers LM15 does not list; `auth` attaches a managed `Auth`
(AUTH-15 mode B: explicit keys and named identities first, then the scope's saved
connection; never an ambient key or the machine's cloud identity).
"""
Base.@kwdef struct RouterConfig
    registry::Maybe{ModelRegistry} = nothing
    rules::Tuple = DEFAULT_RULES
    env::Union{Nothing,AbstractDict} = nothing
    api_keys::AbstractDict = Dict{String,Any}()
    base_urls::AbstractDict = Dict{String,String}()
    settings::AbstractDict = Dict{String,Any}()
    credentials::AbstractDict = Dict{String,String}()
    providers::Tuple = ()
    auth::Any = nothing
    transport::Any = nothing
    adaptations::String = "note"
end
function Base.show(io::IO, ::RouterConfig)
    return print(io, "RouterConfig(<credentials and environment withheld>)")
end
"""A copy of `config` with a managed `Auth` attached (or detached with `nothing`)."""
with_auth(config::RouterConfig, auth) =
    RouterConfig(; (n => getfield(config, n) for n in fieldnames(RouterConfig) if n !== :auth)..., auth)

# Routes that exist only for a managed connection: no LM15 wire receipt, so they
# are not in the registry; declared, marked as such, added only to a router that
# carries a managed Auth (the only way to have a credential for them).
const COPILOT_HEADER_PAIRS = let p = JSON.parse(read(joinpath(@__DIR__, "data", "login_profiles.json"), String))
    [(String(k), String(v)) for (k, v) in p["providers"]["github-copilot"]["headers"]]
end
const KIMI_CODE = ProviderDefinition(
    AccessPolicy(; provider="kimi-code", supports=EndpointSupport(; complete=true, stream=true),
        auth_modes=("bearer",), auth_scheme=("bearer",), base_url="https://api.kimi.com/coding");
    dialect="anthropic", compat=AnthropicCompat(),
    note="Kimi Code subscription over the Anthropic Messages wire (managed login only; no lm15 wire receipt yet)")
const GITHUB_COPILOT = ProviderDefinition(
    AccessPolicy(; provider="github-copilot", supports=EndpointSupport(; complete=true, stream=true, models=true),
        auth_modes=("bearer",), auth_scheme=("bearer",), headers=Tuple(Tuple(h) for h in COPILOT_HEADER_PAIRS),
        base_url="https://api.individual.githubcopilot.com");
    dialect="openai-chat",
    compat=OpenAIChatCompat(; instruction_role="system", max_tokens_field="max_completion_tokens",
        stream_usage="include", thinking_format="reasoning_effort"),
    note="GitHub Copilot over the Chat Completions wire (managed login only; the account's host comes from the token; no lm15 wire receipt yet)")
const DECLARED_LOGIN_PROVIDERS = (KIMI_CODE, GITHUB_COPILOT)

"""The providers this config routes to: the registry, the config's declarations, and the
connection-only routes when a managed Auth is attached."""
function provider_table(config::RouterConfig)
    declared = collect(ProviderDefinition, config.providers)
    if config.auth !== nothing
        taken = Set(d.id for d in declared)
        append!(declared, [d for d in DECLARED_LOGIN_PROVIDERS if !(d.id in taken)])
    end
    isempty(declared) && return PROVIDERS
    table = copy(PROVIDERS)
    for d in declared
        table[d.id] = d
    end
    return table
end
declared_provider(config::RouterConfig, id) = !haskey(PROVIDERS, id) && haskey(provider_table(config), id)
function table_provider(table, name)
    id = canonical_provider(name)
    haskey(table, id) && return id
    for d in values(table)
        id in d.aliases && return d.id
    end
    return nothing
end
function check_declared(providers)
    providers isa Tuple && all(d -> d isa ProviderDefinition, providers) ||
        throw(ArgumentError("RouterConfig(providers=...) takes a tuple of ProviderDefinition"))
    litellm = Set(keys(LITELLM_PROVIDER_PREFIXES))
    taken = Dict{String,String}()
    for d in providers, spelling in spellings(d)
        built_in = haskey(PROVIDERS, spelling) ? PROVIDERS[spelling].id : (spelling in litellm ? LITELLM_PROVIDER_PREFIXES[spelling] : nothing)
        built_in === nothing || throw(NotConfiguredError(
            "RouterConfig(providers=...): $(repr(spelling)) already names lm15's $(repr(built_in)) door; a declared provider takes a new id and aliases"))
        haskey(taken, spelling) && throw(NotConfiguredError(
            "RouterConfig(providers=...): $(repr(spelling)) is spelled by both $(repr(taken[spelling])) and $(repr(d.id))"))
        taken[spelling] = d.id
    end
end

"""
    Resolution

Where a model string routes: the provider, the wire model, the rung that decided
(`prefix`, `catalog`, `rule`), and whether the provider is `declared` (no LM15 receipt).
"""
Base.@kwdef struct Resolution
    requested::String
    model::String
    provider::String
    source::String
    rule::Maybe{RouteRule} = nothing
    model_info::Maybe{ModelInfo} = nothing
    declared::Bool = false
end
function describe(r::Resolution)
    text = "$(repr(r.requested)) → $(r.provider):$(r.model) via $(r.source)"
    r.declared && (text *= " (declared by RouterConfig(providers=...) — no lm15 receipts)")
    return text
end
Base.show(io::IO, r::Resolution) = print(io, describe(r))
mutable struct LMRouter
    config::RouterConfig
    clients::Dict{String,ProviderLM}
    lock::ReentrantLock
end
function LMRouter(config::RouterConfig=RouterConfig())
    check_declared(config.providers)
    config.auth === nothing || config.auth isa Auth || throw(ArgumentError("RouterConfig(auth=...) takes an Auth"))
    table = provider_table(config)
    for (label, mapping) in (
        (:api_keys, config.api_keys), (:base_urls, config.base_urls), (:settings, config.settings),
        (:credentials, config.credentials),
    )
        seen=Set{String}()
        for key in keys(mapping)
            key isa AbstractString ||
                throw(NotConfiguredError("$label keys must be provider names"))
            canonical=table_provider(table, key)
            canonical===nothing && throw(NotConfiguredError("$label contains unknown provider $key"))
            canonical in seen &&
                throw(NotConfiguredError("$label has duplicate spellings for $canonical"))
            push!(seen, canonical)
        end
    end
    # AUTH-1 named credentials: one identity per cloud door, a known name, and
    # never together with an api_keys entry for the same provider.
    for (key, name) in config.credentials
        provider=table_provider(table, key)
        check_named(table[provider].access, name)
        any(k->table_provider(table, k)==provider, keys(config.api_keys)) && throw(NotConfiguredError(
            "$provider: both an api_keys entry and the named credential $(repr(name)); a door has one identity — give one";
            provider))
    end
    return LMRouter(config, Dict{String,ProviderLM}(), ReentrantLock())
end
function resolve(router::LMRouter, model::AbstractString)
    requested=String(model)
    isempty(requested) && throw(UnknownModelError("model must not be empty"; model=requested))
    table=provider_table(router.config)
    bits=split(requested, ':'; limit=2)
    if length(bits)==2 && !isempty(bits[2])
        provider=table_provider(table, bits[1])
        provider===nothing || return Resolution(;
            requested, model=String(bits[2]), provider, source="prefix", declared=!haskey(PROVIDERS, provider))
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
            provider=table_provider(table, info.provider)
            provider===nothing &&
                throw(UnknownModelError("catalog provider is not routable"; model=requested))
            return Resolution(;
                requested, model=info.id, provider, source="catalog", model_info=info,
                declared=!haskey(PROVIDERS, provider),
            )
        end
    end
    for rule in router.config.rules
        startswith(requested, rule.prefix) || continue
        provider=table_provider(table, rule.provider)
        provider===nothing &&
            throw(UnknownModelError("routing rule names an unknown provider"; model=requested))
        return Resolution(; requested, model=requested, provider, source="rule", rule,
            declared=!haskey(PROVIDERS, provider))
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
    # A managed router consults its scope on every construction; its clients
    # are not cached, so a replaced or signed-out connection is seen at once.
    config.auth===nothing || return managed_lm(router, provider)
    lock(router.lock) do
        haskey(router.clients, provider) && return router.clients[provider]
        return router.clients[provider]=build_lm(router, provider)
    end
end
function router_endpoint(config, definition, provider, env)
    base=provider_setting(config.base_urls, provider)
    # A cloud door takes a URL root: the explicit entry, then the vendor's own
    # endpoint variable, then the template (AUTH-10, 2026-09-19).
    base===nothing && definition.access.host!==nothing &&
        (base=first(endpoint_from_env(definition.access.host, env)))
    return base
end
function build_lm(router::LMRouter, provider)
    config=router.config
    table=provider_table(config)
    definition=table[provider]
    source=if definition.access.credential_policy=="oauth"
        nothing
    else
        explicit_source(provider, config.api_keys; table)
    end
    key=source===nothing ? nothing : config.api_keys[source]
    env=config.env===nothing ? ENV : config.env
    return ProviderLM(
        haskey(PROVIDERS, provider) ? provider : definition;
        api_key=key,
        base_url=router_endpoint(config, definition, provider, env),
        settings=provider_setting(config.settings, provider, Dict{String,String}()),
        env,
        transport=config.transport,
        adaptations=config.adaptations,
        credential=provider_setting(config.credentials, provider),
    )
end
"""AUTH-15 mode B: the explicit api_keys entry; an explicit named cloud identity; the
scope's saved connection (renewed per request); a keyless local server's placeholder.
Never an environment key, a foreign CLI file or the machine's cloud chain."""
function managed_lm(router::LMRouter, provider)
    config=router.config
    auth=config.auth
    table=provider_table(config)
    definition=table[provider]
    source=explicit_source(provider, config.api_keys; table)
    api_key=source===nothing ? nothing : config.api_keys[source]
    named=provider_setting(config.credentials, provider)
    env=config.env===nothing ? ENV : config.env
    base_url=router_endpoint(config, definition, provider, env)
    hosted=definition.access.host!==nothing
    account_id=nothing
    origin=nothing
    access=definition.access
    if source===nothing && named===nothing
        saved=try
            request_auth(auth, provider)
        catch e
            e isa AuthOperationError && e.reason=="login_required" || rethrow()
            if definition.placeholder_key!==nothing && !status(auth, provider).logged_out
                api_key=definition.placeholder_key
                origin="the local server's placeholder key"
                nothing
            elseif hosted
                throw(auth_operation_error(
                    "$provider: no saved connection in this scope; the machine's cloud identity is not used under a managed Auth — save a named identity (configure(auth, $(repr(provider)); method=\"cloud\")) or pass RouterConfig(credentials=...) explicitly";
                    reason="login_required", stage="resolution", recovery="select_connection", provider))
            else
                rethrow()
            end
        end
        if saved!==nothing
            connection=status(auth, provider).connection
            if saved.named!==nothing
                named=saved.named  # a saved cloud recipe names the identity; the chain rung runs
            else
                api_key=credential_provider(auth, provider)
                account_id=saved.account_id
                saved.base_url===nothing || base_url!==nothing || (base_url=saved.base_url)
                if !isempty(saved.headers) && !hosted
                    have=Set(lowercase(first(h)) for h in access.headers)
                    extra=Tuple((k, v) for (k, v) in saved.headers if !(lowercase(k) in have) && lowercase(k)!="chatgpt-account-id")
                    isempty(extra) || (access=AccessPolicy(; (n=>getfield(access, n) for n in fieldnames(AccessPolicy) if n!==:headers)..., headers=(access.headers..., extra...)))
                end
            end
            origin=connection===nothing ? "managed connection" : "managed connection $(connection.id) ($(connection.label))"
        end
    end
    if api_key===nothing && named===nothing
        throw(auth_operation_error("$provider: no credential for this door under a managed Auth";
            reason="login_required", stage="resolution", recovery="select_connection", provider))
    end
    base=haskey(PROVIDERS, provider) && access===definition.access ? provider :
        ProviderDefinition(; id=definition.id, dialect=definition.dialect, access, compat=definition.compat,
            placeholder_key=definition.placeholder_key, console_url=definition.console_url, note=definition.note,
            aliases=definition.aliases)
    return ProviderLM(
        base;
        api_key,
        base_url,
        settings=provider_setting(config.settings, provider, Dict{String,String}()),
        env,
        transport=config.transport,
        adaptations=config.adaptations,
        credential=named,
        account_id,
        credential_origin=origin,
    )
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
        e isa Union{NotConfiguredError,AuthOperationError} || rethrow()
        definition=provider_table(router.config)[resolution.provider]
        ProviderLM(haskey(PROVIDERS, resolution.provider) ? resolution.provider : definition;
            api_key="unused: plan sends nothing", adaptations=router.config.adaptations,
            account_id=definition.access.backend=="chatgpt-codex" ? "planning" : nothing)
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
