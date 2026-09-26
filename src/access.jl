Base.@kwdef struct EndpointSupport
    complete::Bool = true
    stream::Bool = true
    live::Bool = false
    files::Bool = false
    batches::Bool = false
    images::Bool = false
    speech::Bool = false
    video::Bool = false
    responses_api::Bool = false
    models::Bool = false
    caches::Bool = false
    extra::Tuple = ()
end

function supports_endpoint(support::EndpointSupport, name::Symbol)
    string(name) in support.extra && return true
    return name in fieldnames(EndpointSupport) && name !== :extra && getfield(support, name)
end
Base.@kwdef struct HostSetting
    name::String
    env::Tuple = ()
    default::Maybe{String} = nothing
end
Base.@kwdef struct HostSpec
    base_url::String
    settings::Tuple = ()
    paths::JsonObject = obj()
    model_in::String = "body"
    anthropic_version_in::String = "header"
    stream_framing::String = "sse"
    required_headers::Tuple = ()
    sigv4_service::Maybe{String} = nothing
    endpoint_env::Tuple = ()
end
Base.@kwdef struct AccessPolicy
    provider::String
    supports::EndpointSupport = EndpointSupport()
    auth_modes::Tuple = ()
    enterprise_variants::Tuple = ()
    env_keys::Tuple = ()
    credential_policy::String = "key"
    auth_scheme::Tuple = ("bearer",)
    headers::Tuple = ()
    host::Maybe{HostSpec} = nothing
    login_hint::Maybe{String} = nothing
    backend::String = "api"
    backend_options::JsonObject = obj()
    system_prefix::Maybe{String} = nothing
    base_url::Maybe{String} = nothing
end
Base.@kwdef struct ProviderDefinition
    id::String
    dialect::String
    access::AccessPolicy
    compat::Maybe{String} = nothing
    placeholder_key::Maybe{String} = nothing
    console_url::Maybe{String} = nothing
    note::String = ""
    aliases::Tuple = ()
end
canonical_provider(name::AbstractString) = replace(String(name), '_'=>'-')
tuple_of(v) = v isa AbstractVector ? Tuple(x isa AbstractVector ? Tuple(x) : x for x in v) : v
function policy_from_dict(d)
    kw=Dict{Symbol,Any}(Symbol(k)=>v for (k, v) in d if v!==nothing)
    kw[:supports]=EndpointSupport(;
        (Symbol(k)=>tuple_of(v) for (k, v) in get(d, "supports", obj()))...
    )
    for key in (:auth_modes, :enterprise_variants, :env_keys, :auth_scheme, :headers)
        haskey(kw, key) && (kw[key]=tuple_of(kw[key]))
    end
    if get(kw, :host, nothing) !== nothing
        host=kw[:host]
        h=Dict{Symbol,Any}(Symbol(k)=>v for (k, v) in host)
        h[:settings]=Tuple(
            HostSetting(;
                name=s["name"], env=Tuple(get(s, "env", [])), default=get(s, "default", nothing)
            ) for s in get(host, "settings", [])
        )
        h[:required_headers]=tuple_of(get(host, "required_headers", []))
        h[:endpoint_env]=tuple_of(get(host, "endpoint_env", []))
        filter!(kv->first(kv) in fieldnames(HostSpec) && last(kv)!==nothing, h)
        kw[:host]=HostSpec(; h...)
    end
    return AccessPolicy(; kw...)
end
const PROVIDERS = let rows=JSON.parse(read(joinpath(@__DIR__, "data", "providers.json"), String))
    table=OrderedDict{String,ProviderDefinition}()
    for row in rows
        kw=Dict{Symbol,Any}(Symbol(k)=>v for (k, v) in row)
        kw[:access]=policy_from_dict(row["access"])
        haskey(kw, :aliases) && (kw[:aliases]=tuple_of(kw[:aliases]))
        table[row["id"]]=ProviderDefinition(; kw...)
    end
    table
end
providers() = collect(values(PROVIDERS))
known_providers() = sort!(collect(keys(PROVIDERS)))
function provider_definition(name)
    id=canonical_provider(name)
    haskey(PROVIDERS, id) || throw(UnknownProviderError(id))
    return PROVIDERS[id]
end

abstract type CredentialValue <: Canonical end
@canonical ApiKey <: CredentialValue begin
    value::String
    kind::String = "api_key"
end
@canonical BearerToken <: CredentialValue begin
    value::String
    expires_at::Maybe{String} = nothing
    kind::String = "bearer_token"
end
@canonical AwsCredentials <: CredentialValue begin
    access_key_id::String
    secret_access_key::String
    session_token::Maybe{String} = nothing
    expires_at::Maybe{String} = nothing
    kind::String = "aws"
end
function validate(c::CredentialValue)
    invoke(validate, Tuple{Canonical}, c)
    expected=if c isa ApiKey
        "api_key"
    elseif c isa BearerToken
        "bearer_token"
    else
        "aws"
    end
    c.kind==expected || throw(ArgumentError("incorrect credential kind discriminator"))
    c isa ApiKey || c.expires_at===nothing || expiry_seconds(c.expires_at)
    return c
end
ApiKey(s::AbstractString) = ApiKey(; value=s)
BearerToken(s::AbstractString; kw...) = BearerToken(; value=s, kw...)
const Credential = Union{AbstractString,CredentialValue,Function}
const StaticCredential = ApiKey
coerce_credential(c::AbstractString) = ApiKey(c)
coerce_credential(c::CredentialValue) = validate(c)
function coerce_credential(c)
    return throw(
        ArgumentError(
            "a credential provider must return a string, ApiKey, BearerToken, or AwsCredentials"
        ),
    )
end
resolve_credential(c::Union{AbstractString,CredentialValue}) = coerce_credential(c)
function resolve_credential(c)
    applicable(c) || throw(ArgumentError("credential must be a value or a zero-argument callable"))
    return coerce_credential(c())
end
Base.show(io::IO, c::CredentialValue) = print(io, nameof(typeof(c)), "(<redacted>)")
Base.show(io::IO, ::MIME"text/plain", c::CredentialValue) = show(io, c)
function from_dict(::Type{CredentialValue}, d::AbstractDict)
    T=get(
        Dict("api_key"=>ApiKey, "bearer_token"=>BearerToken, "aws"=>AwsCredentials),
        get(d, "kind", nothing),
        nothing,
    )
    T === nothing && throw(ArgumentError("unknown credential kind"))
    return from_dict(T, d)
end
SERDE_KINDS["credential"] = CredentialValue
function validate(policy::AccessPolicy)
    isempty(policy.provider) && throw(ArgumentError("access policy needs a provider name"))
    policy.credential_policy in
    ("key", "oauth", "oauth-unless-explicit", "aws-chain", "azure-chain", "gcp-chain") ||
        throw(ArgumentError("unknown credential policy"))
    policy.credential_policy == "oauth" &&
        !isempty(policy.env_keys) &&
        throw(ArgumentError("an OAuth-only policy must not declare environment keys"))
    !isempty(policy.auth_scheme) &&
    all(s -> s in ("bearer", "x-api-key", "api-key", "query-key", "sigv4"), policy.auth_scheme) ||
        throw(ArgumentError("access policy needs known authentication schemes"))
    for fields in
        (policy.env_keys, policy.auth_modes, policy.enterprise_variants, policy.supports.extra)
        all(v -> v isa AbstractString && !isempty(v), fields) ||
            throw(ArgumentError("policy declarations must be non-empty strings"))
    end
    for pair in policy.headers
        length(pair) == 2 && all(v -> v isa AbstractString, pair) ||
            throw(ArgumentError("policy headers must be name/value pairs"))
        occursin(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$", first(pair)) ||
            throw(ArgumentError("invalid policy header name"))
        occursin(r"[\r\n]", last(pair)) &&
            throw(ArgumentError("policy header must not contain a newline"))
    end
    check_json(policy.backend_options)
    if policy.host !== nothing
        host = policy.host
        host.model_in in ("body", "path") || throw(ArgumentError("unknown model placement"))
        host.stream_framing in ("sse", "aws-event-stream") ||
            throw(ArgumentError("unknown stream framing"))
        host.anthropic_version_in == "header" ||
            startswith(host.anthropic_version_in, "body:") ||
            throw(ArgumentError("unknown Anthropic version placement"))
        all(s -> s isa HostSetting && !isempty(s.name), host.settings) ||
            throw(ArgumentError("invalid host setting"))
        length(unique(s.name for s in host.settings)) == length(host.settings) ||
            throw(ArgumentError("duplicate host settings"))
        all(
            p -> length(p) == 2 && last(p) in (s.name for s in host.settings), host.required_headers
        ) || throw(ArgumentError("required header refers to an undeclared setting"))
    end
    if "sigv4" in policy.auth_scheme
        policy.host !== nothing && !empty_optional(policy.host.sigv4_service) ||
            throw(ArgumentError("SigV4 requires a host signing service"))
    end
    endswith(policy.credential_policy, "-chain") &&
        policy.host === nothing &&
        throw(ArgumentError("cloud credential policy requires a host"))
    return policy
end

function select_scheme(policy::AccessPolicy, c::CredentialValue)
    validate(policy)
    validate(c)
    accepted = if c isa ApiKey
        ("bearer", "x-api-key", "api-key", "query-key")
    elseif c isa BearerToken
        ("bearer", "x-api-key")
    else
        ("sigv4",)
    end
    for scheme in (c isa BearerToken ? accepted : policy.auth_scheme)
        scheme in accepted && scheme in policy.auth_scheme && return scheme
    end
    return throw(
        NotConfiguredError(
            "credential kind cannot travel under this access policy"; provider=policy.provider
        ),
    )
end

function looks_like_jwt(value::AbstractString)
    segments = split(value, '.')
    length(segments) == 3 && all(!isempty, segments) || return false
    try
        head = replace(first(segments), '-' => '+', '_' => '/')
        header = JSON.parse(String(base64decode(head * "="^mod(-ncodeunits(head), 4))))
        header isa AbstractDict && haskey(header, "alg")
    catch
        false
    end
end

"""
    looks_like_access_token(text)

The token shape a plain string has, if any (AUTH-2, amended 2026-09-19 and 2026-09-26):
`"JWT"` (JWS compact: Entra, OIDC) or `"Google access token"` (`ya29.`). No key any door
issues has either shape; `nothing` otherwise.
"""
function looks_like_access_token(text::AbstractString)
    startswith(text, "ya29.") && return "Google access token"
    looks_like_jwt(text) && return "JWT"
    return nothing
end
function credential_headers(
    policy::AccessPolicy, credential::CredentialValue; api_key_header="x-api-key"
)
    scheme = select_scheme(policy, credential)
    # AUTH-2 (2026-09-19, 2026-09-26): a token-shaped string on a key header door
    # that also lists bearer travels as bearer, the only reading that can succeed.
    if credential isa ApiKey &&
        scheme in ("api-key", "x-api-key") &&
        "bearer" in policy.auth_scheme &&
        looks_like_access_token(credential.value)!==nothing
        scheme = "bearer"
    end
    if scheme == "bearer"
        return Dict("authorization" => "Bearer " * credential.value)
    elseif scheme == "x-api-key"
        return Dict(api_key_header => credential.value)
    elseif scheme == "api-key"
        return Dict("api-key" => credential.value)
    end
    return Dict{String,String}()
end

# AUTH-10 host templates: a root (scheme and host) and a door path (2026-09-19 D4).
function root_template(host::HostSpec)
    scheme, rest=split(host.base_url, "://"; limit=2)
    return scheme*"://"*first(split(rest, '/'; limit=2))
end
function path_template(host::HostSpec)
    rest=split(host.base_url, "://"; limit=2)[2]
    bits=split(rest, '/'; limit=2)
    return length(bits)==2 ? "/"*bits[2] : ""
end
template_names(t) = Set(m.captures[1] for m in eachmatch(r"\{(\w+)\}", t))
"""Settings an endpoint override makes unnecessary: in the root template only, not in
the door path, not in a required header, not the SigV4 signing region."""
function url_only_settings(host::HostSpec)
    in_root=template_names(root_template(host))
    if "location_host" in in_root
        delete!(in_root, "location_host")
        push!(in_root, "location")
    end
    out=setdiff(in_root, template_names(path_template(host)), Set(last(p) for p in host.required_headers))
    host.sigv4_service===nothing || delete!(out, "region")
    return out
end
"""The first non-empty vendor endpoint variable this door honours (HostSpec.endpoint_env)."""
function endpoint_from_env(host, env)
    (host===nothing || env===nothing) && return nothing, nothing
    for var in host.endpoint_env
        value=strip(get(env, var, ""))
        isempty(value) || return String(value), var
    end
    return nothing, nothing
end
"""
    join_endpoint(endpoint, path)

An endpoint (a URL root) joined with a door's path, appended unless the endpoint already
ends with it or with a leading part of it (AUTH-10, amended 2026-09-19).
"""
function join_endpoint(endpoint::AbstractString, path::AbstractString; provider="")
    uri=try
        HTTP.URI(strip(endpoint))
    catch
        throw(NotConfiguredError("$(isempty(provider) ? "host" : provider): endpoint must be an http(s) URL with a host"; provider=isempty(provider) ? nothing : provider))
    end
    uri.scheme in ("http", "https") && !isempty(uri.host) || throw(NotConfiguredError(
        "$(isempty(provider) ? "host" : provider): endpoint must be an http(s) URL with a host"; provider=isempty(provider) ? nothing : provider))
    (!isempty(uri.query) || !isempty(uri.fragment) || !isempty(uri.userinfo) || occursin(r"[?#]", endpoint)) && throw(NotConfiguredError(
        "$(isempty(provider) ? "host" : provider): endpoint must not carry a query, fragment or userinfo"; provider=isempty(provider) ? nothing : provider))
    given=[String(x) for x in split(uri.path, '/') if !isempty(x)]
    door=[String(x) for x in split(path, '/') if !isempty(x)]
    base=given
    for k in min(length(given), length(door)):-1:1
        if given[(end - k + 1):end]==door[1:k]
            base=given[1:(end - k)]
            break
        end
    end
    joined=join(vcat(base, door), "/")
    authority=uri.host*(isempty(uri.port) ? "" : ":"*uri.port)
    return "$(uri.scheme)://$(authority)"*(isempty(joined) ? "" : "/"*joined)
end
"""
    resolve_settings(policy, given; env, profile, endpoint, sources, unprobed_ok, problems)

A host's settings: the caller's values, then the setting's env variables (when `env` is
given), then the cloud's own configuration (`profile(name)` → `(value, from)`, `value`
`nothing` meaning only a network source could answer), then defaults (AUTH-10). With an
`endpoint`, settings only the URL root needed are optional. `sources` receives each
setting's origin in the AUTH-10 `from` vocabulary; `unprobed_ok` (the offline doctor)
records a network-only setting as `unprobed:<from>`; `problems` collects missing-setting
errors instead of raising them.
"""
function resolve_settings(
    policy::AccessPolicy, given; env=Dict{String,String}(), profile=nothing, endpoint=nothing,
    sources=nothing, unprobed_ok=false, problems=nothing,
)
    host=policy.host
    host === nothing && return Dict{String,String}(given)
    left=Dict{String,String}(String(k)=>String(v) for (k, v) in given)
    relaxed=endpoint===nothing ? Set{String}() : url_only_settings(host)
    record=sources===nothing ? Dict{String,String}() : sources
    out=Dict{String,String}()
    missing=nothing
    for setting in host.settings
        value=pop!(left, setting.name, nothing)
        origin="explicit"
        if value === nothing || isempty(value)
            value=nothing
            if env!==nothing
                for key in setting.env
                    isempty(get(env, key, "")) || (value=env[key]; origin="env:$key"; break)
                end
            end
        end
        unprobed=nothing
        if value===nothing && profile!==nothing
            found=profile(setting.name)
            if found!==nothing
                if found[1]===nothing || isempty(found[1])
                    unprobed=found[2]
                else
                    value, origin=found
                end
            end
        end
        if value===nothing && setting.default!==nothing
            value, origin=setting.default, "default"
        end
        if value===nothing
            setting.name in relaxed && continue
            if unprobed!==nothing && unprobed_ok
                record[setting.name]="unprobed:$unprobed"
                continue
            end
            hint=isempty(setting.env) ? "pass settings=Dict(\"$(setting.name)\" => ...)" : "set $(join(setting.env, " or "))"
            setting.name=="project" && (hint*=", run `gcloud config set project <id>`, or pass settings=Dict(\"project\" => ...)")
            setting.name in url_only_settings(host) && !isempty(host.endpoint_env) &&
                (hint*=", or the endpoint: $(join(host.endpoint_env, " or "))")
            record[setting.name]="missing"
            missing===nothing && (missing=NotConfiguredError(
                "$(policy.provider): setting $(repr(setting.name)) is required and has no default";
                provider=policy.provider, credential_hint=hint,
            ))
            continue
        end
        out[setting.name]=value
        record[setting.name]=origin
    end
    isempty(left) || throw(ArgumentError(
        "$(policy.provider): unknown host setting(s) $(join(sort!(collect(keys(left))), ", ")); known: $(join((s.name for s in host.settings), ", "))"))
    if missing!==nothing
        problems===nothing && throw(missing)
        push!(problems, missing)
    end
    return out
end
function location_host(loc)
    loc == "global" && return "aiplatform.googleapis.com"
    loc in ("us", "eu") && return "aiplatform.$loc.rep.googleapis.com"
    return "$loc-aiplatform.googleapis.com"
end
"""The base URL for these settings; an endpoint replaces the template's root and the
door path is appended unless already present (`join_endpoint`)."""
function render_base_url(host::HostSpec, settings, endpoint=nothing; provider="")
    values=Dict{String,String}(settings)
    for key in ("region", "resource", "location")
        haskey(values, key) &&
            !occursin(r"^[A-Za-z0-9-]+$", values[key]) &&
            throw(NotConfiguredError("host setting $key must be a DNS label"))
    end
    haskey(values, "location") && (values["location_host"]=location_host(values["location"]))
    haskey(values, "project") && (values["project"]=percent_encode(values["project"]))
    url=endpoint===nothing ? host.base_url : path_template(host)
    for (k, v) in values
        url=replace(url, "{$k}"=>v)
    end
    m=match(r"\{(\w+)\}", url)
    m===nothing || throw(NotConfiguredError("host base URL needs setting $(repr(m.captures[1]))"))
    return endpoint===nothing ? url : join_endpoint(endpoint, url; provider)
end
host_base_url(host::HostSpec, settings) = render_base_url(host, settings)
