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
end
canonical_provider(name::AbstractString) = replace(String(name), '_'=>'-')
function policy_from_dict(d)
    kw=Dict{Symbol,Any}(Symbol(k)=>v for (k, v) in d)
    kw[:supports]=EndpointSupport(; (Symbol(k)=>v for (k, v) in get(d, "supports", obj()))...)
    for key in (:auth_modes, :enterprise_variants, :env_keys, :auth_scheme, :headers)
        haskey(kw, key) && (kw[key]=Tuple(kw[key]))
    end
    if get(kw, :host, nothing) !== nothing
        host=kw[:host]
        h=Dict{Symbol,Any}(Symbol(k)=>v for (k, v) in host)
        h[:settings]=Tuple(
            HostSetting(;
                name=s["name"], env=Tuple(get(s, "env", [])), default=get(s, "default", nothing)
            ) for s in get(host, "settings", [])
        )
        h[:required_headers]=Tuple(get(host, "required_headers", []))
        kw[:host]=HostSpec(; h...)
    end
    return AccessPolicy(; kw...)
end
const PROVIDERS = let rows=JSON.parse(read(joinpath(@__DIR__, "data", "providers.json"), String))
    table=OrderedDict{String,ProviderDefinition}()
    for row in rows
        kw=Dict{Symbol,Any}(Symbol(k)=>v for (k, v) in row)
        kw[:access]=policy_from_dict(row["access"])
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

function credential_headers(
    policy::AccessPolicy, credential::CredentialValue; api_key_header="x-api-key"
)
    scheme = select_scheme(policy, credential)
    if credential isa ApiKey &&
        scheme in ("api-key", "x-api-key") &&
        "bearer" in policy.auth_scheme &&
        looks_like_jwt(credential.value)
        throw(
            NotConfiguredError(
                "this credential looks like a bearer token; wrap it in BearerToken(token) instead of passing a key string";
                provider=policy.provider,
            ),
        )
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

function resolve_settings(policy::AccessPolicy, given; env=Dict{String,String}())
    host=policy.host
    host === nothing && return Dict{String,String}(given)
    out=Dict{String,String}()
    for setting in host.settings
        value=get(given, setting.name, nothing)
        if value === nothing || isempty(value)
            value=nothing
            for key in setting.env
                isempty(get(env, key, "")) || (value=env[key]; break)
            end
        end
        value === nothing && (value=setting.default)
        value === nothing && throw(
            NotConfiguredError(
                "required host setting $(setting.name) is missing"; provider=policy.provider
            ),
        )
        out[setting.name]=value
    end
    all(k->haskey(out, k), keys(given)) || throw(ArgumentError("unknown host setting"))
    return out
end
function host_base_url(host::HostSpec, settings)
    values=copy(settings)
    for key in ("region", "resource", "location")
        haskey(values, key) &&
            !occursin(r"^[A-Za-z0-9-]+$", values[key]) &&
            throw(NotConfiguredError("host setting $key must be a DNS label"))
    end
    if haskey(values, "location")
        loc=values["location"]
        values["location_host"]=if loc == "global"
            "aiplatform.googleapis.com"
        elseif loc in ("us", "eu")
            "aiplatform.$loc.rep.googleapis.com"
        else
            "$loc-aiplatform.googleapis.com"
        end
    end
    haskey(values, "project") && (values["project"]=percent_encode(values["project"]))
    url=host.base_url
    for (k, v) in values
        url=replace(url, "{$k}"=>v)
    end
    occursin('{', url) && throw(NotConfiguredError("unresolved host URL setting"))
    return url
end
