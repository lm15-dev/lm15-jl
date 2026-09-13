abstract type LM15Error <: Exception end
abstract type ProviderError <: LM15Error end
abstract type ConfigurationError <: LM15Error end
abstract type CapabilityError <: LM15Error end
abstract type InvalidRequestError <: ProviderError end
abstract type AuthError <: ProviderError end
const InvalidRequestFailure = InvalidRequestError

Base.@kwdef struct ErrorMetadata
    message::String = ""
    provider::Union{Nothing,String} = nothing
    provider_code::Union{Nothing,String} = nothing
    status::Union{Nothing,Int} = nothing
    request_id::Union{Nothing,String} = nothing
    retry_after::Union{Nothing,Float64} = nothing
    credential_hint::Union{Nothing,String} = nothing
    model::String = ""
    providers::Tuple = ()
    path::String = ""
    lock_path::String = ""
    partial::Any = nothing
    part_index::Union{Nothing,Int} = nothing
    cleanup_errors::Vector{Exception} = Exception[]
end

const ERROR_TYPES = Dict{String,DataType}()
const ERROR_CODES = Dict{DataType,String}()
for (name, parent, code) in (
    (:GenericProviderError, :ProviderError, "provider"),
    (:TransportError, :LM15Error, "transport"),
    (:LockTimeoutError, :LM15Error, "lock_timeout"),
    (:StreamAssemblyError, :LM15Error, "stream_assembly"),
    (:NotConfiguredError, :ConfigurationError, "not_configured"),
    (:UnknownModelError, :ConfigurationError, "unknown_model"),
    (:AmbiguousModelError, :ConfigurationError, "ambiguous_model"),
    (:UnsupportedFeatureError, :CapabilityError, "unsupported_feature"),
    (:DefaultAuthError, :AuthError, "auth"),
    (:BillingError, :ProviderError, "billing"),
    (:RateLimitError, :ProviderError, "rate_limit"),
    (:DefaultInvalidRequestError, :InvalidRequestError, "invalid_request"),
    (:ContextLengthError, :InvalidRequestError, "context_length"),
    (:UnsupportedModelError, :InvalidRequestError, "unsupported_model"),
    (:TimeoutError, :ProviderError, "timeout"),
    (:ServerError, :ProviderError, "server"),
)
    @eval begin
        struct $name <: $parent
            metadata::ErrorMetadata
        end
        $name(message::AbstractString=""; kwargs...) =
            $name(ErrorMetadata(; message=String(message), kwargs...))
        ERROR_TYPES[$code] = $name
        ERROR_CODES[$name] = $code
    end
end
# Public family names remain constructible while narrower errors subtype them.
# This gives ordinary Julia `error isa AuthError` / `isa InvalidRequestError`
# the same catch semantics as the contract's class hierarchy.
AuthError(message::AbstractString=""; kwargs...) = DefaultAuthError(message; kwargs...)
function InvalidRequestError(message::AbstractString=""; kwargs...)
    return DefaultInvalidRequestError(message; kwargs...)
end
ProviderError(message::AbstractString=""; kwargs...) = GenericProviderError(message; kwargs...)
ConfigurationError(message::AbstractString=""; kwargs...) = NotConfiguredError(message; kwargs...)
CapabilityError(message::AbstractString=""; kwargs...) = UnsupportedFeatureError(message; kwargs...)
ERROR_TYPES["auth"] = AuthError
ERROR_TYPES["invalid_request"] = InvalidRequestError
ERROR_CODES[AuthError] = "auth"
ERROR_CODES[InvalidRequestError] = "invalid_request"
const RequestTimeoutError = TimeoutError

function NotConfiguredError(provider::AbstractString, message::AbstractString, hint::AbstractString)
    return NotConfiguredError(message; provider=String(provider), credential_hint=String(hint))
end
error_code(e::LM15Error) = ERROR_CODES[typeof(e)]
class_name(e::LM15Error) = string(nameof(typeof(e)))
class_name(::GenericProviderError) = "ProviderError"
class_name(::DefaultAuthError) = "AuthError"
class_name(::DefaultInvalidRequestError) = "InvalidRequestError"
function Base.getproperty(e::LM15Error, name::Symbol)
    name === :metadata && return getfield(e, :metadata)
    name === :code && return error_code(e)
    return getproperty(getfield(e, :metadata), name)
end
function Base.propertynames(::LM15Error, private::Bool=false)
    return if private
        (:metadata, :code, fieldnames(ErrorMetadata)...)
    else
        (:code, fieldnames(ErrorMetadata)...)
    end
end
function Base.showerror(io::IO, e::LM15Error)
    print(io, class_name(e), ": ", e.message)
    e.provider === nothing || print(io, " (", e.provider, ")")
    e.status === nothing || print(io, " [HTTP ", e.status, "]")
    return e.credential_hint === nothing || print(io, "\nTo fix: ", e.credential_hint)
end
Base.show(io::IO, e::LM15Error) = showerror(io, e)
function retryable(e::LM15Error)
    return e isa Union{TransportError,LockTimeoutError,RateLimitError,TimeoutError,ServerError}
end
retryable(::Exception) = false
function error_for_code(code, message=""; kwargs...)
    return get(ERROR_TYPES, code, GenericProviderError)(message; kwargs...)
end
function map_http_error(status, message; kwargs...)
    T = if status in (401, 403)
        AuthError
    elseif status == 402
        BillingError
    elseif status in (408, 504)
        TimeoutError
    elseif status == 429
        RateLimitError
    elseif status in (400, 404, 409, 413, 422)
        InvalidRequestError
    elseif 500 <= status <= 599
        ServerError
    else
        GenericProviderError
    end
    return T(message; status=Int(status), kwargs...)
end
struct UnknownProviderError <: Exception
    provider::String
end
function Base.showerror(io::IO, e::UnknownProviderError)
    return print(
        io, "Unknown provider ", repr(e.provider), "; known: ", join(known_providers(), ", ")
    )
end
function unsupported(provider, feature)
    return throw(
        UnsupportedFeatureError("$feature cannot be carried by this provider"; provider=provider)
    )
end
