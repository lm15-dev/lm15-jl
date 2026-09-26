abstract type LM15Error <: Exception end
abstract type ProviderError <: LM15Error end
abstract type ConfigurationError <: LM15Error end
abstract type CapabilityError <: LM15Error end
abstract type InvalidRequestError <: ProviderError end
abstract type AuthError <: ProviderError end
abstract type NotConfiguredError <: ConfigurationError end
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
    feature::Union{Nothing,String} = nothing
    partial::Any = nothing
    part_index::Union{Nothing,Int} = nothing
    cleanup_errors::Vector{Exception} = Exception[]
    # Error diagnostics (docs/error-diagnostics.md, 2026-09-19): an immutable
    # lowercase-name → ordered string-array snapshot of the closed header set.
    rate_limit_headers::Any = nothing
    env_keys::Tuple = ()
    # AUTH-1 provenance: where a rejected credential came from, never its value.
    credential_origin::Union{Nothing,String} = nothing
    # AUTH-24 (AuthOperationError): sanitized lifecycle metadata.
    reason::Union{Nothing,String} = nothing
    stage::Union{Nothing,String} = nothing
    commit_state::Union{Nothing,String} = nothing
    recovery::Union{Nothing,String} = nothing
    operation::Union{Nothing,String} = nothing
    connection_id::Union{Nothing,String} = nothing
    attempt_id::Union{Nothing,String} = nothing
    method_id::Union{Nothing,String} = nothing
    # Live collection limits (changes/2026-09-15-live-collection-limits.md).
    limit::Union{Nothing,String} = nothing
    maximum::Union{Nothing,Int} = nothing
    retained_bytes::Int = 0
    partial_events::Tuple = ()
    rejected_event::Any = nothing
end

const ERROR_TYPES = Dict{String,DataType}()
const ERROR_CODES = Dict{DataType,String}()
for (name, parent, code) in (
    (:GenericProviderError, :ProviderError, "provider"),
    (:TransportError, :LM15Error, "transport"),
    (:LockTimeoutError, :LM15Error, "lock_timeout"),
    (:StreamAssemblyError, :LM15Error, "stream_assembly"),
    (:AuthOperationError, :LM15Error, "auth_operation"),
    (:CollectionLimitError, :LM15Error, "collection_limit"),
    (:DefaultNotConfiguredError, :NotConfiguredError, "not_configured"),
    (:MissingCredentialError, :NotConfiguredError, "not_configured"),
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
NotConfiguredError(message::AbstractString=""; kwargs...) = DefaultNotConfiguredError(message; kwargs...)
ConfigurationError(message::AbstractString=""; kwargs...) = NotConfiguredError(message; kwargs...)
CapabilityError(message::AbstractString=""; kwargs...) = UnsupportedFeatureError(message; kwargs...)
ERROR_TYPES["auth"] = AuthError
ERROR_TYPES["invalid_request"] = InvalidRequestError
ERROR_TYPES["not_configured"] = NotConfiguredError
ERROR_CODES[AuthError] = "auth"
ERROR_CODES[InvalidRequestError] = "invalid_request"
ERROR_CODES[NotConfiguredError] = "not_configured"
# AUTH-24: the closed vocabularies of a managed-auth lifecycle failure.
const AUTH_OPERATION_REASONS = (
    "interaction_required", "method_unavailable", "connection_exists",
    "login_in_progress", "login_required", "connection_changed", "login_denied",
    "login_expired", "invalid_login_state", "attempt_unavailable", "indeterminate",
    "storage_unavailable", "unsupported_store_version", "selection_mismatch",
    "credential_rejected",
)
const AUTH_OPERATION_STAGES = (
    "discovery", "reservation", "interaction", "authorization", "polling", "exchange",
    "persistence", "resolution", "renewal", "verification", "catalog", "dispatch",
)
const AUTH_OPERATION_RECOVERIES = (
    "provide_input", "choose_method", "resume_attempt", "inspect_attempt", "restart_login",
    "select_connection", "repair_storage", "operator_action", "none",
)
const AUTH_COMMIT_STATES = ("not_committed", "committed", "unknown")
"""
    AuthOperationError(message; reason, stage="resolution", commit_state="not_committed", recovery="none", ...)

A managed-auth lifecycle operation failed locally (AUTH-24). Root-level: nothing here is a
provider HTTP reply and nothing is safe to retry blindly. Programs match on `reason`.
"""
function auth_operation_error(
    message::AbstractString="";
    reason, stage="resolution", commit_state="not_committed", recovery="none", kwargs...,
)
    reason in AUTH_OPERATION_REASONS || throw(ArgumentError("AuthOperationError: unknown reason $(repr(reason))"))
    stage in AUTH_OPERATION_STAGES || throw(ArgumentError("AuthOperationError: unknown stage $(repr(stage))"))
    commit_state in AUTH_COMMIT_STATES ||
        throw(ArgumentError("AuthOperationError: unknown commit_state $(repr(commit_state))"))
    recovery in AUTH_OPERATION_RECOVERIES ||
        throw(ArgumentError("AuthOperationError: unknown recovery $(repr(recovery))"))
    return AuthOperationError(ErrorMetadata(; message=String(message), reason, stage, commit_state, recovery, kwargs...))
end
const RequestTimeoutError = TimeoutError

function NotConfiguredError(provider::AbstractString, message::AbstractString, hint::AbstractString)
    return NotConfiguredError(message; provider=String(provider), credential_hint=String(hint))
end
error_code(e::LM15Error) = ERROR_CODES[typeof(e)]
class_name(e::LM15Error) = string(nameof(typeof(e)))
class_name(::GenericProviderError) = "ProviderError"
class_name(::DefaultAuthError) = "AuthError"
class_name(::DefaultInvalidRequestError) = "InvalidRequestError"
class_name(::DefaultNotConfiguredError) = "NotConfiguredError"
function Base.getproperty(e::LM15Error, name::Symbol)
    name === :metadata && return getfield(e, :metadata)
    name === :code && return error_code(e)
    if e isa CollectionLimitError
        name === :retained_events && return length(getfield(e, :metadata).partial_events)
        name === :partial && return partial_turn(e)
    end
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
    e.request_id === nothing || print(io, " [request ", e.request_id, "]")
    pieces = String[]
    e.retry_after === nothing || push!(pieces, "Retry advice: $(e.retry_after) seconds (not a guarantee).")
    if e.rate_limit_headers !== nothing && !isempty(e.rate_limit_headers)
        raw = JSON.serialize(JSONObject(k => collect(e.rate_limit_headers[k]) for k in sort!(collect(keys(e.rate_limit_headers)))))
        ncodeunits(raw) > 2048 && (raw = first(raw, 2048) * "... [full retained values in rate_limit_headers]")
        push!(pieces, "Provider rate-limit headers (raw; advisory): " * escape_string(raw))
    end
    isempty(pieces) || print(io, "\n\n  ", join(pieces, "\n  "))
    e.credential_origin === nothing || print(io, "\n  credential came from: ", e.credential_origin)
    return e.credential_hint === nothing || print(io, "\nTo fix: ", e.credential_hint)
end
Base.show(io::IO, e::LM15Error) = showerror(io, e)
function retryable(e::LM15Error)
    return e isa Union{TransportError,LockTimeoutError,RateLimitError,TimeoutError,ServerError}
end
retryable(::Exception) = false
"""Rebuild `e` with some metadata replaced (errors are immutable values)."""
function with_metadata(e::LM15Error; kwargs...)
    kw = Dict{Symbol,Any}(n => getfield(e.metadata, n) for n in fieldnames(ErrorMetadata))
    merge!(kw, Dict{Symbol,Any}(kwargs))
    return typeof(e)(ErrorMetadata(; kw...))
end
"""The partial live turn a CollectionLimitError preserved, materialized on demand."""
function partial_turn(e::CollectionLimitError)
    t = materialize_turn(e.partial_events)
    return Turn(; (n => getfield(t, n) for n in fieldnames(Turn))..., ended_by="incomplete")
end
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
function unsupported(provider, feature; path=nothing)
    return throw(
        UnsupportedFeatureError(
            "$feature cannot be carried by this provider"; provider=provider, feature=path
        ),
    )
end
# A refusal about one addressable field (MAP-13 4b): `feature` is its path.
refuse(provider, path, message) =
    throw(UnsupportedFeatureError("$provider: $path: $message"; provider=provider, feature=path))
