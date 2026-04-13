# ── Error hierarchy ─────────────────────────────────────────────────

abstract type LM15Error <: Exception end

struct TransportError <: LM15Error; msg::String; end
struct ProviderError <: LM15Error; msg::String; end
struct AuthError <: LM15Error; msg::String; end
struct BillingError <: LM15Error; msg::String; end
struct RateLimitError <: LM15Error; msg::String; end
struct InvalidRequestError <: LM15Error; msg::String; end
struct ContextLengthError <: LM15Error; msg::String; end
struct TimeoutError <: LM15Error; msg::String; end
struct ServerError <: LM15Error; msg::String; end
struct UnsupportedModelError <: LM15Error; msg::String; end
struct UnsupportedFeatureError <: LM15Error; msg::String; end

Base.showerror(io::IO, e::LM15Error) = print(io, typeof(e), ": ", e.msg)

function map_http_error(status::Int, message::String)
    if status in (401, 403); AuthError(message)
    elseif status == 402; BillingError(message)
    elseif status == 429; RateLimitError(message)
    elseif status in (408, 504); TimeoutError(message)
    elseif status in (400, 404, 409, 413, 422); InvalidRequestError(message)
    elseif 500 <= status <= 599; ServerError(message)
    else; ProviderError(message)
    end
end

function canonical_error_code(e::LM15Error)
    e isa ContextLengthError && return "context_length"
    e isa AuthError && return "auth"
    e isa BillingError && return "billing"
    e isa RateLimitError && return "rate_limit"
    e isa InvalidRequestError && return "invalid_request"
    e isa TimeoutError && return "timeout"
    e isa ServerError && return "server"
    return "provider"
end

function error_for_code(code::String, message::String)
    code == "auth" && return AuthError(message)
    code == "billing" && return BillingError(message)
    code == "rate_limit" && return RateLimitError(message)
    code == "invalid_request" && return InvalidRequestError(message)
    code == "context_length" && return ContextLengthError(message)
    code == "timeout" && return TimeoutError(message)
    code == "server" && return ServerError(message)
    return ProviderError(message)
end

is_transient(e) = e isa RateLimitError || e isa TimeoutError || e isa ServerError || e isa TransportError
