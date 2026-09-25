# HTTP.jl's debug records can include full requests, Authorization headers and
# token-exchange bodies. Filter those modules at the logging boundary without
# disabling application logs inside a transport or live-session callback.
struct TransportLogger{L<:Logging.AbstractLogger} <: Logging.AbstractLogger
    parent::L
end
function is_http_module(mod)
    mod isa Module || return false
    while true
        mod === HTTP && return true
        owner = parentmodule(mod)
        owner === mod && return false
        mod = owner
    end
end
Logging.min_enabled_level(logger::TransportLogger) = Logging.min_enabled_level(logger.parent)
Logging.catch_exceptions(logger::TransportLogger) = Logging.catch_exceptions(logger.parent)
function Logging.shouldlog(logger::TransportLogger, level, mod, group, id)
    return !is_http_module(mod) && Logging.shouldlog(logger.parent, level, mod, group, id)
end
function Logging.handle_message(
    logger::TransportLogger, level, message, mod, group, id, file, line; kwargs...
)
    is_http_module(mod) && return nothing
    return Logging.handle_message(
        logger.parent, level, message, mod, group, id, file, line; kwargs...
    )
end
function with_transport_logger(f)
    failure = nothing
    try
        return Logging.with_logger(f, TransportLogger(Logging.current_logger()))
    catch error
        failure = if error isa Union{LM15Error,InterruptException}
            error
        else
            TransportError("HTTP operation failed")
        end
    end
    # Throw outside the catch block. Otherwise Julia displays the original
    # HTTP.RequestError/ConnectError as a cause, including keys in headers or
    # query strings, even though our own error message is redacted.
    return throw(failure)
end
http_request(args...; kwargs...) = with_transport_logger(() -> HTTP.request(args...; kwargs...))
function http_open(f, args...; kwargs...)
    return with_transport_logger(() -> HTTP.open(io -> f(io), args...; kwargs...))
end
function websocket_open(f, args...; kwargs...)
    return with_transport_logger(() -> HTTP.WebSockets.open(ws -> f(ws), args...; kwargs...))
end

"""A transport implements `open_response(f, transport, request)` for buffered and streaming calls."""
abstract type AbstractTransport end

"""
    HTTPTransport(; connect_timeout=30, read_timeout=120)

Direct HTTP with bounded connection/read waits, no automatic retries, and no
redirects carrying credentials to another endpoint. Times are positive whole
seconds, matching HTTP.jl's timeout granularity; fractional values are refused.
"""
struct HTTPTransport <: AbstractTransport
    connect_timeout::Int
    read_timeout::Int

    function HTTPTransport(; connect_timeout=30, read_timeout=120)
        for (name, value) in ((:connect_timeout, connect_timeout), (:read_timeout, read_timeout))
            value isa Real && !(value isa Bool) && isfinite(value) && value > 0 ||
                throw(ArgumentError("$name must be a finite positive number of seconds"))
        end
        return new(integer_value(connect_timeout), integer_value(read_timeout))
    end
end

const DEFAULT_HTTP_TRANSPORT = HTTPTransport()

"""
    open_response(f, transport, request::WireRequest)

Call `f(head::HttpResponse, body::IO)` while the body is open. The transport owns
and closes `body`, including when `f` throws. `head.body` is empty for a streaming
transport; status and headers are available before reading any bytes.

Implement this method for a custom `AbstractTransport`. A callable returning a
buffered `HttpResponse` remains accepted for fixtures and small responses.
"""
function open_response(f, transport::HTTPTransport, request::WireRequest)
    primary = nothing
    returned = nothing
    try
        # INV-053: ask for identity (HTTP.jl otherwise advertises gzip, and a
        # streamed read is not decompressed: the parser saw compressed bytes
        # from every real provider), and never let HTTP.jl decode silently.
        headers = any(h -> lowercase(String(first(h))) == "accept-encoding", request.headers) ?
            request.headers : vcat(collect(request.headers), ["Accept-Encoding" => "identity"])
        http_open(
            request.method,
            request.url,
            headers;
            decompress=false,
            status_exception=false,
            redirect=false,
            retry=false,
            connect_timeout=transport.connect_timeout,
            readtimeout=transport.read_timeout,
        ) do io
            try
                write(io, request.body)
                HTTP.closewrite(io)
                HTTP.startread(io)
            catch
                throw(TransportError("could not open the provider response"))
            end
            head = HttpResponse(;
                status=Int(io.message.status),
                headers=Pair{String,String}[
                    String(k) => String(v) for (k, v) in io.message.headers
                ],
            )
            try
                returned = f(head, io)
            catch error
                primary = error
                rethrow()
            end
        end
    catch error
        # HTTP.jl may wrap a callback exception in RequestError. The caller's
        # original exception wins over both that wrapper and connection cleanup.
        primary === nothing || throw(primary)
        error isa LM15Error && rethrow()
        error isa InterruptException && rethrow()
        throw(TransportError("provider connection failed"))
    end
    return returned
end

function open_response(f, transport::AbstractTransport, request::WireRequest)
    return throw(
        ArgumentError(
            "$(typeof(transport)) must implement LM15.open_response(f, transport, request)"
        ),
    )
end

function open_response(f, transport, request::WireRequest)
    applicable(transport, request) || throw(
        ArgumentError("transport must implement open_response or be callable with a WireRequest"),
    )
    head = transport(request)
    head isa HttpResponse || throw(ArgumentError("transport callback must return HttpResponse"))
    body = IOBuffer(head.body; read=true, write=false)
    try
        f(head, body)
    finally
        close(body)
    end
end

client_transport(client) = client.transport === nothing ? DEFAULT_HTTP_TRANSPORT : client.transport
