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
    Timeouts(; connect=10, read=600, write=600, pool=600)

The ratified connection budget (spec/vocabularies.md § Connection budget), in seconds,
per operation (not total request duration): establishing a connection, waiting for the
next reply bytes, sending request bytes, waiting for a free connection slot.
"""
struct Timeouts
    connect::Float64
    read::Float64
    write::Float64
    pool::Float64
    function Timeouts(; connect=10, read=600, write=600, pool=600)
        for (name, v) in ((:connect, connect), (:read, read), (:write, write), (:pool, pool))
            v isa Real && !(v isa Bool) && isfinite(v) && v > 0 ||
                throw(ArgumentError("$name must be a finite positive number of seconds"))
        end
        return new(Float64(connect), Float64(read), Float64(write), Float64(pool))
    end
end

"""
    HTTPTransport(; timeouts=Timeouts(), max_connections=100)

Direct HTTP with the connection budget, a connection cap, no automatic retries, no
redirects carrying credentials to another endpoint, and replies decoded only as INV-053
allows. HTTP.jl measures its timeouts in whole seconds, so fractional values are rounded
up (never shortened). `connect_timeout=` and `read_timeout=` remain accepted.
"""
struct HTTPTransport <: AbstractTransport
    timeouts::Timeouts
    max_connections::Int
    pool::Base.RefValue{Any}
end
function HTTPTransport(; timeouts=Timeouts(), max_connections=100, connect_timeout=nothing, read_timeout=nothing)
    timeouts isa Timeouts || throw(ArgumentError("timeouts must be a Timeouts"))
    if connect_timeout !== nothing || read_timeout !== nothing
        timeouts = Timeouts(; connect=something(connect_timeout, timeouts.connect),
            read=something(read_timeout, timeouts.read), write=timeouts.write, pool=timeouts.pool)
    end
    max_connections isa Integer && !(max_connections isa Bool) && max_connections > 0 ||
        throw(ArgumentError("max_connections must be a positive integer"))
    return HTTPTransport(timeouts, Int(max_connections), Ref{Any}(nothing))
end
Base.getproperty(t::HTTPTransport, name::Symbol) =
    name === :connect_timeout ? ceil(Int, getfield(t, :timeouts).connect) :
    name === :read_timeout ? ceil(Int, getfield(t, :timeouts).read) : getfield(t, name)
# The pool is created at first use, not at precompilation (it holds locks).
function transport_pool(t::HTTPTransport)
    t.pool[] === nothing && (t.pool[] = HTTP.Pool(t.max_connections))
    return t.pool[]
end
const DEFAULT_HTTP_TRANSPORT = HTTPTransport()

# INV-053: a reply that arrives Content-Encoding gzip, x-gzip or deflate is inflated
# (incrementally on a stream) before any decoding; any other coding is a transport
# error naming it. The codings are undone in reverse of the order listed.
function content_codings(headers)
    codings = String[]
    for (k, v) in headers
        lowercase(k) == "content-encoding" || continue
        for token in split(v, ',')
            coding = lowercase(strip(token))
            (isempty(coding) || coding == "identity") && continue
            coding in ("gzip", "x-gzip", "deflate") || throw(TransportError(
                "response is Content-Encoding: $(repr(coding)), which lm15 cannot decode (it asked for identity; gzip and deflate are decoded, br and zstd are not)"))
            push!(codings, coding)
        end
    end
    return reverse!(codings)
end
"""An IO that yields a few already-read bytes before its source (deflate sniffing)."""
mutable struct PrefixedIO{T<:IO} <: IO
    prefix::Vector{UInt8}
    io::T
end
Base.eof(p::PrefixedIO) = isempty(p.prefix) && eof(p.io)
Base.bytesavailable(p::PrefixedIO) = length(p.prefix) + bytesavailable(p.io)
Base.read(p::PrefixedIO, ::Type{UInt8}) = isempty(p.prefix) ? read(p.io, UInt8) : popfirst!(p.prefix)
function Base.unsafe_read(p::PrefixedIO, ptr::Ptr{UInt8}, n::UInt)
    k = Int(min(n, UInt(length(p.prefix))))
    for i in 1:k
        unsafe_store!(ptr, p.prefix[i], i)
    end
    deleteat!(p.prefix, 1:k)
    k < n && unsafe_read(p.io, ptr + k, n - UInt(k))
    return nothing
end
Base.isopen(p::PrefixedIO) = !isempty(p.prefix) || isopen(p.io)
Base.close(p::PrefixedIO) = close(p.io)
function inflating(io::IO, coding)
    coding in ("gzip", "x-gzip") && return CodecZlib.GzipDecompressorStream(io)
    # deflate is normally zlib-wrapped (RFC 1950); some servers send raw deflate.
    # Two bytes decide: CMF 8 (deflate) and a header checksum divisible by 31.
    head = UInt8[]
    while length(head) < 2 && !eof(io)
        push!(head, read(io, UInt8))
    end
    wrapped = length(head) == 2 && (head[1] & 0x0f) == 0x08 && (UInt16(head[1]) << 8 | head[2]) % 31 == 0
    source = PrefixedIO(head, io)
    return wrapped ? CodecZlib.ZlibDecompressorStream(source) : CodecZlib.DeflateDecompressorStream(source)
end
function decoded_body(io::IO, headers)
    for coding in content_codings(headers)
        io = inflating(io, coding)
    end
    return io
end
function decoded_head(head::HttpResponse)
    codings = content_codings(head.headers)
    isempty(codings) && return head
    body = try
        read(decoded_body(IOBuffer(head.body), head.headers))
    catch e
        e isa InterruptException && rethrow()
        throw(TransportError("the compressed reply could not be decoded"))
    end
    return HttpResponse(; status=head.status, headers=head.headers, body)
end

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
    budget = transport.timeouts
    try
        # INV-053: ask for identity (HTTP.jl otherwise advertises gzip, and a
        # streamed read is not decompressed), and never let HTTP.jl decode silently;
        # what arrives compressed anyway is decoded here or refused by name.
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
            connect_timeout=ceil(Int, budget.connect),
            readtimeout=ceil(Int, budget.read),
            pool=transport_pool(transport),
        ) do io
            # The write budget: a send that stalls longer is abandoned.
            stalled = Ref(false)
            watchdog = Timer(budget.write) do _
                stalled[] = true
                try
                    close(io)
                catch
                end
            end
            try
                write(io, request.body)
                HTTP.closewrite(io)
            catch
                throw(TransportError(stalled[] ? "sending the request took longer than the $(budget.write)-second write budget" :
                    "could not send the request"))
            finally
                close(watchdog)
            end
            try
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
            body = decoded_body(io, head.headers)
            try
                returned = f(head, body)
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
    head = decoded_head(head)
    body = IOBuffer(head.body; read=true, write=false)
    try
        f(head, body)
    finally
        close(body)
    end
end

client_transport(client) = client.transport === nothing ? DEFAULT_HTTP_TRANSPORT : client.transport
