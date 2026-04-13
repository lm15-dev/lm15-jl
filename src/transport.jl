# HTTP transport using stdlib Downloads + minimal SSE parser.

struct HttpRequest
    method::String
    url::String
    headers::Dict{String,String}
    params::Dict{String,String}
    body::Union{Vector{UInt8},Nothing}
    timeout::Union{Float64,Nothing}
end

HttpRequest(; method, url, headers=Dict{String,String}(), params=Dict{String,String}(),
              body=nothing, timeout=nothing) =
    HttpRequest(method, url, headers, params, body, timeout)

struct HttpResponse
    status::Int
    headers::Dict{String,String}
    body::Vector{UInt8}
end

text(r::HttpResponse) = String(r.body)

function build_url(req::HttpRequest)
    isempty(req.params) && return req.url
    qs = join(["$k=$v" for (k, v) in req.params], "&")
    "$(req.url)?$qs"
end

"""Execute a synchronous HTTP request using Downloads.request."""
function http_request(req::HttpRequest)::HttpResponse
    url = build_url(req)
    input = req.body !== nothing ? IOBuffer(req.body) : nothing
    output = IOBuffer()

    kwargs = Dict{Symbol,Any}()
    kwargs[:method] = req.method
    kwargs[:output] = output
    if input !== nothing
        kwargs[:input] = input
    end
    if req.timeout !== nothing
        kwargs[:timeout] = req.timeout
    end

    hdrs = [k => v for (k, v) in req.headers]

    try
        resp = Downloads.request(url; headers=hdrs, kwargs...)
        status = resp.status
        resp_headers = Dict{String,String}()
        for (k, v) in resp.headers
            resp_headers[lowercase(k)] = v
        end
        return HttpResponse(status, resp_headers, take!(output))
    catch e
        throw(TransportError(string(e)))
    end
end

"""Execute a streaming HTTP request — returns raw response body as IO."""
function http_stream(req::HttpRequest)
    url = build_url(req)
    input = req.body !== nothing ? IOBuffer(req.body) : nothing
    output = IOBuffer()

    kwargs = Dict{Symbol,Any}()
    kwargs[:method] = req.method
    kwargs[:output] = output
    if input !== nothing
        kwargs[:input] = input
    end
    if req.timeout !== nothing
        kwargs[:timeout] = req.timeout
    end

    hdrs = [k => v for (k, v) in req.headers]

    try
        resp = Downloads.request(url; headers=hdrs, kwargs...)
        if resp.status >= 400
            throw(TransportError("HTTP $(resp.status): $(String(take!(output)))"))
        end
        return IOBuffer(take!(output))
    catch e
        e isa TransportError && rethrow()
        throw(TransportError(string(e)))
    end
end

# ── SSE Parser ──────────────────────────────────────────────────────

struct SSEEvent
    event::Union{String,Nothing}
    data::String
end

"""Parse SSE events from an IO stream."""
function parse_sse(io::IO)::Vector{SSEEvent}
    events = SSEEvent[]
    event_name = nothing
    data_lines = String[]

    for line in eachline(io)
        line = rstrip(line, ['\r', '\n'])

        if isempty(line)
            if !isempty(data_lines)
                push!(events, SSEEvent(event_name, join(data_lines, "\n")))
                data_lines = String[]
            end
            event_name = nothing
            continue
        end

        startswith(line, ':') && continue

        if startswith(line, "event:")
            event_name = strip(line[7:end])
        elseif startswith(line, "data:")
            push!(data_lines, lstrip(line[6:end]))
        end
    end

    # Flush remaining
    if !isempty(data_lines)
        push!(events, SSEEvent(event_name, join(data_lines, "\n")))
    end

    events
end
