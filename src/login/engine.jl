# The machinery every provider flow runs on (spec/auth-managed.md AUTH-18 connected
# login, device polling, browser/OAuth protections; AUTH-20 bounded exchanges and
# uncertainty; AUTH-21 what a diagnostic may carry). Provider flows describe their
# protocol; this file supplies the parts that must be identical for all of them.
# Nothing here knows a provider's URL, client id or token shape.

const ATTEMPT_LIFETIME_S = 15 * 60.0      # AUTH-18
const EXCHANGE_TIMEOUT_S = 30.0           # AUTH-20.5
const DEVICE_DEFAULT_INTERVAL_S = 5.0     # RFC 8628 §3.2
const DEVICE_SLOW_DOWN_STEP_S = 5.0       # RFC 8628 §3.5
const AUTH_RESPONSE_LIMIT = 1024 * 1024   # AUTH-18: 1 MiB auth HTTP body
const CALLBACK_TARGET_LIMIT = 8 * 1024    # AUTH-18: 8 KiB request target
const CALLBACK_HEADER_LIMIT = 32 * 1024   # AUTH-18: 32 KiB callback headers

"""
    LoginCancelled

The caller, the UI or `cancel_login` stopped an attempt. Not an LM15Error: the conformance
outcome is `cancelled` (AUTH-24). A UI throws it (or an `InterruptException`) to cancel.
"""
struct LoginCancelled <: Exception
    message::String
end
LoginCancelled() = LoginCancelled("login cancelled")
Base.showerror(io::IO, e::LoginCancelled) = print(io, "LoginCancelled: ", e.message)
struct LoginExpired <: Exception
    message::String
end
"""A validated provider denial; the message is LM15's, never provider text."""
struct LoginDenied <: Exception
    message::String
    status::Maybe{Int}
    provider_code::Maybe{String}
    stage::String
end
LoginDenied(message; status=nothing, provider_code=nothing, stage="authorization") =
    LoginDenied(message, status, provider_code, stage)
Base.showerror(io::IO, e::LoginDenied) = print(io, e.message)
"""A network failure during an auth exchange: `uncertain` when the request may have
reached the provider (a timeout or dropped connection), false when it never left."""
struct AuthTransportFailure <: Exception
    provider::String
    url::String
    uncertain::Bool
end

"""
    LoginContext

One attempt's deadline (monotonic), cancellation flag, UI, and the seams tests inject:
`clock` (monotonic seconds), `wall_clock` (epoch seconds), `opener` (the auth HTTP
function) and `sleep`.
"""
mutable struct LoginContext
    ui::Any
    deadline::Float64
    cancel::Base.RefValue{Bool}
    provider::String
    clock::Function
    wall_clock::Function
    opener::Any
    sleep::Any
end
remaining(ctx::LoginContext) = ctx.deadline - ctx.clock()
function check!(ctx::LoginContext)
    ctx.cancel[] && throw(LoginCancelled())
    remaining(ctx) <= 0 && throw(LoginExpired("login attempt deadline reached"))
    return nothing
end
function wait!(ctx::LoginContext, seconds)
    check!(ctx)
    seconds = min(max(Float64(seconds), 0.0), max(remaining(ctx), 0.0))
    if seconds <= 0
        check!(ctx)
        return nothing
    end
    if ctx.sleep !== nothing
        ctx.sleep(seconds)
    else
        # Wake on cancel: poll the flag rather than one long sleep.
        stop = time() + seconds
        while time() < stop
            ctx.cancel[] && throw(LoginCancelled())
            sleep(min(0.25, stop - time()))
        end
    end
    check!(ctx)
    return nothing
end
Base.notify(ctx::LoginContext, n::Notice) = (notify(ctx.ui, n); nothing)
function ask(ctx::LoginContext, p::Prompt)
    check!(ctx)
    answer = try
        prompt(ctx.ui, p)
    catch e
        e isa Union{InterruptException,EOFError,LoginCancelled} && throw(LoginCancelled("login cancelled at the prompt"))
        e isa AuthOperationError && rethrow()
        throw(auth_operation_error("the application's UI failed while prompting ($(nameof(typeof(e))))";
            reason="interaction_required", stage="interaction", recovery="operator_action"))
    end
    answer isa AbstractString || throw(ArgumentError("prompt must return a String"))
    return String(answer)
end
budget(ctx::LoginContext) = max(0.1, min(EXCHANGE_TIMEOUT_S, remaining(ctx)))

# Only these fixed protocol words can leave a private auth response (AUTH-24).
const OAUTH_ERROR_CODES = (
    "invalid_request", "invalid_client", "invalid_grant", "unauthorized_client",
    "unsupported_grant_type", "invalid_scope", "access_denied", "server_error",
    "temporarily_unavailable", "authorization_pending", "slow_down", "expired_token",
)
"""One auth HTTP reply: `body` may hold tokens and is never rendered."""
struct HttpReply
    status::Int
    body::JSONObject
    ok::Bool
    response_format::String
    oauth_error::Maybe{String}
    security_challenge::Bool
end
Base.show(io::IO, r::HttpReply) = print(io, "HttpReply(", r.status, ", <body withheld>)")
function failure_summary(r::HttpReply)
    details = ["HTTP $(r.status)", "response=$(r.response_format)"]
    push!(details, r.oauth_error === nothing ? "no recognized OAuth error code; cause not established" : "OAuth error=$(r.oauth_error)")
    if r.security_challenge
        push!(details, "response explicitly marked as a security challenge")
    elseif r.response_format == "html"
        push!(details, "HTML alone does not establish a security block")
    end
    return join(details, "; ")
end
const LM15_USER_AGENT = "lm15/$(pkgversion_string())"
"""The production auth opener: TLS verified, no redirect, bounded body, one attempt."""
function default_auth_opener(method, url, headers, body, timeout)
    t = max(1, ceil(Int, timeout))
    response = try
        http_request(method, url, headers, body;
            status_exception=false, redirect=false, retry=false, decompress=true,
            readtimeout=t, connect_timeout=t)
    catch e
        e isa InterruptException && rethrow()
        # A connection that never opened did not reach the provider; anything
        # after it (a timeout, a dropped connection) may have.
        cause = e isa LM15Error ? nothing : e
        throw(AuthTransportFailure("", url, !(cause isa HTTP.Exceptions.ConnectError)))
    end
    return Int(response.status), Dict(lowercase(String(k)) => String(v) for (k, v) in response.headers), response.body
end
function auth_send(ctx::LoginContext, method, url, headers::Vector{Pair{String,String}}, body::Vector{UInt8})
    uri = HTTP.URI(url)
    uri.scheme == "https" && !isempty(uri.host) || throw(auth_operation_error(
        "refusing a credential-bearing exchange over a non-HTTPS URL";
        reason="method_unavailable", stage="exchange", recovery="operator_action"))
    any(h -> lowercase(first(h)) == "user-agent", headers) || push!(headers, "User-Agent" => LM15_USER_AGENT)
    opener = ctx.opener === nothing ? default_auth_opener : ctx.opener
    status, response_headers, raw = try
        opener(method, url, headers, body, budget(ctx))
    catch e
        if e isa AuthTransportFailure
            throw(AuthTransportFailure(ctx.provider, url, e.uncertain))
        end
        rethrow()
    end
    length(raw) > AUTH_RESPONSE_LIMIT && throw(AuthError(
        "$(isempty(ctx.provider) ? "auth" : ctx.provider): authentication response exceeded $AUTH_RESPONSE_LIMIT bytes; refused";
        provider=isempty(ctx.provider) ? nothing : ctx.provider))
    content_type = lowercase(strip(first(split(get(response_headers, "content-type", ""), ';'))))
    response_format = "empty"
    parsed_body = JSONObject()
    oauth_error = nothing
    if !isempty(raw)
        response_format = content_type in ("text/html", "application/xhtml+xml") ? "html" :
            (content_type == "application/json" || endswith(content_type, "+json")) ? "invalid_json" : "text_or_binary"
        parsed = try
            text = String(copy(raw))
            isvalid(text) ? strict_json_parse(text) : nothing
        catch
            nothing
        end
        if parsed !== nothing
            response_format = "json"
            if parsed isa AbstractDict
                parsed_body = JSONObject(parsed)
                candidate = get(parsed, "error", nothing)
                candidate isa AbstractDict && (candidate = get(candidate, "code", get(candidate, "type", nothing)))
                candidate isa AbstractString && candidate in OAUTH_ERROR_CODES && (oauth_error = String(candidate))
            end
        end
    end
    challenge = lowercase(strip(get(response_headers, "cf-mitigated", ""))) == "challenge"
    status >= 500 && throw(ServerError(
        "$(isempty(ctx.provider) ? "auth" : ctx.provider): the authentication server answered HTTP $status";
        provider=isempty(ctx.provider) ? nothing : ctx.provider, status))
    return HttpReply(status, parsed_body, 200 <= status < 300, response_format, oauth_error, challenge)
end
function http_json(ctx::LoginContext, url, payload; headers=Pair{String,String}[], method="POST")
    body = method == "GET" ? UInt8[] : Vector{UInt8}(codeunits(JSON.serialize(payload)))
    h = Pair{String,String}["Content-Type" => "application/json", "Accept" => "application/json"]
    append!(h, [String(k) => String(v) for (k, v) in headers])
    return auth_send(ctx, method, url, h, body)
end
function http_form(ctx::LoginContext, url, payload; headers=Pair{String,String}[])
    h = Pair{String,String}["Content-Type" => "application/x-www-form-urlencoded", "Accept" => "application/json"]
    append!(h, [String(k) => String(v) for (k, v) in headers])
    return auth_send(ctx, "POST", url, h, Vector{UInt8}(codeunits(form_encode(payload))))
end
function http_get(ctx::LoginContext, url; headers=Pair{String,String}[])
    h = Pair{String,String}["Accept" => "application/json"]
    append!(h, [String(k) => String(v) for (k, v) in headers])
    return auth_send(ctx, "GET", url, h, UInt8[])
end

# Device flow (RFC 8628, AUTH-18). `poll()` returns (status, value, interval_s):
# status is "pending", "slow_down", "complete", "denied" or "expired".
positive_number(v) = v isa Real && !(v isa Bool) && isfinite(v) && v > 0 ? Float64(v) : nothing
function run_device_flow(ctx::LoginContext, poll; interval_s, expires_in_s, wait_before_first_poll=true)
    interval = interval_s === nothing || interval_s <= 0 ? DEVICE_DEFAULT_INTERVAL_S : Float64(interval_s)
    interval = max(interval, 1.0)
    expires_in_s === nothing || (ctx.deadline = min(ctx.deadline, ctx.clock() + Float64(expires_in_s)))
    wait_before_first_poll && wait!(ctx, interval)
    while true
        check!(ctx)
        status, value, proposed = poll()
        status == "complete" && return value
        status == "denied" && throw(LoginDenied("the provider reported that authorization was denied"))
        status == "expired" && throw(LoginExpired("the provider reported that the device code expired"))
        if status == "slow_down"
            interval = max(interval + DEVICE_SLOW_DOWN_STEP_S, something(proposed, 0.0))
        elseif status != "pending"
            throw(ArgumentError("device poll returned unknown status $(repr(status))"))
        end
        wait!(ctx, interval)
    end
end
device_step(status, value=nothing, interval=nothing) = (status, value, interval)

"""A validated authorization-code return: the code (private) and the state."""
struct CallbackReturn
    code::String
    state::Maybe{String}
end
Base.show(io::IO, ::CallbackReturn) = print(io, "CallbackReturn(<code withheld>)")
invalid_return(message) = auth_operation_error(message; reason="invalid_login_state", stage="interaction", recovery="provide_input")
function parse_query_pairs(q)
    out = Pair{String,String}[]
    isempty(q) && return out
    for piece in split(q, '&'; keepempty=false)
        k, v = occursin('=', piece) ? split(piece, '='; limit=2) : (piece, "")
        push!(out, HTTP.URIs.unescapeuri(replace(k, '+' => ' ')) => HTTP.URIs.unescapeuri(replace(v, '+' => ' ')))
    end
    return out
end
default_port(scheme) = scheme == "https" ? "443" : scheme == "http" ? "80" : ""
"""
    parse_manual_return(text; expected_state, allow_bare_code, registered_path, registered_uri=nothing)

Validate a pasted return (a redirect URL, `code#state`, a query, or a bare code where the
profile allows it) against the attempt's registered return context (AUTH-18). No pasted
value is included in an error.
"""
function parse_manual_return(text; expected_state, allow_bare_code, registered_path, registered_uri=nothing)
    value = strip(something(text, ""))
    isempty(value) && throw(invalid_return("nothing was pasted"))
    ncodeunits(value) > CALLBACK_TARGET_LIMIT && throw(invalid_return("pasted return is too long"))
    code = nothing
    state = nothing
    params = nothing
    bare = false
    if occursin("://", value)
        uri = try
            HTTP.URI(value)
        catch
            throw(invalid_return("the pasted URL is not this sign-in's registered return URL"))
        end
        (!isempty(uri.userinfo) || !isempty(uri.fragment)) &&
            throw(invalid_return("the pasted URL is not this sign-in's registered return URL"))
        registered_path === nothing || uri.path == registered_path ||
            throw(invalid_return("the pasted URL is not this sign-in's registered return URL"))
        if registered_uri !== nothing
            expected = HTTP.URI(registered_uri)
            address(u) = (u.scheme, lowercase(u.host), isempty(u.port) ? default_port(u.scheme) : u.port, u.path)
            address(uri) == address(expected) ||
                throw(invalid_return("the pasted URL is not this sign-in's registered return URL"))
        end
        params = parse_query_pairs(uri.query)
    elseif startswith(value, "code=") || startswith(value, "state=") || startswith(value, "error=")
        params = parse_query_pairs(value)
    elseif occursin('#', value)
        code, state = split(value, '#'; limit=2)
    else
        code, bare = value, true
    end
    denied = false
    if params !== nothing
        names = first.(params)
        length(unique(names)) == length(names) || throw(invalid_return("the pasted return repeats a parameter"))
        query = Dict(params)
        haskey(query, "code") && haskey(query, "error") &&
            throw(invalid_return("the pasted return contains both a code and an error"))
        denied = haskey(query, "error")
        code, state = get(query, "code", nothing), get(query, "state", nothing)
    end
    bare && !allow_bare_code && throw(invalid_return("paste the complete code#state or return URL, not the code alone"))
    if expected_state !== nothing
        if state === nothing
            bare && allow_bare_code || throw(invalid_return("this provider's return must carry its state value"))
        elseif !constant_time_equal(String(state), String(expected_state))
            throw(invalid_return("the pasted return does not belong to this sign-in attempt"))
        end
    end
    # A wrong-state error never terminates the legitimate attempt as denied.
    denied && throw(LoginDenied("the validated pasted return carries a provider error"))
    (code === nothing || isempty(code)) && throw(invalid_return("no authorization code in the pasted text"))
    return CallbackReturn(String(code), state === nothing ? nothing : String(state))
end
function constant_time_equal(a::String, b::String)
    x = codeunits(a)
    y = codeunits(b)
    length(x) == length(y) || return false
    acc = 0x00
    for i in eachindex(x)
        acc |= x[i] ⊻ y[i]
    end
    return acc == 0x00
end

"""
    CallbackListener(; path, expected_state, bind_host="127.0.0.1", port=0, redirect_host=nothing)

A one-shot loopback listener for an authorization-code return (AUTH-18): loopback bind
only, exact path, state checked on success and error returns, bounded request size, no
access log. `redirect_host` is the registered return host (it may say `localhost`),
deliberately separate from the bound address.
"""
mutable struct CallbackListener
    path::String
    expected_state::Maybe{String}
    bind_host::String
    redirect_host::String
    server::Any
    port::Int
    result::Maybe{CallbackReturn}
    denied::Bool
    done::Bool
    closed::Bool
    lock::ReentrantLock
end
function CallbackListener(; path, expected_state, bind_host="127.0.0.1", port=0, redirect_host=nothing)
    bind_host in ("127.0.0.1", "::1") || throw(auth_operation_error(
        "callback listener may bind loopback only, not $(repr(bind_host))";
        reason="method_unavailable", stage="reservation", recovery="operator_action"))
    startswith(path, "/") || throw(ArgumentError("callback path must start with '/'"))
    listener = CallbackListener(path, expected_state, bind_host, something(redirect_host, bind_host), nothing, 0,
        nothing, false, false, true, ReentrantLock())
    page(status, title, message) = HTTP.Response(status,
        ["Content-Type" => "text/html; charset=utf-8", "Cache-Control" => "no-store", "Referrer-Policy" => "no-referrer"],
        "<!doctype html><meta charset='utf-8'><meta name='referrer' content='no-referrer'><title>$(html_escape(title))</title><p>$(html_escape(message))</p>")
    function handle(request::HTTP.Request)
        request.method == "GET" || return page(405, "Rejected", "Sign-in return was not accepted.")
        (ncodeunits(request.target) > CALLBACK_TARGET_LIMIT ||
            sum((ncodeunits(k) + ncodeunits(v) for (k, v) in request.headers); init=0) > CALLBACK_HEADER_LIMIT) &&
            return page(414, "Rejected", "Request too large.")
        uri = HTTP.URI(request.target)
        uri.path == listener.path || return page(404, "Not found", "Callback route not found.")
        lock(listener.lock) do
            listener.done && return page(409, "Already used", "This sign-in return was already handled.")
            params = parse_query_pairs(uri.query)
            names = first.(params)
            length(unique(names)) == length(names) || return page(400, "Rejected", "Sign-in return was not accepted.")
            query = Dict(params)
            state = get(query, "state", nothing)
            if listener.expected_state !== nothing &&
               (state === nothing || !constant_time_equal(state, listener.expected_state))
                # Wrong state on a success OR an error return: generic rejection;
                # the legitimate wait continues.
                return page(400, "Rejected", "Sign-in return was not accepted.")
            end
            has_code = !isempty(get(query, "code", ""))
            has_error = haskey(query, "error")
            has_code == has_error && return page(400, "Rejected", "Sign-in return was not accepted.")
            if has_error
                listener.denied = true
                listener.done = true
                return page(400, "Not completed", "Sign-in was not completed.")
            end
            listener.result = CallbackReturn(query["code"], state)
            listener.done = true
            return page(200, "Signed in", "Sign-in completed. You can close this window.")
        end
    end
    server = try
        with_transport_logger(() -> HTTP.serve!(handle, bind_host, port; access_log=nothing, verbose=false))
    catch
        throw(auth_operation_error(
            "could not listen on $(bind_host):$(port == 0 ? "ephemeral" : port) for the sign-in return; another program may be using the port";
            reason="method_unavailable", stage="reservation", recovery="choose_method"))
    end
    listener.server = server
    listener.port = Int(HTTP.port(server))
    listener.closed = false
    return listener
end
function redirect_uri(l::CallbackListener)
    host = l.redirect_host
    occursin(':', host) && !startswith(host, "[") && (host = "[$host]")
    return "http://$host:$(l.port)$(l.path)"
end
function stop!(l::CallbackListener)
    l.closed && return nothing
    l.closed = true
    try
        close(l.server)
    catch
    end
    return nothing
end
html_escape(s) = replace(String(s), '&' => "&amp;", '<' => "&lt;", '>' => "&gt;", '"' => "&quot;", '\'' => "&#39;")

"""Wait for the loopback return or a pasted value, whichever arrives first (AUTH-16)."""
function race_callback_and_manual(ctx::LoginContext, listener, p::ManualCodePrompt)
    listener === nothing && return nothing, ask(ctx, p)
    box = Ref{Any}(nothing)
    finished = Threads.Atomic{Bool}(false)
    task = @async begin
        try
            box[] = (:value, prompt(ctx.ui, p))
        catch e
            box[] = e isa Union{InterruptException,EOFError,LoginCancelled} ? (:cancelled, nothing) : (:error, nameof(typeof(e)))
        finally
            finished[] = true
        end
    end
    while true
        check!(ctx)
        if listener.done
            listener.denied && throw(LoginDenied("the provider returned an error to the sign-in callback"))
            dismiss(ctx.ui, p)
            return listener.result, nothing
        end
        if finished[]
            kind, value = box[]
            kind == :cancelled && throw(LoginCancelled("login cancelled at the prompt"))
            kind == :error && throw(auth_operation_error(
                "the application's UI failed while prompting ($(value))";
                reason="interaction_required", stage="interaction", recovery="operator_action"))
            return nothing, value isa AbstractString ? String(value) : ""
        end
        ctx.sleep === nothing ? sleep(0.05) : yield()
    end
end
"""The authorization return, from the listener or a paste, validated. A paste that fails
validation is rejected with a notice and the legitimate wait goes on (AUTH-18)."""
function await_return(ctx::LoginContext, listener, p::ManualCodePrompt, parse)
    while true
        returned, pasted = race_callback_and_manual(ctx, listener, p)
        returned === nothing || return returned
        try
            return parse(something(pasted, ""))
        catch e
            e isa AuthOperationError && e.reason == "invalid_login_state" || rethrow()
            notify(ctx, InfoNotice("$(e.message). Try again."))
        end
    end
end
