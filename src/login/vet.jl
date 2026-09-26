# The vet shim's `managed_run` op (lm15-contract harness/PROTOCOL.md § managed): one
# scripted program against the public managed-auth API, every seam injected — the store
# file the harness created, a fake wall and monotonic clock (waits advance them), a
# scripted auth server as the Auth's opener, and a scripted UI. It reports one outcome
# per step, the ordered trace and the store file afterwards; the harness compares.

mutable struct VetClock
    start_s::Float64
    elapsed_s::Float64
end
const VET_TRANSPORT_HEADERS = Set(("accept", "accept-encoding", "connection", "content-length", "content-type", "host"))
function vet_request_body(content_type, body::Vector{UInt8})
    isempty(body) && return nothing
    text = String(copy(body))
    content_type == "application/x-www-form-urlencoded" && return JSONObject(k => v for (k, v) in parse_query_pairs(text))
    return try
        JSON.parse(text)
    catch
        text
    end
end
function vet_server(script, events)
    queue = collect(script)
    return function (method, url, headers, body, timeout)
        content_type = nothing
        recorded = JSONObject()
        for (k, v) in headers
            key = lowercase(k)
            key == "content-type" && (content_type = String(strip(first(split(v, ';')))))
            key in VET_TRANSPORT_HEADERS && continue
            recorded[key] = key == "user-agent" && startswith(v, "lm15/") ? "lm15" : v
        end
        push!(events, obj("http" => obj("method" => method, "url" => url, "content_type" => content_type,
            "headers" => recorded, "body" => vet_request_body(content_type, body))))
        isempty(queue) && throw(AuthTransportFailure("", url, false))
        reply = popfirst!(queue)
        delay = get(reply, "delay_ms", nothing)
        delay isa Real && delay > 0 && sleep(delay / 1000)  # real time: lets another process race this exchange
        network = get(reply, "network", nothing)
        network == "timeout" && throw(AuthTransportFailure("", url, true))
        network == "refused" && throw(AuthTransportFailure("", url, false))
        status = Int(get(reply, "status", 200))
        if haskey(reply, "json")
            return status, Dict("content-type" => "application/json"), Vector{UInt8}(codeunits(JSON.serialize(reply["json"])))
        end
        return status, Dict("content-type" => String(get(reply, "content_type", "text/plain"))),
            Vector{UInt8}(codeunits(string(get(reply, "text", ""))))
    end
end
mutable struct VetUI <: AuthUI
    answers::Vector{Any}
    events::Vector{Any}
    last_auth_url::Maybe{String}
end
function Base.notify(ui::VetUI, n::Notice)
    n isa AuthUrlNotice && (ui.last_auth_url = n.url)
    event = if n isa AuthUrlNotice
        obj("type" => "auth_url", "url" => n.url)
    elseif n isa DeviceCodeNotice
        obj("type" => "device_code", "user_code" => n.user_code, "verification_url" => n.verification_url,
            "expires_in_s" => n.expires_in_s, "interval_s" => n.interval_s)
    elseif n isa ProgressNotice
        obj("type" => "progress", "stage" => n.stage)
    else
        obj("type" => "info")
    end
    push!(ui.events, obj("notice" => event))
    return nothing
end
function prompt(ui::VetUI, p::Prompt)
    event = obj("type" => prompt_type(p), "field_id" => p.field_id)
    p isa SelectPrompt && (event["options"] = [o.id for o in p.options])
    push!(ui.events, obj("prompt" => event))
    isempty(ui.answers) && throw(LoginCancelled("the script has no more answers"))
    answer = popfirst!(ui.answers)
    answer isa AbstractString && return String(answer)
    get(answer, "cancel", false) === true && throw(LoginCancelled("the script cancels here"))
    query = ui.last_auth_url === nothing ? Dict{String,String}() : Dict(parse_query_pairs(HTTP.URI(ui.last_auth_url).query))
    state = get(query, "state", "")
    haskey(answer, "paste") && return "$(answer["paste"])#$state"
    haskey(answer, "paste_wrong_state") && return "$(answer["paste_wrong_state"])#not-the-state-of-this-attempt"
    haskey(answer, "paste_url") && return "$(get(query, "redirect_uri", ""))?$(form_encode(obj("code" => answer["paste_url"], "state" => state)))"
    throw(ArgumentError("unknown scripted answer"))
end
function vet_error(e)
    e isa LoginCancelled && return obj("type" => "cancelled")
    e isa AuthOperationError && return obj("type" => "AuthOperationError", "code" => "auth_operation", "reason" => e.reason,
        "stage" => e.stage, "commit_state" => e.commit_state, "recovery" => e.recovery)
    e isa LM15Error && return obj("type" => class_name(e), "code" => error_code(e))
    return obj("type" => string(nameof(typeof(e))))
end
function vet_connection(c)
    c === nothing && return nothing
    out = obj("id" => c.id, "provider" => c.provider, "instance_id" => c.instance_id, "kind" => c.kind,
        "method_id" => c.method_id, "routes" => collect(c.routes), "label" => c.label, "created_at" => c.created_at,
        "identity_generation" => c.identity_generation, "credential_revision" => c.credential_revision,
        "settings" => JSONObject(c.settings))
    c.account_label === nothing || (out["account_label"] = c.account_label)
    return out
end
vet_status(s) = obj("provider" => s.provider, "presence" => s.presence, "usability" => s.usability,
    "connection" => vet_connection(s.connection), "expires_at" => s.expires_at, "logged_out" => s.logged_out,
    "verification" => s.verification === nothing ? nothing : obj("result" => s.verification.result, "check" => s.verification.check))
function vet_credential(v)
    v === nothing && return nothing
    v isa BearerToken && return obj("kind" => "bearer", "value" => v.value)
    v isa ApiKey && return obj("kind" => "api_key", "value" => v.value)
    v isa AbstractString && return obj("kind" => "api_key", "value" => String(v))
    return obj("kind" => string(nameof(typeof(v))))
end
vet_method(m) = obj("id" => m.id, "kind" => m.kind, "flow" => m.flow, "availability" => m.availability,
    "subscription" => m.subscription, "delivery" => collect(m.delivery),
    "fields" => [obj("id" => f.id, "type" => f.type, "required" => f.required, "options" => [o.id for o in f.options]) for f in m.fields])
function vet_refs(step, outcomes)
    value(n) = begin
        o = outcomes[Int(n) + 1]
        get(o, "ok", false) && get(o, "value", nothing) isa AbstractDict || throw(ArgumentError("step $n returned no connection to refer to"))
        o["value"]
    end
    out = JSONObject()
    for (k, v) in step
        out[k] = if v isa AbstractDict && haskey(v, "id_of_step")
            value(v["id_of_step"])["id"]
        elseif v isa AbstractDict && haskey(v, "of_step")
            c = value(v["of_step"])
            [c["id"], c["identity_generation"]]
        else
            v
        end
    end
    return out
end
strdict(d) = d === nothing ? Dict{String,String}() : Dict{String,String}(String(k) => String(v) for (k, v) in d)
function vet_step(a::Auth, step, clock, ui, env, sentinel)
    act = step["do"]
    act == "advance" && (clock.elapsed_s += step["ms"] / 1000; return nothing)
    act == "login" && return vet_connection(login(a, step["provider"], get(step, "method", nothing); ui,
        answers=strdict(get(step, "answers", nothing)), settings=strdict(get(step, "settings", nothing)),
        replace=get(step, "replace", nothing), allow_unverified=get(step, "allow_unverified", false) === true))
    act == "configure" && return vet_connection(configure(a, step["provider"]; method=step["method"],
        answers=strdict(get(step, "answers", nothing)), settings=strdict(get(step, "settings", nothing)),
        replace=get(step, "replace", nothing)))
    act == "set_api_key" && return vet_connection(set_api_key(a, step["provider"], step["key"]; replace=get(step, "replace", nothing)))
    act == "status" && return vet_status(status(a, step["provider"]))
    act == "connections" && return [vet_connection(c) for c in connections(a)]
    if act == "logout"
        f = logout(a, step["target"])
        return obj("provider" => f.provider, "forgot" => f.forgot, "routes" => collect(f.routes), "identity_generation" => f.identity_generation)
    end
    act == "cancel_login" && return cancel_login(a, step["provider"])
    if act == "request_auth"
        pinned = get(step, "pinned", nothing)
        r = request_auth(a, step["provider"]; pinned=pinned === nothing ? nothing : (String(pinned[1]), String(pinned[2])))
        return obj("credential" => vet_credential(r.credential), "headers" => JSONObject(r.headers),
            "base_url" => r.base_url, "account_id" => r.account_id, "named" => r.named)
    end
    act == "methods" && return [vet_method(m) for m in login_methods(a, step["provider"])]
    act == "providers" && return sort!([d.id for d in login_providers(a)])
    if act == "explain"
        keysmap = Dict{String,Any}(p => "$sentinel-explicit" for p in get(step, "api_keys", []))
        report = explain_auth(step["provider"]; env, api_keys=keysmap, auth=a)
        return obj("configured" => report.configured, "steps" => [obj("kind" => s.kind, "state" => string(s.state)) for s in report.steps])
    end
    throw(ArgumentError("unknown managed step $(repr(act))"))
end
function vet_managed_run(msg)
    events = Any[]
    clock = VetClock(msg["clock_ms"] / 1000, 0.0)
    env = Dict{String,String}(String(k) => String(v) for (k, v) in something(get(msg, "env", nothing), Dict()))
    record_sleep = seconds -> (clock.elapsed_s += seconds; push!(events, obj("sleep_ms" => round(Int, seconds * 1000))); nothing)
    ui = VetUI(collect(something(get(msg, "ui", nothing), [])), events, nothing)
    store_path = msg["store_path"]
    saved = Dict(ENV)
    clear_env!() = foreach(k -> delete!(ENV, k), collect(keys(ENV)))
    clear_env!()
    merge!(ENV, env)
    steps = Any[]
    try
        a = Auth(FileStore(store_path); clock=() -> clock.start_s + clock.elapsed_s, monotonic=() -> clock.elapsed_s,
            opener=vet_server(something(get(msg, "http", nothing), []), events), sleep=record_sleep)
        for (index, step) in enumerate(msg["steps"])
            push!(events, obj("step" => index - 1))
            try
                push!(steps, obj("ok" => true, "value" => vet_step(a, vet_refs(step, steps), clock, ui, env, String(msg["sentinel"]))))
            catch e
                e isa Union{LM15Error,LoginCancelled,ArgumentError,KeyError} || rethrow()
                push!(steps, obj("ok" => false, "error" => vet_error(e)))
            end
        end
    finally
        clear_env!()
        merge!(ENV, saved)
    end
    store = nothing
    if isfile(store_path)
        text = read(store_path, String)
        store = try
            obj("document" => JSON.parse(text))
        catch
            obj("raw" => text)
        end
    end
    return obj("steps" => steps, "events" => events, "store" => store)
end
