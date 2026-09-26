# `Auth`: one scope's connections and their lifecycle (spec/auth-managed.md AUTH-14
# construction is inert, scope is explicit; AUTH-17 what each operation may touch;
# AUTH-19 generations, replacement, logout, cancellation ordered against commit; AUTH-20
# renewal under the lock with a durable in-flight marker, uncertainty never retried
# blind; AUTH-24 typed outcomes). A slot is one provider route in the scope; its record
# in `_lm15.slots` carries an identity generation (bumped on every new connection and
# on logout, never reused) and a credential revision (bumped on every renewal). A
# legacy entry without a record reads as generation 1 with a stable id; the record is
# written on the first managed commit, never on read.

const RENEWAL_LEAD_S = 300.0  # AUTH-20.3: min(300 s, lifetime / 10)
iso_utc(seconds) = Dates.format(unix2datetime(floor(seconds)), dateformat"yyyy-mm-ddTHH:MM:SS") * "Z"
iso_ms(ms) = iso_utc(ms / 1000)

"""
    ForgetResult

What `logout` did: the provider, whether a connection was forgotten, the routes that
lost access, and the slot's identity generation afterwards.
"""
struct ForgetResult
    provider::String
    forgot::Bool
    routes::Tuple
    identity_generation::String
end

mutable struct Slot
    provider::String
    generation::Int
    connection_id::Maybe{String}
    revision::Int
    kind::String
    method_id::String
    instance_id::String
    label::String
    account_label::Maybe{String}
    created_at::String
    routes::Vector{String}
    settings::Dict{String,String}
    state::String
    renewal::String
    logged_out::Bool
    renewal_in_flight::Any
    attempt::Any
    verification::Any
    previous_ids::Vector{String}
    legacy::Bool
end
empty_slot(provider) = Slot(provider, 0, nothing, 0, "account", "", "public", "", nothing, "", String[],
    Dict{String,String}(), "ready", "refresh_token", false, nothing, nothing, nothing, String[], false)
record_string(r, k, default="") = (v = get(r, k, default); v === nothing ? default : v)
function slot_from_record(provider, r)
    return Slot(provider, parse(Int, string(record_string(r, "generation", "0"))), get(r, "connection_id", nothing),
        parse(Int, string(record_string(r, "revision", "0"))), record_string(r, "kind", "account"),
        record_string(r, "method_id"), record_string(r, "instance_id", "public"), record_string(r, "label"),
        get(r, "account_label", nothing), record_string(r, "created_at"),
        String[string(x) for x in record_string(r, "routes", [])],
        Dict{String,String}(string(k) => string(v) for (k, v) in record_string(r, "settings", obj())),
        record_string(r, "state", "ready"), record_string(r, "renewal", "refresh_token"),
        get(r, "logged_out", false) === true, get(r, "renewal_in_flight", nothing), get(r, "attempt", nothing),
        get(r, "verification", nothing), String[string(x) for x in record_string(r, "previous_ids", [])], false)
end
function slot_record(s::Slot)
    r = JSONObject("generation" => string(s.generation), "connection_id" => s.connection_id,
        "revision" => string(s.revision), "kind" => s.kind, "method_id" => s.method_id,
        "instance_id" => s.instance_id, "label" => s.label, "created_at" => s.created_at,
        "routes" => copy(s.routes), "settings" => JSONObject(s.settings), "state" => s.state, "renewal" => s.renewal)
    s.account_label === nothing || isempty(s.account_label) || (r["account_label"] = s.account_label)
    s.logged_out && (r["logged_out"] = true)
    s.renewal_in_flight === nothing || isempty(s.renewal_in_flight) || (r["renewal_in_flight"] = s.renewal_in_flight)
    s.attempt === nothing || isempty(s.attempt) || (r["attempt"] = s.attempt)
    s.verification === nothing || isempty(s.verification) || (r["verification"] = s.verification)
    isempty(s.previous_ids) || (r["previous_ids"] = s.previous_ids[max(1, end - 7):end])
    return r
end
function slot_connection(s::Slot)
    (s.connection_id === nothing || isempty(s.connection_id)) && return nothing
    return Connection(s.connection_id, s.provider, s.instance_id, s.kind, s.method_id,
        Tuple(isempty(s.routes) ? [s.provider] : s.routes), isempty(s.label) ? s.provider : s.label, s.created_at,
        string(s.generation), string(s.revision), copy(s.settings), s.account_label)
end
const LEGACY_METHOD = Dict("xai" => "device", "claude-code" => "external:claude-code-cli", "openai-codex" => "external:codex-cli")

"""
    Auth(store; clock=time, monotonic=..., opener=nothing, sleep=nothing)
    local_auth(path=nothing)
    memory_auth()

A scope's connections: sign in once, use everywhere (AUTH-12–26). Construction reads
nothing. `clock` is epoch seconds, `monotonic` a monotonic clock, `opener` the auth HTTP
seam `(method, url, headers, body, timeout) -> (status, headers, body)`, `sleep` a wait
seam; tests inject them, production leaves the defaults.
"""
mutable struct Auth
    store::Store
    clock::Function
    monotonic::Function
    opener::Any
    sleep::Any
    active::Dict{String,Base.RefValue{Bool}}
    lock::ReentrantLock
    closed::Bool
    pin::Union{Nothing,Tuple{String,String}}
end
Auth(store::Store; clock=time, monotonic=() -> time_ns() / 1e9, opener=nothing, sleep=nothing) =
    Auth(store, clock, monotonic, opener, sleep, Dict{String,Base.RefValue{Bool}}(), ReentrantLock(), false, nothing)
"""Managed authentication over the private file store (AUTH-8 path unless given)."""
local_auth(path=nothing; kw...) = Auth(FileStore(path); kw...)
"""Managed authentication over a process-lifetime store."""
memory_auth(; kw...) = Auth(MemoryStore(); kw...)
Base.show(io::IO, a::Auth) = print(io, "Auth(", a.store, ")")
function check_open(a::Auth)
    a.closed && throw(auth_operation_error("this Auth was closed"; reason="storage_unavailable", stage="resolution", recovery="operator_action"))
end
"""
    close(auth)

Cancel logins this manager is running; never a logout (AUTH-14).
"""
function Base.close(a::Auth)
    lock(a.lock) do
        a.closed = true
        foreach(flag -> flag[] = true, values(a.active))
    end
    return nothing
end

# ─── discovery (AUTH-13): definitions only ─────────────────────────────
"""
    login_providers(auth) / login_providers()

The providers a manager can connect, from definitions only (AUTH-13): no credential
file, environment or network is read.
"""
login_providers(::Auth) = login_providers()
login_providers() = Tuple(login_descriptor(p) for p in login_provider_ids())
"""
    login_methods(auth, provider) / login_methods(provider)

A provider's login methods (AUTH-13), from definitions only.
"""
login_methods(a::Auth, provider) = descriptor(a, provider).methods
login_methods(provider) = descriptor(nothing, provider).methods
function descriptor(::Any, provider)
    try
        return login_descriptor(provider)
    catch e
        e isa KeyError || rethrow()
        throw(auth_operation_error("$(repr(provider)) is not a provider lm15 can connect; see login_providers()";
            reason="method_unavailable", stage="discovery", recovery="choose_method", provider=String(provider)))
    end
end

# ─── store views ───────────────────────────────────────────────────────
function slot_view(document, provider)
    meta = get(document, META_KEY, nothing)
    slots = meta isa AbstractDict ? get(meta, "slots", obj()) : obj()
    record = get(slots, provider, nothing)
    material = get(document, provider, nothing)
    material isa AbstractDict || (material = nothing)
    record === nothing || return slot_from_record(provider, record), material
    slot = empty_slot(provider)
    if material !== nothing
        oauth = get(material, "type", nothing) == "oauth"
        slot = empty_slot(provider)
        slot.generation = 1
        slot.connection_id = "legacy-$provider"
        slot.revision = 1
        slot.kind = oauth ? "account" : "api_key"
        slot.method_id = get(LEGACY_METHOD, provider, "api_key")
        slot.label = "$provider (existing login)"
        slot.routes = [provider]
        slot.legacy = true
        slot.renewal = oauth ? "refresh_token" : "none"
    end
    return slot, material
end
function slot_put!(document, slot::Slot, material)
    meta = get!(document, META_KEY, JSONObject("version" => STORE_VERSION, "slots" => JSONObject()))
    haskey(meta, "version") || (meta["version"] = STORE_VERSION)
    get!(meta, "slots", JSONObject())[slot.provider] = slot_record(slot)
    material === nothing ? delete!(document, slot.provider) : (document[slot.provider] = material)
    return document
end

# ─── inspection (AUTH-17: store reads only) ────────────────────────────
"""
    connections(auth) -> Vector{Connection}

The saved connections in this scope (secret-free metadata; a store read only).
"""
function connections(a::Auth)
    document = read_document(a.store)
    known = Set(login_provider_ids())
    found = Connection[]
    for key in sort!(collect(keys(document)))
        (key == META_KEY || !(key in known)) && continue
        c = slot_connection(first(slot_view(document, key)))
        c === nothing || push!(found, c)
    end
    meta = get(document, META_KEY, obj())
    for (key, record) in (meta isa AbstractDict ? get(meta, "slots", obj()) : obj())
        haskey(document, key) && continue
        c = slot_connection(slot_from_record(key, record))
        c === nothing || push!(found, c)
    end
    return found
end
"""
    status(auth, provider) -> ConnectionStatus

Presence, local usability and the last verification of the provider's saved connection
(AUTH-24). A store read only: no refresh, no network.
"""
function status(a::Auth, provider)
    provider = descriptor(a, provider).id
    slot, material = slot_view(read_document(a.store), provider)
    connection = slot_connection(slot)
    verification = slot.verification === nothing ? nothing : Verification(get(slot.verification, "result", "unverified");
        checked_at=get(slot.verification, "checked_at", nothing), check=get(slot.verification, "check", nothing),
        detail=get(slot.verification, "detail", nothing))
    connection === nothing && return ConnectionStatus(provider, "absent", "unknown", nothing, nothing, slot.logged_out,
        nothing, slot.logged_out ? "signed out; sign in again or pass a key explicitly" : nothing)
    usability, expires, detail = usability_of(a, slot, material)
    return ConnectionStatus(provider, "saved", usability, connection, expires, false, verification, detail)
end
lead_ms(lifetime) = round(Int64, (lifetime === nothing ? RENEWAL_LEAD_S : min(RENEWAL_LEAD_S, lifetime / 10)) * 1000)
renewable(slot, material) = !(slot.renewal in ("none", "recipe")) && string_body(material, "refresh") !== nothing
function usability_of(a::Auth, slot, material)
    slot.state == "needs_login" && return "needs_login", nothing, "the provider rejected the saved credential; sign in again"
    (slot.state == "indeterminate" || slot.renewal_in_flight !== nothing) &&
        return "indeterminate", nothing, "a renewal was interrupted; sign in again to be safe"
    material === nothing && return "needs_login", nothing, "credential material is missing"
    flow = flow_for_material(slot.provider, material)
    expiry = flow_expiry(flow, material)
    expiry == "never" && return "ready", "never", nothing
    expiry === nothing && return get(material, "type", nothing) == "external" ? ("ready", "unknown", nothing) : ("unknown", "unknown", nothing)
    now = round(Int64, a.clock() * 1000)
    if now >= expiry - lead_ms(flow_lifetime_s(flow, material))
        renewable(slot, material) || return "needs_login", iso_ms(expiry), "expired and not renewable"
        return "renewal_due", iso_ms(expiry), nothing
    end
    return "ready", iso_ms(expiry), nothing
end

# ─── login (AUTH-16/17/18/19) ──────────────────────────────────────────
"""
    login(auth, provider, method=nothing; ui, settings=Dict(), answers=Dict(), replace=nothing,
          lifetime_s=900, allow_unverified=false) -> Connection

Run one login to completion and save the connection. `method` is a method id (or a
`LoginMethod`); omitted, the UI is asked when more than one selectable method remains.
`replace` names the connection id being replaced; without it an occupied slot is
`connection_exists`. The login is saved before this returns.
"""
function login(a::Auth, provider::AbstractString, method=nothing; ui, settings=Dict{String,String}(),
    answers=Dict{String,String}(), replace=nothing, lifetime_s=ATTEMPT_LIFETIME_S, allow_unverified=false)
    check_open(a)
    d = descriptor(a, provider)
    provider = d.id
    lifetime_s isa Real && !(lifetime_s isa Bool) && isfinite(lifetime_s) && lifetime_s > 0 ||
        throw(ArgumentError("lifetime_s must be a positive number of seconds"))
    chosen = choose_method(d, method, ui; allow_unverified)
    answers = Dict{String,String}(String(k) => String(v) for (k, v) in something(answers, Dict()))
    settings = Dict{String,String}(String(k) => String(v) for (k, v) in something(settings, Dict()))
    cancel = Ref(false)
    ctx = LoginContext(ui, a.monotonic() + Float64(lifetime_s), cancel, provider, a.monotonic, a.clock, a.opener, a.sleep)
    attempt_id = "at_$(token_urlsafe(16))"
    # Reservation (AUTH-17/18): storage proven writable, one active attempt per
    # slot, generation observed — before any browser opens.
    reserve!(a.store)
    expected = reserve_slot(a, provider, attempt_id, replace, lifetime_s)
    lock(() -> (a.active[provider] = cancel), a.lock)
    try
        for f in chosen.fields
            haskey(answers, f.id) && continue
            answers[f.id] = if !f.required && f.type != "select"
                ask(ctx, TextPrompt(f.id, f.label))
            elseif f.type == "secret"
                ask(ctx, SecretPrompt(f.id, f.label))
            elseif f.type == "select"
                ask(ctx, SelectPrompt(f.id, f.label, f.options))
            else
                ask(ctx, TextPrompt(f.id, f.label))
            end
        end
        flow = login_flow(provider, chosen.id)
        result = try
            flow_login(flow, ctx, chosen, settings, answers)
        catch e
            release_slot(a, provider, attempt_id)
            e isa LoginExpired && throw(auth_operation_error(
                "$provider: the sign-in was not completed within $(floor(Int, lifetime_s ÷ 60)) minutes; start again";
                reason="login_expired", stage="polling", recovery="restart_login", provider, attempt_id, method_id=chosen.id))
            e isa LoginDenied && throw(auth_operation_error("$provider: $(e.message)";
                reason="login_denied", stage=e.stage, recovery="restart_login", provider, attempt_id,
                method_id=chosen.id, status=e.status, provider_code=e.provider_code))
            if e isa AuthTransportFailure
                e.uncertain && throw(auth_operation_error(
                    "$provider: the network failed after the authorization code may have been sent; the code is one-use, so sign in again rather than retry";
                    reason="indeterminate", stage="exchange", commit_state="not_committed", recovery="restart_login",
                    provider, attempt_id, method_id=chosen.id))
                throw(TransportError("$provider: network failure during an authentication exchange to $(e.url)"; provider))
            end
            rethrow()
        end
        return commit(a, provider, attempt_id, expected, chosen, result, settings)
    catch
        # Any exit without a saved connection ends the attempt, so its
        # reservation never outlives it (releasing someone else's is a no-op).
        release_slot(a, provider, attempt_id)
        rethrow()
    finally
        lock(() -> delete!(a.active, provider), a.lock)
    end
end
function choose_method(d::ProviderDescriptor, m, ui; allow_unverified)
    m isa LoginMethod && (m = m.id)
    if m !== nothing
        chosen = try
            method(d, m)
        catch e
            e isa KeyError || rethrow()
            throw(auth_operation_error("$(d.id): no login method $(repr(m)); see login_methods($(repr(d.id)))";
                reason="method_unavailable", stage="discovery", recovery="choose_method", provider=d.id))
        end
        chosen.availability == "unavailable" && throw(auth_operation_error(
            "$(d.id): method $(repr(m)) is unavailable: $(chosen.reason)";
            reason="method_unavailable", stage="discovery", recovery="choose_method", provider=d.id, method_id=String(m)))
        chosen.availability == "unverified" && !allow_unverified && throw(auth_operation_error(
            "$(d.id): method $(repr(m)) has no live receipt yet ($(chosen.reason)); pass allow_unverified=true to try it knowing that";
            reason="method_unavailable", stage="discovery", recovery="choose_method", provider=d.id, method_id=String(m)))
        return chosen
    end
    candidates = [x for x in d.methods if x.availability == "supported" || (allow_unverified && x.availability == "unverified")]
    isempty(candidates) && throw(auth_operation_error("$(d.id): no selectable login method here";
        reason="method_unavailable", stage="discovery", recovery="choose_method", provider=d.id))
    length(candidates) == 1 && return only(candidates)
    options = Tuple(SelectOption(x.id, x.label, something(x.billing_note, x.reason, Some(nothing))) for x in candidates)
    answer = try
        prompt(ui, SelectPrompt("method", "How do you want to connect to $(d.label)?", options))
    catch e
        e isa Union{InterruptException,EOFError,LoginCancelled} && throw(LoginCancelled("login cancelled at the method prompt"))
        rethrow()
    end
    for x in candidates
        x.id == answer && return x
    end
    throw(auth_operation_error("$(d.id): the UI answered an id that is not one of the offered methods";
        reason="invalid_login_state", stage="interaction", recovery="choose_method", provider=d.id))
end
function reserve_slot(a::Auth, provider, attempt_id, replace, lifetime_s)
    now = a.clock()
    document = mutate(a.store) do document
        slot, material = slot_view(document, provider)
        pending = slot.attempt
        if pending isa AbstractDict && get(pending, "id", nothing) != attempt_id
            started = Float64(get(pending, "started_at_s", 0))
            span = Float64(get(pending, "lifetime_s", ATTEMPT_LIFETIME_S))
            now - started < span && throw(auth_operation_error(
                "$provider: another sign-in is already in progress in this scope; finish it or cancel it (cancel_login)";
                reason="login_in_progress", stage="reservation", recovery="inspect_attempt", provider,
                attempt_id=get(pending, "id", nothing)))
        end
        slot.connection_id !== nothing && replace === nothing && throw(auth_operation_error(
            "$provider: a connection is already saved ($(slot.connection_id)); pass replace=that id to replace it, or logout first";
            reason="connection_exists", stage="reservation", recovery="select_connection", provider,
            connection_id=slot.connection_id))
        replace !== nothing && slot.connection_id != replace && throw(auth_operation_error(
            "$provider: replace does not name the current connection; select again";
            reason="connection_changed", stage="reservation", recovery="select_connection", provider,
            connection_id=slot.connection_id))
        slot.attempt = JSONObject("id" => attempt_id, "expected_generation" => string(slot.generation),
            "started_at_s" => now, "lifetime_s" => Float64(lifetime_s))
        slot_put!(document, slot, material)
    end
    return first(slot_view(document, provider)).generation
end
function release_slot(a::Auth, provider, attempt_id)
    try
        mutate(a.store) do document
            slot, material = slot_view(document, provider)
            slot.attempt isa AbstractDict && get(slot.attempt, "id", nothing) == attempt_id || return nothing
            slot.attempt = nothing
            slot_put!(document, slot, material)
        end
    catch e
        e isa LM15Error || rethrow()  # releasing a reservation must not mask the real failure
    end
    return nothing
end
function commit(a::Auth, provider, attempt_id, expected, m::LoginMethod, result::LoginResult, settings)
    created = iso_utc(a.clock())
    connection_id = "cn_$(token_urlsafe(12))"
    routes = collect(String, login_descriptor(provider).routes)
    isempty(routes) && (routes = [provider])
    document = try
        mutate(a.store) do document
            slot, _ = slot_view(document, provider)
            slot.attempt isa AbstractDict && get(slot.attempt, "id", nothing) == attempt_id || throw(auth_operation_error(
                "$provider: this sign-in was cancelled before it could be saved";
                reason="invalid_login_state", stage="persistence", commit_state="not_committed",
                recovery="restart_login", provider, attempt_id))
            slot.generation == expected || throw(auth_operation_error(
                "$provider: the saved connection changed while you were signing in; select again";
                reason="connection_changed", stage="persistence", commit_state="not_committed",
                recovery="select_connection", provider, attempt_id))
            merged = slot.connection_id === nothing ? merge(settings, result.settings) : merge(slot.settings, settings, result.settings)
            previous = slot.connection_id === nothing ? copy(slot.previous_ids) : vcat(slot.previous_ids, [slot.connection_id])
            new = empty_slot(provider)
            new.generation = slot.generation + 1
            new.connection_id = connection_id
            new.revision = 1
            new.kind = m.kind
            new.method_id = m.id
            new.label = result.label
            new.account_label = result.account_label
            new.created_at = created
            new.routes = routes
            new.settings = merged
            new.renewal = result.renewal
            new.previous_ids = previous
            slot_put!(document, new, result.material)
        end
    catch e
        e isa AuthOperationError && rethrow()
        e isa LM15Error || rethrow()
        # A grant may exist at the provider; nothing usable is returned and
        # nothing else is revoked as compensation (AUTH-19).
        throw(auth_operation_error(
            "$provider: signed in, but the credential could not be saved ($(error_code(e))); repair the store and sign in again";
            reason="storage_unavailable", stage="persistence", commit_state="not_committed", recovery="repair_storage",
            provider, attempt_id))
    end
    return slot_connection(first(slot_view(document, provider)))
end
"""
    cancel_login(auth, provider) -> "cancelled" | "complete" | "none"

Durably cancel the slot's active attempt (AUTH-19). `"complete"` means a commit already
won (undo is `logout`); `"none"` means nothing was pending.
"""
function cancel_login(a::Auth, provider)
    provider = descriptor(a, provider).id
    outcome = Ref("none")
    mutate(a.store) do document
        slot, material = slot_view(document, provider)
        if !(slot.attempt isa AbstractDict)
            outcome[] = slot.connection_id === nothing ? "none" : "complete"
            return nothing
        end
        slot.attempt = nothing
        outcome[] = "cancelled"
        slot_put!(document, slot, material)
    end
    flag = lock(() -> get(a.active, provider, nothing), a.lock)
    flag === nothing || (flag[] = true)
    return outcome[]
end

# ─── setup without a provider round-trip (AUTH-17) ─────────────────────
"""
    set_api_key(auth, provider, key; replace=nothing) -> Connection

Save a literal key (no interpolation, no network verification).
"""
function set_api_key(a::Auth, provider, key; replace=nothing)
    key isa AbstractString && !isempty(strip(key)) || throw(auth_operation_error("set_api_key: the key is empty";
        reason="interaction_required", stage="interaction", recovery="provide_input", provider=String(provider)))
    return configure(a, provider; method="api_key", answers=Dict("key" => key), replace)
end
"""
    configure(auth, provider; method, answers=Dict(), settings=Dict(), replace=nothing) -> Connection

Save a recipe connection: `env` (use `\$VAR` at request time), `external:<source>`
(another tool's login, read in place), `cloud` (a named cloud identity), `local` (a
keyless server) or `api_key`. No credential is acquired and nothing is verified.
"""
function configure(a::Auth, provider; method, answers=Dict{String,String}(), settings=Dict{String,String}(), replace=nothing)
    check_open(a)
    d = descriptor(a, provider)
    provider = d.id
    chosen = try
        LM15.method(d, method)
    catch e
        e isa KeyError || rethrow()
        throw(auth_operation_error("$provider: no setup method $(repr(method)); see login_methods($(repr(provider)))";
            reason="method_unavailable", stage="discovery", recovery="choose_method", provider))
    end
    chosen.flow in ("form", "source_recipe") || throw(auth_operation_error(
        "$provider: $(repr(method)) is an interactive login; use login";
        reason="method_unavailable", stage="discovery", recovery="choose_method", provider))
    answers = Dict{String,String}(String(k) => String(v) for (k, v) in something(answers, Dict()))
    settings = Dict{String,String}(String(k) => String(v) for (k, v) in something(settings, Dict()))
    for f in chosen.fields
        f.required && isempty(get(answers, f.id, "")) && throw(auth_operation_error("$provider: $(repr(method)) needs $(repr(f.id))";
            reason="interaction_required", stage="interaction", recovery="provide_input", provider))
    end
    attempt_id = "at_$(token_urlsafe(16))"
    reserve!(a.store)
    expected = reserve_slot(a, provider, attempt_id, replace, 60.0)
    ctx = LoginContext(NoUI(), a.monotonic() + 60.0, Ref(false), provider, a.monotonic, a.clock, a.opener, nothing)
    result = try
        flow_login(login_flow(provider, method), ctx, chosen, settings, answers)
    catch e
        release_slot(a, provider, attempt_id)
        e isa LoginDenied && throw(auth_operation_error("$provider: $(e.message)";
            reason="login_denied", stage="interaction", recovery="provide_input", provider))
        rethrow()
    end
    return commit(a, provider, attempt_id, expected, chosen, result, settings)
end

# ─── logout (AUTH-19) ──────────────────────────────────────────────────
"""
    logout(auth, provider_or_connection_id) -> ForgetResult

Forget the connection locally: material removed, generation bumped, a pending attempt
cancelled, and a suppression marker kept so a restart cannot fall back to an ambient key
(R3). Never calls a provider's revoke endpoint; never touches another tool's file.
Repeating it, or naming an old id, never removes a newer connection.
"""
function logout(a::Auth, target::AbstractString)
    check_open(a)
    provider, target_id = resolve_target(a, target)
    outcome = Dict{Symbol,Any}()
    mutate(a.store) do document
        slot, material = slot_view(document, provider)
        if target_id !== nothing && slot.connection_id != target_id
            merge!(outcome, Dict(:forgot => false, :generation => slot.generation, :routes => slot.routes))
            return nothing  # idempotent: a newer id occupying the slot is untouched
        end
        if slot.connection_id === nothing && !(slot.attempt isa AbstractDict)
            merge!(outcome, Dict(:forgot => false, :generation => slot.generation, :routes => slot.routes))
            return nothing
        end
        new = empty_slot(provider)
        new.generation = slot.generation + 1
        new.kind = slot.kind
        new.method_id = slot.method_id
        new.routes = isempty(slot.routes) ? [provider] : copy(slot.routes)
        new.renewal = "none"
        new.logged_out = true
        new.previous_ids = slot.connection_id === nothing ? copy(slot.previous_ids) : vcat(slot.previous_ids, [slot.connection_id])
        merge!(outcome, Dict(:forgot => true, :generation => new.generation, :routes => new.routes))
        flag = lock(() -> get(a.active, provider, nothing), a.lock)
        flag === nothing || (flag[] = true)
        slot_put!(document, new, nothing)
    end
    routes = get(outcome, :routes, String[])
    return ForgetResult(provider, get(outcome, :forgot, false), Tuple(isempty(routes) ? [provider] : routes),
        string(get(outcome, :generation, 0)))
end
function resolve_target(a::Auth, target)
    if startswith(target, "cn_") || startswith(target, "legacy-")
        for c in connections(a)
            c.id == target && return c.provider, c.id
        end
        document = read_document(a.store)
        meta = get(document, META_KEY, obj())
        for (key, record) in (meta isa AbstractDict ? get(meta, "slots", obj()) : obj())
            target in get(record, "previous_ids", []) && return key, target  # an old id: a no-op, never a newer id's removal
        end
        throw(auth_operation_error("no saved connection has that id";
            reason="attempt_unavailable", stage="resolution", recovery="select_connection"))
    end
    return descriptor(a, target).id, nothing
end

# ─── verification (AUTH-17) ────────────────────────────────────────────
"""
    verify(auth, provider; router_config=RouterConfig()) -> Verification

An explicit non-inference check: resolve the saved credential (renewing if due) and list
the route's models. Not universal, possibly metered by the provider.
"""
function verify(a::Auth, provider; router_config=RouterConfig())
    check_open(a)
    provider = descriptor(a, provider).id
    definition = get(PROVIDERS, provider, nothing)
    (definition === nothing || !definition.access.supports.models) &&
        return Verification("unverified"; check="models", detail="this route has no safe non-inference check")
    router = LMRouter(with_auth(router_config, a))
    checked = iso_utc(a.clock())
    result = try
        list_models(lm(router, "$provider:verify"))
        Verification("valid"; checked_at=checked, check="models")
    catch e
        e isa AuthError || rethrow()
        Verification("rejected"; checked_at=checked, check="models", detail=error_code(e))
    finally
        close(router)
    end
    try
        mutate(a.store) do document
            slot, material = slot_view(document, provider)
            slot.connection_id === nothing && return nothing
            slot.verification = JSONObject("result" => result.result, "checked_at" => result.checked_at,
                "check" => result.check, "detail" => result.detail)
            slot_put!(document, slot, material)
        end
    catch e
        e isa LM15Error || rethrow()
    end
    return result
end

# ─── request-time resolution (AUTH-15/20) ──────────────────────────────
"""
    request_auth(auth, provider; pinned=nothing) -> RequestAuth

What a request on `provider` sends now: the saved connection's credential, renewed under
the lock if due. `pinned` is a bound client's `(connection_id, generation)`; a mismatch is
`connection_changed`, never a silent rebind (AUTH-20.1).
"""
function request_auth(a::Auth, provider; pinned=a.pin)
    provider = descriptor(a, provider).id
    slot, material = slot_view(read_document(a.store), provider)
    # A sibling may be renewing now, or died mid-exchange: only the lock can tell.
    slot.renewal_in_flight !== nothing && slot.state != "indeterminate" && return renew(a, provider, pinned)
    check_selected(provider, slot, material, pinned)
    flow = flow_for_material(provider, material)
    expiry = flow_expiry(flow, material)
    (expiry == "never" || expiry === nothing) && return auth_from(provider, flow, material, slot)
    round(Int64, a.clock() * 1000) < expiry - lead_ms(flow_lifetime_s(flow, material)) &&
        return auth_from(provider, flow, material, slot)
    return renew(a, provider, pinned)
end
function check_selected(provider, slot, material, pinned)
    if pinned !== nothing && (slot.connection_id != pinned[1] || string(slot.generation) != pinned[2])
        slot.connection_id === nothing && throw(auth_operation_error(
            "$provider: the connection this client was bound to was signed out; connect again";
            reason="login_required", stage="resolution", recovery="restart_login", provider, connection_id=pinned[1]))
        throw(auth_operation_error("$provider: the saved connection was replaced after this client was bound; connect again";
            reason="connection_changed", stage="resolution", recovery="select_connection", provider, connection_id=pinned[1]))
    end
    if slot.connection_id === nothing || material === nothing
        throw(auth_operation_error(slot.logged_out ?
            "$provider: signed out; sign in again (login) or pass a key explicitly (api_keys)" :
            "$provider: no saved connection in this scope; sign in with login or connect()";
            reason="login_required", stage="resolution", recovery="restart_login", provider))
    end
    slot.state == "needs_login" && throw(auth_operation_error("$provider: the saved credential was rejected by the provider; sign in again";
        reason="login_required", stage="resolution", recovery="restart_login", provider, connection_id=slot.connection_id))
    (slot.state == "indeterminate" || slot.renewal_in_flight !== nothing) && throw(auth_operation_error(
        "$provider: a credential renewal was interrupted and its outcome is unknown; sign in again rather than reuse a possibly consumed token";
        reason="indeterminate", stage="resolution", commit_state="unknown", recovery="restart_login", provider,
        connection_id=slot.connection_id))
    return nothing
end
function auth_from(provider, flow, material, slot)
    try
        return flow_request_auth(flow, material, slot.settings)
    catch e
        e isa LoginDenied || rethrow()
        throw(auth_operation_error("$provider: $(e.message)"; reason="login_required", stage="resolution",
            recovery="restart_login", provider, connection_id=slot.connection_id))
    end
end
function mark!(txn, document, slot, material; state, drop_material=false, keep_marker=false)
    slot.state = state
    keep_marker || (slot.renewal_in_flight = nothing)
    write_document!(txn, slot_put!(document, slot, drop_material ? nothing : material))
end
"""AUTH-20.4: lock, re-read, reuse a sibling's fresh result, else mark in-flight,
exchange, write, all under the lock."""
function renew(a::Auth, provider, pinned)
    ctx = LoginContext(NoUI(), a.monotonic() + 60.0, Ref(false), provider, a.monotonic, a.clock, a.opener, nothing)
    return transaction(a.store) do txn
        document = read_document(txn)
        slot, material = slot_view(document, provider)
        check_selected(provider, slot, material, pinned)
        flow = flow_for_material(provider, material)
        expiry = flow_expiry(flow, material)
        now = round(Int64, a.clock() * 1000)
        (expiry == "never" || expiry === nothing || now < expiry - lead_ms(flow_lifetime_s(flow, material))) &&
            return auth_from(provider, flow, material, slot)  # a sibling renewed while we waited
        if !renewable(slot, material)
            mark!(txn, document, slot, material; state="needs_login", drop_material=true)
            throw(auth_operation_error("$provider: the saved credential expired and cannot be renewed; sign in again";
                reason="credential_rejected", stage="renewal", commit_state="committed", recovery="restart_login",
                provider, connection_id=slot.connection_id))
        end
        # A durable in-flight marker before the possibly rotating exchange.
        slot.renewal_in_flight = JSONObject("started_at" => iso_utc(a.clock()), "revision" => string(slot.revision))
        write_document!(txn, slot_put!(document, slot, material))
        result = try
            flow_renew(flow, ctx, material, slot.settings)
        catch e
            if e isa LoginDenied
                mark!(txn, document, slot, material; state="needs_login", drop_material=true)
                throw(auth_operation_error("$provider: renewal failed ($(e.message)); sign in again";
                    reason="credential_rejected", stage="renewal", commit_state="committed", recovery="restart_login",
                    provider, connection_id=slot.connection_id, status=e.status, provider_code=e.provider_code))
            elseif e isa Union{RateLimitError,ServerError}
                mark!(txn, document, slot, material; state="ready")  # known safe: keep the credential
                rethrow()
            elseif e isa AuthTransportFailure
                if e.uncertain
                    mark!(txn, document, slot, material; state="indeterminate", keep_marker=true)
                    throw(auth_operation_error(
                        "$provider: the renewal exchange timed out after it may have reached the provider; a rotated token cannot be spent twice, so sign in again";
                        reason="indeterminate", stage="renewal", commit_state="unknown", recovery="restart_login",
                        provider, connection_id=slot.connection_id))
                end
                mark!(txn, document, slot, material; state="ready")
                throw(TransportError("$provider: network failure during a credential renewal"; provider))
            end
            mark!(txn, document, slot, material; state="indeterminate", keep_marker=true)
            rethrow()
        end
        slot.renewal_in_flight = nothing
        slot.revision += 1
        slot.state = "ready"
        result.account_label === nothing || (slot.account_label = result.account_label)
        write_document!(txn, slot_put!(document, slot, result.material))
        return auth_from(provider, flow, result.material, slot)
    end
end
"""
    credential_provider(auth, provider; pinned=nothing)

A zero-argument callable adapters resolve per request (AUTH-2): each call is
`request_auth(auth, provider).credential`.
"""
function credential_provider(a::Auth, provider; pinned=a.pin)
    provider = descriptor(a, provider).id
    return ManagedCredential(a, provider, pinned)
end
struct ManagedCredential
    auth::Auth
    provider::String
    pinned::Any
end
(m::ManagedCredential)() = request_auth(m.auth, m.provider; pinned=m.pinned).credential
Base.show(io::IO, m::ManagedCredential) = print(io, "<managed credential for ", m.provider, ">")
"""A copy of `a` (same store, same seams, same in-process attempts) whose every
request-time resolution is checked against one `(connection_id, generation)`."""
pinned_auth(a::Auth, s::ModelSelection) =
    Auth(a.store, a.clock, a.monotonic, a.opener, a.sleep, a.active, a.lock, false, (s.connection_id, s.identity_generation))

"""A UI that cannot answer: any prompt is `interaction_required`."""
struct NoUI <: AuthUI end
prompt(::NoUI, ::Prompt) = throw(auth_operation_error("this operation needs a choice or input and no UI was supplied";
    reason="interaction_required", stage="interaction", recovery="provide_input"))
Base.notify(::NoUI, ::Notice) = nothing
