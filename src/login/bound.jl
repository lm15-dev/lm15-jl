# Model choices, the bound client, the terminal UI and `connect()` (spec/auth-managed.md
# AUTH-16, AUTH-23). A BoundClient is one connection id, one generation, one route, one
# model: it follows that connection's renewals and nothing else — a replacement or logout
# makes it fail `connection_changed` / `login_required` instead of switching who pays. It
# builds ordinary canonical Requests and returns ordinary Responses and events; it keeps
# no conversation, runs no tool loop, retries nothing.

const MODEL_CAPABILITIES = ("reasoning", "vision", "structured-output")
function capability_of(info, name)
    info.inference === nothing && return "unknown"
    name == "reasoning" && return info.inference.supports_reasoning ? "supported" : "unsupported"
    name == "vision" && return "image" in info.inference.input_modalities ? "supported" : "unsupported"
    return "unknown"  # structured output is not recorded in ModelInfo; say so
end
"""
    model_choices(auth, provider; refresh=false, capability=nothing, include_unknown=false,
                  registry=nothing, router_config=RouterConfig()) -> Vector{ModelChoice}

The models the saved connection on `provider` can select (AUTH-23). `refresh=false` reads
only a caller-supplied `ModelRegistry` (source `application`); `refresh=true` fetches the
account's own list with the saved credential (source `provider`; renews if due; no
inference). With `capability`, only `supported` choices are returned unless
`include_unknown` is set.
"""
function model_choices(a::Auth, provider; refresh=false, capability=nothing, include_unknown=false,
    registry=nothing, router_config=RouterConfig())
    capability === nothing || capability in MODEL_CAPABILITIES ||
        throw(ArgumentError("capability must be one of $(join(MODEL_CAPABILITIES, ", "))"))
    connection = status(a, provider).connection
    connection === nothing && throw(auth_operation_error("$provider: no saved connection to list models for";
        reason="login_required", stage="catalog", recovery="restart_login", provider=String(provider)))
    infos = ModelInfo[]
    source = "application"
    fetched = nothing
    if refresh
        router = LMRouter(with_auth(router_config, a))
        try
            infos = list_models(lm(router, "$(connection.provider):catalog"))
        finally
            close(router)
        end
        source, fetched = "provider", iso_utc(time())
    elseif registry !== nothing
        infos = list_models(registry; provider=connection.provider)
    end
    out = ModelChoice[]
    for info in infos
        caps = capability === nothing ? Dict{String,String}() : Dict(capability => capability_of(info, capability))
        capability !== nothing && caps[capability] == "unsupported" && continue
        capability !== nothing && caps[capability] == "unknown" && !include_unknown && continue
        push!(out, ModelChoice(connection.provider, info.id, connection.id, source, info.id, fetched, caps))
    end
    return out
end

"""
    BoundClient(auth, selection; router_config=RouterConfig())

One connection, one model (AUTH-23): `request`, `complete`, `stream` and `plan` build and
send ordinary canonical values for exactly that selection. Closing it is not a logout.
"""
struct BoundClient
    auth::Auth
    selection::ModelSelection
    router::LMRouter
end
function BoundClient(a::Auth, s::ModelSelection; router_config=RouterConfig())
    router_config.auth === nothing || router_config.auth === a ||
        throw(ArgumentError("router_config.auth must be this client's Auth or nothing"))
    return BoundClient(a, s, LMRouter(with_auth(router_config, pinned_auth(a, s))))
end
Base.show(io::IO, b::BoundClient) = print(io, "BoundClient(", repr(routed(b.selection)), ", connection=", repr(b.selection.connection_id), ")")
"""
    request(client::BoundClient, messages; tools=(), config=Config(), system=nothing)

The canonical Request a call would send, with the selected routed model (pure; no I/O).
"""
function request(b::BoundClient, messages; tools=(), config=Config(), system=nothing)
    messages isa AbstractString && (messages = (user(messages),))
    messages isa Message && (messages = (messages,))
    return Request(; model=routed(b.selection), messages=Tuple(messages), tools=Tuple(tools), config, system)
end
function coerce(b::BoundClient, r)
    r isa Request || return request(b, r)
    r.model in (routed(b.selection), b.selection.model) || throw(auth_operation_error(
        "this client is bound to $(repr(routed(b.selection))); the Request names $(repr(r.model))";
        reason="selection_mismatch", stage="dispatch", recovery="none", provider=b.selection.provider))
    return r.model == routed(b.selection) ? r : reconstruct(r; model=routed(b.selection))
end
complete(b::BoundClient, r; kw...) = complete(b.router, isempty(kw) ? coerce(b, r) : request(b, r; kw...))
stream(b::BoundClient, r; kw...) = stream(b.router, isempty(kw) ? coerce(b, r) : request(b, r; kw...))
plan(b::BoundClient, r; kw...) = plan(b.router, isempty(kw) ? coerce(b, r) : request(b, r; kw...))
Base.close(b::BoundClient) = close(b.router)
function BoundClient(f::Function, args...; kw...)
    b = BoundClient(args...; kw...)
    try
        return f(b)
    finally
        close(b)
    end
end

"""
    TerminalUI(; out=stderr, input=stdin, open_browser=false)

The terminal adapter for AUTH-16: notices to `out`, answers from `input`. It opens a
browser only when `open_browser=true` (an https URL only); otherwise it prints the link,
which is what a machine reached over SSH needs. End of input cancels.
"""
Base.@kwdef struct TerminalUI <: AuthUI
    out::IO = stderr
    input::IO = stdin
    open_browser::Bool = false
end
say(ui::TerminalUI, text) = (println(ui.out, text); flush(ui.out))
function Base.notify(ui::TerminalUI, n::Notice)
    if n isa AuthUrlNotice
        say(ui, "\nOpen this link to sign in:\n  $(n.url)\n$(n.instructions)")
        ui.open_browser && open_url(n.url)
    elseif n isa DeviceCodeNotice
        say(ui, "\nOpen $(n.verification_url)\nand enter this code:  $(n.user_code)\n(the code is valid for about $(floor(Int, n.expires_in_s ÷ 60)) minutes)")
        ui.open_browser && open_url(n.verification_url)
    elseif n isa ProgressNotice
        say(ui, "… $(n.message)")
    elseif n isa InfoNotice
        say(ui, n.message * join(("\n  $label: $url" for (label, url) in n.links)))
    end
    return nothing
end
function open_url(url)
    startswith(url, "https://") || return nothing  # AUTH-18: never launch a scheme a provider chose
    try
        cmd = Sys.isapple() ? `open $url` : Sys.iswindows() ? `cmd /c start "" $url` : `xdg-open $url`
        run(pipeline(cmd; stdout=devnull, stderr=devnull); wait=false)
    catch
    end
    return nothing
end
function read_answer(ui::TerminalUI, label; secret=false)
    print(ui.out, label)
    flush(ui.out)
    eof(ui.input) && throw(LoginCancelled("end of input"))
    if secret && ui.input === stdin && isa(stdin, Base.TTY)
        return String(read(Base.getpass(""), String))
    end
    return readline(ui.input)
end
function prompt(ui::TerminalUI, p::Prompt)
    if p isa SecretPrompt
        return read_answer(ui, "$(p.label): "; secret=true)
    elseif p isa SelectPrompt
        say(ui, "\n$(p.label)")
        for (i, o) in enumerate(p.options)
            say(ui, "  $i. $(o.label)" * (o.description === nothing ? "" : "  — $(o.description)"))
        end
        while true
            raw = strip(read_answer(ui, "Choose a number: "))
            n = tryparse(Int, raw)
            n !== nothing && 1 <= n <= length(p.options) && return p.options[n].id
            for o in p.options
                raw == o.id && return o.id
            end
            say(ui, "Not one of the choices.")
        end
    elseif p isa ManualCodePrompt
        return read_answer(ui, "$(p.label)\n> ")
    end
    return read_answer(ui, "$(p.label)" * (p.placeholder === nothing ? "" : " [$(p.placeholder)]") * ": ")
end
dismiss(ui::TerminalUI, ::Prompt) = say(ui, "\n(the browser finished — press Enter to continue)")

const CONNECT_NEW = "__new__"
const CONNECT_MANUAL = "__manual__"
interactive_terminal() = stdin isa Base.TTY && stderr isa Base.TTY
function connect_ask(ui, p)
    try
        return prompt(ui, p)
    catch e
        e isa Union{InterruptException,EOFError,LoginCancelled} && throw(auth_operation_error("cancelled";
            reason="interaction_required", stage="interaction", recovery="restart_login"))
        rethrow()
    end
end
"""
    connect(provider=nothing; model=nothing, auth=local_auth(), ui=nothing, capability=nothing,
            open_browser=false, router_config=RouterConfig(), allow_unverified=false) -> BoundClient
    connect(f, provider=nothing; kw...)

Choose (or make) a connection and a model, and return a client bound to them (AUTH-23).
Shows where connections are saved; offers saved connections and "connect another"
(subscriptions first; an ambient key is offered only as an explicit choice); runs the
chosen login or setup; fetches the account's models (or takes `model`) and asks. A
completed login is saved before the model picker runs. It never sends a prompt, sets a
process default, or falls back to another account. Without a terminal it needs `ui`.
"""
function connect(provider=nothing; model=nothing, auth=nothing, ui=nothing, capability=nothing,
    open_browser=false, router_config=RouterConfig(), allow_unverified=false)
    if ui === nothing
        interactive_terminal() || throw(auth_operation_error(
            "connect() needs a person: no interactive terminal here and no ui= was supplied. On a server, attach an Auth with saved connections (RouterConfig(auth=...)) instead of calling connect().";
            reason="interaction_required", stage="interaction", recovery="provide_input"))
        ui = TerminalUI(; open_browser)
    end
    auth = something(auth, local_auth())
    notify(ui, InfoNotice("Connections are saved privately in $(description(auth.store))."))
    connection = connect_connection(auth, ui, provider; allow_unverified)
    selection = connect_model(auth, ui, connection; model, capability, router_config)
    notify(ui, InfoNotice("Ready: $(routed(selection)) through $(connection.label)."))
    return BoundClient(auth, selection; router_config)
end
function connect(f::Function, args...; kw...)
    b = connect(args...; kw...)
    try
        return f(b)
    finally
        close(b)
    end
end
function connect_connection(a::Auth, ui, provider; allow_unverified)
    wanted = provider === nothing ? nothing : descriptor(a, provider).id
    saved = [c for c in connections(a) if wanted === nothing || c.provider == wanted]
    sort!(saved; by=c -> (c.kind == "account" ? 0 : 1, c.provider))  # subscriptions first (R2)
    usable = [c for c in saved if status(a, c.provider).usability in ("ready", "renewal_due", "unknown")]
    provider !== nothing && length(usable) == 1 && return only(usable)
    options = [SelectOption(c.id, c.label, "$(c.provider) · saved") for c in usable]
    push!(options, SelectOption(CONNECT_NEW, "Connect another account or API key"))
    length(options) == 1 && return connect_new(a, ui, provider; allow_unverified)
    answer = connect_ask(ui, SelectPrompt("connection", "Use a saved connection, or connect another?", Tuple(options)))
    answer == CONNECT_NEW && return connect_new(a, ui, provider; allow_unverified)
    for c in usable
        c.id == answer && return c
    end
    throw(auth_operation_error("the UI answered with an unknown connection id";
        reason="invalid_login_state", stage="interaction", recovery="select_connection"))
end
function connect_new(a::Auth, ui, provider; allow_unverified)
    if provider === nothing
        ds = [d for d in login_providers() if any(m -> m.availability != "unavailable", d.methods)]
        sort!(ds; by=d -> (any(m -> m.subscription && m.availability == "supported", d.methods) ? 0 : 1, lowercase(d.label)))
        provider = connect_ask(ui, SelectPrompt("provider", "Which provider?",
            Tuple(SelectOption(d.id, d.label, d.service == d.label ? nothing : d.service) for d in ds)))
    end
    d = descriptor(a, provider)
    existing = status(a, d.id).connection
    m = connect_method(ui, d; allow_unverified)
    replace = nothing
    if existing !== nothing
        answer = connect_ask(ui, SelectPrompt("replace", "$(d.label) already has a saved connection ($(existing.label)).",
            (SelectOption("keep", "Keep it"), SelectOption("replace", "Replace it"))))
        answer == "keep" && return existing
        replace = existing.id
    end
    if m.flow in ("form", "source_recipe")
        answers = Dict{String,String}()
        for f in m.fields
            answers[f.id] = if f.type == "select" && length(f.options) == 1
                only(f.options).id
            elseif f.type == "select"
                connect_ask(ui, SelectPrompt(f.id, f.label, f.options))
            elseif f.type == "secret"
                connect_ask(ui, SecretPrompt(f.id, f.label))
            else
                connect_ask(ui, TextPrompt(f.id, f.label))
            end
        end
        return configure(a, d.id; method=m.id, answers, replace)
    end
    try
        return login(a, d.id, m.id; ui, replace, allow_unverified)
    catch e
        e isa LoginCancelled || rethrow()
        throw(auth_operation_error("sign-in cancelled"; reason="interaction_required", stage="interaction",
            recovery="restart_login", provider=d.id))
    end
end
function connect_method(ui, d::ProviderDescriptor; allow_unverified)
    ms = [m for m in d.methods if m.availability == "supported" || (allow_unverified && m.availability == "unverified")]
    isempty(ms) && throw(auth_operation_error("$(d.id): no login method is available here";
        reason="method_unavailable", stage="discovery", recovery="choose_method", provider=d.id))
    sort!(ms; by=m -> (m.subscription ? 0 : 1, m.kind == "account" ? 0 : 1))  # an ambient key is offered, never assumed
    options = SelectOption[]
    for m in ms
        note = m.availability == "unverified" ? "UNVERIFIED — $(m.reason)" : m.billing_note
        if m.id == "env"
            names = isempty(m.fields) ? String[] : [o.id for o in m.fields[1].options if !isempty(get(ENV, o.id, ""))]
            isempty(names) && continue
            note = "\$$(first(names)) is set in this environment; using it is your explicit choice"
        end
        push!(options, SelectOption(m.id, m.label, note))
    end
    chosen = length(options) == 1 ? only(options).id :
        connect_ask(ui, SelectPrompt("method", "How do you want to connect to $(d.label)?", Tuple(options)))
    return method(d, chosen)
end
function connect_model(a::Auth, ui, c::Connection; model, capability, router_config)
    model === nothing || return ModelSelection(c.provider, model, c.id, c.identity_generation)
    choices = ModelChoice[]
    note = nothing
    try
        choices = model_choices(a, c.provider; refresh=true, capability, router_config)
        note = "listed by your account just now"
    catch e
        e isa AuthOperationError && rethrow()
        e isa InterruptException && rethrow()
        # The catalog is a convenience: say why it is missing, never pretend.
        notify(ui, InfoNotice("Could not list models for $(c.provider) ($(nameof(typeof(e)))); type a model id."))
    end
    options = [SelectOption(x.model, x.model, note) for x in choices]
    push!(options, SelectOption(CONNECT_MANUAL, "Type a model id (not verified against your account)"))
    capability !== nothing && isempty(choices) &&
        notify(ui, InfoNotice("No model in the list is known to support $(repr(capability)); you can still type one."))
    answer = connect_ask(ui, SelectPrompt("model", "Which $(c.provider) model?", Tuple(options)))
    if answer == CONNECT_MANUAL
        answer = strip(connect_ask(ui, TextPrompt("model", "Model id")))
        isempty(answer) && throw(auth_operation_error("no model id given"; reason="interaction_required",
            stage="interaction", recovery="provide_input", provider=c.provider))
    end
    return ModelSelection(c.provider, String(answer), c.id, c.identity_generation)
end
