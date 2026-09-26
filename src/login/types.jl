# The public vocabulary of managed authentication (spec/auth-managed.md AUTH-12
# vocabulary, AUTH-13 descriptors, AUTH-16 the UI boundary, AUTH-23 model choices,
# AUTH-24 status). Every value here is secret-free by construction: a Connection is
# metadata about a saved credential, never the credential.

const CONNECTION_KINDS = ("account", "api_key", "cloud_identity", "local_server")
const LOGIN_FLOWS = ("authorization_code", "device_code", "form", "source_recipe")
const METHOD_AVAILABILITY = ("supported", "unavailable", "unverified")
const USABILITY = ("ready", "renewal_due", "needs_login", "indeterminate", "unknown")

"""
    SelectOption(id, label, description=nothing)

One choice of a select prompt or field. The answer to a select is the option `id`.
"""
struct SelectOption
    id::String
    label::String
    description::Maybe{String}
end
SelectOption(id, label) = SelectOption(id, label, nothing)
"""
    MethodField(id, label; type="text", required=true, options=(), help=nothing)

One input a login method needs before it can start (AUTH-13.3).
"""
struct MethodField
    id::String
    label::String
    type::String
    required::Bool
    options::Tuple
    help::Maybe{String}
end
MethodField(id, label; type="text", required=true, options=(), help=nothing) =
    MethodField(id, label, type, required, Tuple(options), help)
"""
    LoginMethod

A named way to establish a connection for a provider (AUTH-13.3): `id`, `label`, `kind`
(`account`, `api_key`, `cloud_identity`, `local_server`), `flow` (`authorization_code`,
`device_code`, `form`, `source_recipe`), `availability` (`supported`, `unverified`,
`unavailable`) with a `reason`, input `fields`, supported `delivery` modes (`loopback`,
`manual`, `device`), whether it is `subscription`-backed per provider docs, a
`billing_note` (known policy, never an entitlement promise) and `guidance`.
"""
struct LoginMethod
    id::String
    label::String
    kind::String
    flow::String
    availability::String
    reason::Maybe{String}
    fields::Tuple
    delivery::Tuple
    subscription::Bool
    billing_note::Maybe{String}
    guidance::Maybe{String}
end
function LoginMethod(id, label, kind, flow; availability="supported", reason=nothing, fields=(),
    delivery=(), subscription=false, billing_note=nothing, guidance=nothing)
    kind in CONNECTION_KINDS || throw(ArgumentError("unknown connection kind $(repr(kind))"))
    flow in LOGIN_FLOWS || throw(ArgumentError("unknown login flow $(repr(flow))"))
    availability in METHOD_AVAILABILITY || throw(ArgumentError("unknown availability $(repr(availability))"))
    return LoginMethod(id, label, kind, flow, availability, reason, Tuple(fields), Tuple(delivery),
        subscription, billing_note, guidance)
end
selectable(m::LoginMethod) = m.availability != "unavailable"
"""
    ProviderDescriptor

A provider a manager can connect (AUTH-13.1): the LM15 route `id`, a display `label`, a
presentation `service` group, the `routes` a connection authenticates, and its `methods`.
"""
struct ProviderDescriptor
    id::String
    label::String
    service::String
    routes::Tuple
    methods::Tuple
    docs_url::Maybe{String}
    console_url::Maybe{String}
end
function method(d::ProviderDescriptor, id::AbstractString)
    for m in d.methods
        m.id == id && return m
    end
    throw(KeyError(id))
end
Base.show(io::IO, d::ProviderDescriptor) = print(io, "ProviderDescriptor(", repr(d.id), ", ", length(d.methods), " methods)")
Base.show(io::IO, m::LoginMethod) = print(io, "LoginMethod(", repr(m.id), ", ", repr(m.kind), ", ", m.availability, ")")

"""
    Connection

Secret-free metadata for one saved credential or source recipe in a scope (AUTH-12):
`id`, `provider`, `instance_id`, `kind`, `method_id`, `routes`, `label`, `created_at`,
`identity_generation`, `credential_revision`, non-secret `settings`, and an optional
`account_label` (untrusted display text, never verified).
"""
struct Connection
    id::String
    provider::String
    instance_id::String
    kind::String
    method_id::String
    routes::Tuple
    label::String
    created_at::String
    identity_generation::String
    credential_revision::String
    settings::Dict{String,String}
    account_label::Maybe{String}
end
function Base.show(io::IO, c::Connection)
    print(io, "Connection(id=", repr(c.id), ", provider=", repr(c.provider), ", kind=", repr(c.kind),
        ", method=", repr(c.method_id), ", label=", repr(c.label), ")")
end
"""
    Verification(result; checked_at=nothing, check=nothing, detail=nothing)

The outcome of an explicit `verify`: `valid`, `rejected` or `unverified`, for the check
named, at a time. Never a promise about future requests.
"""
struct Verification
    result::String
    checked_at::Maybe{String}
    check::Maybe{String}
    detail::Maybe{String}
end
Verification(result; checked_at=nothing, check=nothing, detail=nothing) = Verification(result, checked_at, check, detail)
"""
    ConnectionStatus

AUTH-24: `presence` (`saved`, `absent`), `usability` (`ready`, `renewal_due`,
`needs_login`, `indeterminate`, `unknown`), the connection, `expires_at` (RFC 3339,
`"never"` or `"unknown"`), whether the slot was `logged_out`, and the last explicit
`verification`. `ready` is a local assessment, not remote verification.
"""
struct ConnectionStatus
    provider::String
    presence::String
    usability::String
    connection::Maybe{Connection}
    expires_at::Maybe{String}
    logged_out::Bool
    verification::Maybe{Verification}
    detail::Maybe{String}
end
is_ready(s::ConnectionStatus) = s.usability in ("ready", "renewal_due")

# The UI boundary (AUTH-16). A prompt returns the answer (a select's answer is the
# option id); a notice is shown and returns.
abstract type Prompt end
abstract type Notice end
"""A text prompt: `field_id`, `label`, an optional `placeholder`."""
struct TextPrompt <: Prompt
    field_id::String
    label::String
    placeholder::Maybe{String}
end
TextPrompt(field_id, label) = TextPrompt(field_id, label, nothing)
"""A secret prompt (keys, passwords): the answer is private input."""
struct SecretPrompt <: Prompt
    field_id::String
    label::String
    placeholder::Maybe{String}
end
SecretPrompt(field_id, label) = SecretPrompt(field_id, label, nothing)
"""A choice among `options` (`SelectOption`s); the answer is the chosen option id."""
struct SelectPrompt <: Prompt
    field_id::String
    label::String
    options::Tuple
end
"""Paste the redirect URL or code when the browser cannot reach the loopback listener."""
struct ManualCodePrompt <: Prompt
    field_id::String
    label::String
    accepted::String
end
ManualCodePrompt(field_id, label) = ManualCodePrompt(field_id, label, "the full redirect URL, or the code")
prompt_type(::TextPrompt) = "text"
prompt_type(::SecretPrompt) = "secret"
prompt_type(::SelectPrompt) = "select"
prompt_type(::ManualCodePrompt) = "manual_code"
"""Open this authorization URL (session-sensitive: show it to the person signing in only)."""
struct AuthUrlNotice <: Notice
    url::String
    instructions::String
end
"""Enter `user_code` at `verification_url` before `expires_in_s` seconds pass."""
struct DeviceCodeNotice <: Notice
    user_code::String
    verification_url::String
    expires_in_s::Float64
    interval_s::Float64
end
"""A safe progress stage (`exchange`, ...) with a human message."""
struct ProgressNotice <: Notice
    stage::String
    message::String
end
"""Safe guidance, with optional `(label, url)` links."""
struct InfoNotice <: Notice
    message::String
    links::Tuple
end
InfoNotice(message) = InfoNotice(message, ())
notice_type(::AuthUrlNotice) = "auth_url"
notice_type(::DeviceCodeNotice) = "device_code"
notice_type(::ProgressNotice) = "progress"
notice_type(::InfoNotice) = "info"
Base.show(io::IO, ::AuthUrlNotice) = print(io, "AuthUrlNotice(<session-sensitive URL withheld>)")
Base.show(io::IO, n::DeviceCodeNotice) = print(io, "DeviceCodeNotice(<user code withheld>, ", repr(n.verification_url), ")")

"""
    AuthUI

What an application supplies to let a login talk to a person (AUTH-16). Subtype it and
implement `LM15.prompt(ui, p::Prompt) -> String` (throw `LoginCancelled` or
`InterruptException` to cancel) and `Base.notify(ui, n::Notice)`; optionally
`LM15.dismiss(ui, p)` for a prompt made stale by a loopback callback. Or pass functions to
[`FunctionUI`](@ref). A UI opens a browser only when the application chose that.
"""
abstract type AuthUI end
function prompt end
dismiss(::AuthUI, ::Prompt) = nothing
"""
    FunctionUI(; prompt, notify=(n)->nothing, dismiss=(p)->nothing)

An `AuthUI` from plain functions: `prompt(p::Prompt) -> String`, `notify(n::Notice)`.
"""
Base.@kwdef struct FunctionUI <: AuthUI
    prompt::Function
    notify::Function = n -> nothing
    dismiss::Function = p -> nothing
end
prompt(ui::FunctionUI, p::Prompt) = ui.prompt(p)
Base.notify(ui::FunctionUI, n::Notice) = (ui.notify(n); nothing)
dismiss(ui::FunctionUI, p::Prompt) = (ui.dismiss(p); nothing)

"""
    ModelChoice

A model the saved connection can select (AUTH-23), with where it came from (`source`:
`application`, `provider`, `cached`, `bundled`, `manual`), when it was fetched, and the
requested capabilities (`supported`, `unsupported`, `unknown`).
"""
struct ModelChoice
    provider::String
    model::String
    connection_id::String
    source::String
    label::Maybe{String}
    fetched_at::Maybe{String}
    capabilities::Dict{String,String}
end
routed(c::ModelChoice) = "$(c.provider):$(c.model)"
"""
    ModelSelection(provider, model, connection_id, identity_generation; instance_id="public")

An exact route and model bound to one connection id and generation (AUTH-23). Contains
no credential; serialized ids confer no access.
"""
struct ModelSelection
    provider::String
    model::String
    connection_id::String
    identity_generation::String
    instance_id::String
end
ModelSelection(provider, model, connection_id, identity_generation; instance_id="public") =
    ModelSelection(provider, model, connection_id, identity_generation, instance_id)
routed(s::ModelSelection) = "$(s.provider):$(s.model)"
to_dict(s::ModelSelection) = obj("provider"=>s.provider, "model_id"=>s.model, "connection_id"=>s.connection_id,
    "identity_generation"=>s.identity_generation, "instance_id"=>s.instance_id)
