# Help for the 2026-09 additions whose definitions carry no docstring of their own.

@doc """
    Adaptation(; field, action, reason, asked=nothing, applied=nothing)

One record of MAP-13: the wire got something other than what was asked, and this says
what. `field` is the config path (`config.seed`), `action` one of `dropped`, `clamped`,
`substituted`, `client_side`, `satisfied`, `defaulted`; `reason` is LM15's own sentence.
Responses and the first stream event carry them; `plan(client, request)` lists them with
no network. A client or router built with `adaptations="refuse"` refuses every deviation
instead; `"silent"` records nothing.
""" Adaptation

@doc """
    DataPart(; value, probabilities=nothing, method=nothing, continuation=())
    data(value)

Structured JSON content. In a user or system message it is input the provider reads as
JSON (or its compact JSON text, on a wire that takes only text). In an assistant message
it is the answer to a judgment request (MAP-14): `value` is the answer object;
`probabilities` a distribution per judgment over its declared keys when one was measured;
`method` how (`"provider_classification"` or `"candidate_sequence_likelihood"`). Only an
assistant's data part may carry probabilities (INV-052). `value` is emitted verbatim,
`nothing` included.
""" DataPart

@doc """
    AuthOperationError <: LM15Error

A managed sign-in operation failed locally (AUTH-24); code `auth_operation`. It is not a
provider 401. Programs match on `reason` (`login_required`, `connection_exists`,
`connection_changed`, `login_denied`, `login_expired`, `indeterminate`,
`storage_unavailable`, `interaction_required`, …); `stage`, `commit_state`
(`not_committed`, `committed`, `unknown`) and `recovery` say where it stopped, whether the
store changed, and what to do. Never retried automatically.
""" AuthOperationError

@doc """
    CollectionLimitError <: LM15Error

A live turn view reached its byte or event budget (`turn(session; max_bytes, max_events)`);
code `collection_limit`, not retryable. `limit` names the budget, `maximum` its value;
`partial_events` are the accepted events (`partial` materializes them as an incomplete
`Turn`), and after a byte overflow `rejected_event` is the event that did not fit. The
session stays open: process the rejected event, then read raw, interrupt or close.
""" CollectionLimitError

@doc """
    MissingCredentialError <: NotConfiguredError

No credential was found for a provider: none was passed, none of its environment keys was
set (a router reads them; a bare client does not), or a failed or signed-out subscription
blocks the ambient key (R3). The message names what to set.
""" MissingCredentialError

@doc """
    Prompt

A question a sign-in asks through an `AuthUI`: `TextPrompt`, `SecretPrompt`,
`SelectPrompt` (answer with an option id) or `ManualCodePrompt` (paste a return URL or
code). Every prompt has a `field_id` and a `label`.
""" Prompt

@doc """
    Notice

Something a sign-in shows through an `AuthUI`: `AuthUrlNotice` (a URL to open),
`DeviceCodeNotice` (a code to enter at a verification URL), `ProgressNotice`, or
`InfoNotice`. Authorization URLs and user codes are session-sensitive: show them to the
person signing in, not to logs.
""" Notice

@doc """
    is_ready(status::ConnectionStatus) -> Bool

True when the saved connection is `ready` or `renewal_due`: a local assessment, not a
remote verification (use `verify` for that).
""" is_ready

@doc """
    routed(selection) -> String

The `provider:model` string of a `ModelSelection` or `ModelChoice`.
""" routed
