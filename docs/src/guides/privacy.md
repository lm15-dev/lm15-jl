# Credentials, data and costs

LM15 calls the selected provider directly. It does not add an LM15 relay, but that
is not the same as keeping data on your computer: a provider request sends its
messages and configured content to that provider.

## Decide what may leave the machine

Return small summaries from scientific tools. Select table columns and rows before
encoding them. Avoid sending identifiers, patient records, unpublished results or
other sensitive material unless your account and organization allow it.

A tool runs locally, but the tool result can be sent in the next request. A local
function can also perform network or filesystem work itself. Review the function,
not just its name or input schema. Input validation is not an execution sandbox.

## Keep keys out of examples and logs

Use your environment or a secret manager. Never paste real keys into committed
source, rendered documentation, shared notebooks or issue reports. Do not dump
`ENV`, request headers, raw provider requests or credential serialization to debug
a failed call. Redacted `show` methods do not hide explicit field access.

`explain_auth` reports how credentials would be selected without acquiring a live
token. Review custom settings and paths before sharing even that report.

LM15 filters low-level HTTP diagnostic logs and wraps transport exceptions because
the originals can expose headers or query keys. The trade-off is losing those raw
traces. Keep diagnostics at the application level: operation, selected provider,
normalized error code, and a reviewed request ID—not private prompts or tokens.

## Bound actions, not just text

`max_tokens` is not a total-spend cap. Input tokens, tools hosted by the provider,
media, storage and long-running jobs can have separate prices. `estimate` is a simple sum
of available rates/counters, without every provider's overlap or pricing rules—not an invoice.

An application tool loop should have a call limit, a request limit and an approval
policy. Require explicit approval for operations that change data or spend money.
Never automatically retry a function after its output conversion fails: the
function may already have performed its side effect.

Provider retention rules still apply. A requested `store=false` option is not a
universal promise about logging, training or data retention. Check the provider's
terms and your organization's policy.

## Documentation is not a credential interface

These pages have no credential-entry or live execution widget. Local tutorials use
synthetic data and explicitly labeled fixtures. Live scripts are separate and
require deliberate authorization. See [building documentation](../project/documentation.md).
