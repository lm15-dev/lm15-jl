# Status, support and evidence

## LM15 1.0.0 (2026-09-26)

| Check | Result |
|---|---|
| Shared contract, every direction, at `CONTRACT_PIN` | 1,788 of 1,788 (the same count as Python, TypeScript, Rust and Go) |
| Managed sign-in runs (`managed` direction) | 43 of 43 |
| One sign-in store shared with Python, TypeScript, Rust and Go, including two processes renewing one login | 40 of 40 mixed-language runs |
| Package tests, including contract consumer vectors and Aqua | pass on Julia 1.12.7 and 1.10.12 (Linux); pass with every dependency at its lowest allowed version on 1.10 |
| Manual | builds in a network-free sandbox; doctests and scientific examples run |
| Live smoke with real keys (`receipts/2026-09-26-live-smoke`) | 63 ok of 69 checks on 14 providers and 4 saved sign-ins; the xAI and Claude sign-ins renewed live |

The six live checks that were not "ok": Z.AI's `json_schema` adapted as receipted,
DeepSeek refusing `json_schema` as receipted, Grok declining to repeat an exact phrase,
and an OpenRouter minted key that had reached its \$1 limit (three checks). None is a
defect in LM15.jl.

Not checked by the maintainers: macOS and Windows (the CI matrix runs them), a
screen-reader or mobile pass over the manual, and every provider's files, batches,
video and realtime endpoints live (the contract grades their wire forms).

## Platforms

The package targets Julia 1.10 or later on native Linux, macOS and Windows.
Local checks used Julia 1.12.7 and 1.10.12 on Linux.

There is no supported browser-WASM build. A browser using a Julia server is a
different architecture, not Julia running directly in the browser. Notebook advice
here is not a tested Pluto or IJulia integration claim.

## Protocol and account boundaries

The implementation declares OpenAI Responses, Chat Completions, Anthropic Messages
and Gemini content protocols, plus their supported access policies. Model names,
account permissions and provider feature availability change independently.
A shipped endpoint-support flag is not proof of live acceptance.

Explicit remaining credential gaps include:

- AWS login-session refresh requiring DPoP: renew with `aws login`; fresh cached
  credentials can be read.
- Azure Service Fabric managed identity requiring TLS thumbprint pinning: supply
  an explicit credential provider.
- GCP external-account credentials using an AWS subject-token source, plus
  external-account-authorized-user and GDCH service-account credential formats:
  supply an explicit bearer-token provider.

Bedrock Converse/event-stream and Vertex Chat are not silently substituted for
other dialects. Unsupported operations, including declared xAI video-listing gaps,
remain explicit refusals.

## Representations can be narrower than the type vocabulary

A canonical request type does not promise that every field has a mapping on every
provider. Examples include video source images, some explicit video durations,
assistant citation replay, reasoning replay and tool-result media. Keep refusals
visible rather than dropping content to make a request appear to succeed.

Earlier outside-corpus probes also recorded stricter Julia refusals where the
Python reference dropped assistant media or cache intent. Those differences remain
findings to review, not permission to alter fixtures or silently weaken behavior.

See [the contract](contract.md), [provider configuration](../guides/providers.md),
and [privacy and costs](../guides/privacy.md).
