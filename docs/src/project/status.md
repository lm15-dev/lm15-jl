# Status, support and evidence

## This documentation revision

The manual, new docstrings, build scripts and refactored tutorial sources were
**written without running tests, examples, formatting checks or a site build**.
No live provider requests were made. Do not treat authored expected outputs or a
configured CI workflow as a new verification record.

The default documentation preview workflow is manual-only. Publication requires
separate execution, review and authorization. See [documentation work](documentation.md).

## Earlier implementation evidence

The previous source snapshot at
[`d6758fd`](https://github.com/lm15-dev/lm15-jl/tree/d6758fdffb0256b95735f5ec8f727cb35fac56da)
recorded:

| Check | Earlier result |
|---|---|
| Ordinary package tests | 874 assertions across 34 test sets |
| Real scientific integrations | 48 assertions across 4 test sets |
| Shared contract, all 16 directions | 1,380 pass, zero failures, two existing skips |
| Execution environment | Julia 1.12.5, Linux x86-64 |
| Provider access | Network-isolated fixtures; no real provider calls |

The skips were existing `openai.computer_use` corpus gaps: a missing canonical
request and missing response golden. They were not new exclusions.
[The recorded logs and source digest](https://github.com/lm15-dev/lm15-jl/tree/d6758fdffb0256b95735f5ec8f727cb35fac56da/verification/function-tools)
remain historical evidence, not a receipt for this new documentation revision.

## Platforms

The package targets Julia 1.10 or later on native Linux, macOS and Windows.
Recorded local checks used Julia 1.12.5 on Linux. Other platforms and Julia 1.10
require their own execution; a CI matrix alone does not establish success.

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
