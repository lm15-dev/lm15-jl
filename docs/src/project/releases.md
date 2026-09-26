# Releases and migration notes

## 1.0.0 (2026-09-26)

The first release of lm15 for Julia, at parity with Python, TypeScript, Rust and Go
(see [the changelog](https://github.com/lm15-dev/lm15-jl/blob/main/CHANGELOG.md)).
Coming from the 0.3 development versions:

- When a stored xAI subscription login is unusable (expired, no refresh token) or was
  signed out, a router no longer falls back to the ambient `XAI_API_KEY`; it raises
  `MissingCredentialError`. Pass the key in `api_keys` to use it deliberately (R3).
- A token-shaped string (a JWT, a Google `ya29.` token) on a door that accepts both
  keys and bearer tokens now travels as a bearer token instead of being refused.
- `HTTPTransport()` defaults to the ratified budget: connect 10 s (was 30 s), read
  600 s (was 120 s), 100 connections.
- `choice` and `score` take options in order, as vectors of `key => description` pairs.
- Refusals carry the field they are about in `error.feature`.

## Upgrading an application

The previously recorded implementation includes canonical request/response data,
provider adapters, routing, authentication, streams/live sessions and resources/jobs.
Function tools are written-out `FunctionTool`s (`@tool` and its bindings were removed
before 1.0, 2026-09-23), with optional Tables/Unitful result extensions. See [status](status.md) for recorded evidence and gaps.

When updating an application:

- Keep the provider/account identity explicit. Do not confuse Responses with Chat
  merely because both providers use an OpenAI-style name.
- Retain original assistant messages and continuation state when replaying history.
- Recreate trusted callable bindings separately from deserialized request data.
- Use checked conversions rather than assuming parsed JSON integers are machine Int.
- Own streams/sessions explicitly and do not rely on retries or background polling.
- Review provider mappings and model IDs; library versioning cannot freeze a remote API.

## Documentation versions

Development documentation belongs under `dev`. `stable` should point only to an
actual reviewed release, with its package version and source recorded. A version
field in Project.toml alone is not a published registry release or evidence of a
Git tag. Keep an application's Manifest for reproducibility, and review upgrades.

For Chat-shaped request imports, read [the migration guide](../guides/migration.md).
For publishing the documentation artifact, read [publishing](publishing.md).
