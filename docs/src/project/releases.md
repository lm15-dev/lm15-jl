# Releases and migration notes

## Unreleased documentation work

The current work adds the task-oriented manual, source help, reader-facing tutorial
sources, a documentation project and manual preview/publication entry points.
It also extracts the scientific operation definitions so the reader examples and
existing assertions use the same sources.

**This work has not been tested or built in the writing pass.** No release,
registration, deployment or new platform support is implied. Earlier verification
records are retained unchanged for their original source snapshots.

## Existing Julia interface to preserve

The previously recorded implementation includes canonical request/response data,
provider adapters, routing, authentication, streams/live sessions and resources/jobs.
Function tools use a separate ToolBinding, checked explicit execution and optional
Tables/Unitful extensions. See [status](status.md) for recorded evidence and gaps.

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
