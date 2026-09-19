# Select credentials deliberately

A configuration error should not quietly send a request through another account.
LM15 distinguishes explicit keys, stored subscription logins and cloud credentials.
The snippets below illustrate setup; they are not executed by the docs build.

## Explicit keys and tokens

```julia
using LM15
client = OpenAILM(api_key=ENV["OPENAI_API_KEY"])
```

Ordinary direct clients require an explicit value or callable unless an `env` mapping
is deliberately supplied. A router consults its configured environment by default.
An explicit empty credential is an error, not permission to fall back.

`ApiKey` and `BearerToken` represent different authentication schemes. Do not label
a short-lived access token as an API key merely because both contain strings.
A credential callback must take no arguments and return a string or credential value,
not another callback. It runs once per built request and can perform I/O.

## Subscription accounts

`ClaudeCodeLM` and `OpenAICodexLM` use their declared login stores when appropriate.
Codex also needs account identity. Supply `credentials_path`/`account_id` deliberately
when using nonstandard stores. These path/profile choices remain meaningful even
when ordinary ambient API-key lookup is not enabled.

Stored-token refresh may read and write a credential file and contact a token
endpoint. LM15 uses private atomic writes and cooperating file locks. That does
not make foreign CLI writers obey the same lock. See the remaining mechanisms in
[status](../project/status.md).

## Cloud accounts

Cloud providers use their declared AWS, Azure or GCP chain and host settings. For
example, a region belongs in `settings`, not in a guessed generic base URL. Profiles,
credential files, declared CLI mechanisms and metadata services can participate.
An explicit `AwsCredentials` or bearer-token provider can replace unsupported parts
of a chain; do not silently select a different identity.

Use `explain_cloud` or `explain_auth` before acquisition. `ChainContext` and its
HTTP/command/file hooks are advanced integration surfaces, not a general sandbox.

## Diagnose selection without printing secrets

```julia
using LM15
report = explain_auth("openai"; env=Dict{String,String}(), api_keys=Dict())
println(describe(report))
```

This synthetic environment does not inspect a real key. With real configuration,
a report may inspect local files. `selected` means a source was selected; `unprobed`
means acquisition was intentionally not performed. `configured=true` does not prove
a live login succeeds. Custom settings and paths can still be sensitive.

Never print `access_token`, credential fields, serialized credentials, raw HTTP
headers or the whole process environment in a bug report.

## Interactive login

`login("xai")` performs an explicit device authorization flow, waits for approval,
and writes a credential store. It is not a harmless lookup. Other providers return
instructions for their provider-owned login/key workflow rather than inventing one.
Lower-level `start_xai_device_login` and `poll_xai_device_login` separate acquisition
from persistence; `write_xai_credential` is the explicit write.

See [authentication reference](../reference/auth.md) and [privacy](privacy.md).
