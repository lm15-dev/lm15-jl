# Authentication

```@meta
CurrentModule = LM15
```

Credential values can be sensitive even when their default display is redacted.
Read [credential selection](../guides/credentials.md) and [privacy](../guides/privacy.md)
before inspecting fields or sharing diagnostics.

## Values and resolution

```@docs
CredentialValue
Credential
ApiKey
BearerToken
AwsCredentials
LocalOAuthCredential
resolve_credential
access_token
account_id
has_refresh_token
is_expired
```

### StaticCredential

`StaticCredential` is an alias for [`ApiKey`](@ref), not a second credential type.
Use `ApiKey` in new examples. Its serialization contains the key.

## Diagnostics and stored readers

```@docs
AuthStep
AuthReport
selected_step
explain_auth
default_claude_credentials_path
default_codex_auth_path
read_claude_code_credential
read_codex_cli_credential
read_xai_credential
```

## Cloud context

```@docs
ChainContext
cloud_credential_provider
explain_cloud
```

## Login and persistence

These functions can perform network requests or write local credentials. They are
not executed by the documentation examples.

```@docs
PKCEPair
generate_pkce
pkce_challenge
DeviceAuthorization
start_xai_device_login
poll_xai_device_login
CredentialFileStore
mutate!
delete_credential!
write_xai_credential
login
```

## Where a credential came from

```@docs
CredentialSource
credential_origin
looks_like_access_token
```
