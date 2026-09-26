# Managed sign-in

```@meta
CurrentModule = LM15
```

The managed-authentication surface (spec AUTH-12 to AUTH-26). See the
[sign-in guide](../guides/sign-in.md) for how the pieces fit.

## The manager and its store

```@docs
Auth
local_auth
memory_auth
Store
FileStore
MemoryStore
default_store_path
Base.close(::Auth)
```

## Discovery

```@docs
login_providers
login_methods
ProviderDescriptor
LoginMethod
MethodField
SelectOption
```

## Connections

`login(auth, provider, method; ui)` is documented with the other [`login`](@ref) methods.

```@docs
configure
set_api_key
connections
status
is_ready
cancel_login
logout
verify
Connection
ConnectionStatus
Verification
ForgetResult
```

## Requests on a saved connection

```@docs
request_auth
credential_provider
RequestAuth
```

## The interaction boundary

```@docs
AuthUI
FunctionUI
TerminalUI
Prompt
TextPrompt
SecretPrompt
SelectPrompt
ManualCodePrompt
Notice
AuthUrlNotice
DeviceCodeNotice
ProgressNotice
InfoNotice
```

## Choosing a model; `connect`

```@docs
model_choices
ModelChoice
ModelSelection
routed
BoundClient
LM15.Interactive
connect
```
