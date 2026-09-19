# Errors

```@meta
CurrentModule = LM15
```

See [handling failures](../guides/errors.md). Local [`ToolInputError`](@ref) and
[`WaitTimeout`](@ref) are documented with tools and jobs. They are not subclasses
of LM15Error.

```@docs
LM15Error
ProviderError
ConfigurationError
CapabilityError
AuthError
InvalidRequestError
BillingError
RateLimitError
ContextLengthError
TimeoutError
ServerError
UnsupportedModelError
UnsupportedFeatureError
NotConfiguredError
UnknownModelError
AmbiguousModelError
UnknownProviderError
TransportError
LockTimeoutError
StreamAssemblyError
DeviceCodeExpiredError
ErrorDetail
error_code
class_name
retryable
```

## RequestTimeoutError

`RequestTimeoutError` is an alias for [`TimeoutError`](@ref), not a separate failure
category. Qualify it with `LM15` if another imported package uses the same name.
