@doc """
    LM15Error <: Exception

Abstract root for normalized LM15 provider, transport, configuration and capability
failures. Inspect code, message, provider, provider_code, status, request_id,
retry_after, credential_hint and model when present. Stream assembly errors may
carry partial content and cleanup_errors. Do not assume every Julia exception is
an LM15Error: ArgumentError, WaitTimeout and UnknownProviderError
are separate, as are errors thrown by your own functions.

Error messages can contain provider-supplied text. Redacted client display does
not make every exception safe to post publicly. retryable is a classification,
not permission to repeat a side effect or a promise of automatic retry.
""" LM15Error

for (name, detail) in (
    (:ProviderError,"Abstract family of provider failures, also constructible as ProviderError(message; metadata...). Catch with isa ProviderError to include authentication, billing, rate limit, request, timeout and server failures."),
    (:ConfigurationError,"Abstract family of local provider/model configuration failures. ConfigurationError(message; metadata...) constructs NotConfiguredError. These are not fixed by blindly retrying a request."),
    (:CapabilityError,"Abstract family for behavior the selected adapter cannot represent. CapabilityError(message; metadata...) constructs UnsupportedFeatureError."),
    (:AuthError,"Authentication/authorization family, including DeviceCodeExpiredError. AuthError(message; metadata...) is constructible. Check the selected identity and credential_hint; do not silently switch accounts."),
    (:InvalidRequestError,"Invalid provider request family, including ContextLengthError and UnsupportedModelError. InvalidRequestError(message; metadata...) is constructible. Fix the request instead of repeatedly resubmitting it."),
    (:BillingError,"The provider reported a billing or credit failure. Repair account billing before trying again."),
    (:RateLimitError,"The provider reported a rate limit. retry_after may offer a delay, but no retry is performed automatically."),
    (:ContextLengthError,"The request exceeds a context limit. This is an InvalidRequestError. Shorten or explicitly summarize history rather than dropping it silently."),
    (:TimeoutError,"Normalized provider/request timeout. A timeout does not prove the provider did not execute the request. RequestTimeoutError is an alias for this same type."),
    (:ServerError,"Provider server failure. Classified retryable, but retry safety still depends on the operation."),
    (:UnsupportedModelError,"The provider rejected the model. This is an InvalidRequestError; verify the provider-specific model ID and account access."),
    (:UnsupportedFeatureError,"The selected adapter cannot represent the requested feature without losing intent. Change provider/configuration or handle the refusal; no silent substitution is made."),
    (:NotConfiguredError,"Required provider credentials or settings are absent, malformed or ambiguous. credential_hint can explain the next step. Empty explicit credentials do not authorize a fallback account."),
    (:UnknownModelError,"The model name cannot be resolved from a known prefix, catalog or rule. Prefer an explicit provider:model identifier."),
    (:AmbiguousModelError,"Multiple catalog/provider matches exist. Select an explicit provider and canonical model ID."),
    (:TransportError,"The provider connection could not be opened, read or written. Low-level failures are redacted to avoid exposing credential-bearing requests; that sacrifices the raw HTTP traceback."),
    (:LockTimeoutError,"The local credential-store lock was not acquired before its deadline. Another process may be refreshing the same store; no credential is overwritten to bypass the lock."),
    (:StreamAssemblyError,"A streamed answer could not be assembled or ended without completion. partial contains the available Response when recoverable; it is not a completed successful answer. Cleanup errors do not overwrite the primary failure."),
)
    doc = "    $name(message=\"\"; metadata...)\n\n" * detail * "\n\nSee also [`LM15Error`](@ref), [`retryable`](@ref)."
    @eval @doc $doc $name
end

@doc """
    UnknownProviderError(provider)

Report an unknown provider identifier. This exception is separate from LM15Error;
inspect known_providers for the shipped choices. No network probe is performed.
""" UnknownProviderError

@doc """
    DeviceCodeExpiredError(provider)

Report that an OAuth device authorization expired. This is an AuthError with code
`auth`. Begin a fresh, explicitly authorized device flow; do not keep polling the
expired code.
""" DeviceCodeExpiredError

@doc """
    error_code(error::LM15Error)

Return the canonical error-code string. This does not classify arbitrary exceptions;
WaitTimeout has its own local meaning.
""" error_code

@doc """
    class_name(error::LM15Error)

Return the canonical public error-family name, hiding internal concrete storage
names used to implement Julia's exception hierarchy.
""" class_name

@doc """
    retryable(error::Exception) -> Bool

Return true for TransportError, LockTimeoutError, RateLimitError, TimeoutError and
ServerError; false for other exceptions. This is only a category hint. LM15 does
not retry automatically, and operations with side effects may be unsafe to repeat
when their first outcome is unknown.
""" retryable
