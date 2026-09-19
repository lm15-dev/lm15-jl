@doc """
    complete(client::ProviderLM, request::Request) -> Response
    complete(router::LMRouter, request::Request) -> Response

Submit one request and return a normalized answer. The router resolves the model
and selects credentials; a direct client uses its configured provider and model
name. Provider access can incur charges. Complete responses and media are buffered.
The Codex backend assembles its streamed protocol into the same Response type.

Inspect `finish_reason`, `text(answer)` and `tool_calls(answer)`: success at the
transport layer does not guarantee a textual answer. This function never runs
Julia tools, repeats a failed request, or automatically appends conversation history.

Provider failures use LM15Error families; malformed local arguments can raise
ArgumentError. A connection failure does not prove the provider did no work, so
retry decisions belong to your application.

See also [`stream`](@ref), [`Request`](@ref), [`Response`](@ref), [`retryable`](@ref).
""" complete

@doc """
    ProviderLM(provider; api_key=nothing, base_url=nothing, compat=nothing,
               settings=Dict{String,String}(), access=nothing,
               credentials_path=nothing, account_id=nothing, transport=nothing,
               env=nothing, clock=time, upload_base_url=nothing)

Configure a direct provider client. `api_key` accepts a string, credential value,
or zero-argument callable returning a credential value. Callables are resolved
once for each built request, not when a request object is constructed. Ordinary
API-key clients do not read ambient keys unless `env` is explicitly supplied;
cloud and subscription clients use their declared profile/login policies.

A named compatibility preset chooses that server's address. A custom Gemini
base_url needs an explicit upload_base_url before file uploads are enabled.
Access policies bind provider identity, authentication and compatibility together,
not merely additional headers. Client construction can inspect profile/login files;
credential acquisition can later contact token services or execute declared CLI
mechanisms. A custom transport does not intercept those separate credential chains.

Use a direct provider model name in Request. Use LMRouter for provider:model routing.
""" ProviderLM

for (name, provider, detail) in (
    (:OpenAILM,"openai","OpenAI Responses protocol."),
    (:OpenAIChatLM,"openai-chat","Chat Completions protocol, including named compatible servers through compat."),
    (:AnthropicLM,"anthropic","Anthropic Messages protocol."),
    (:GeminiLM,"gemini","Gemini content protocol. A custom host requires a separate upload_base_url for uploads."),
    (:XaiLM,"xai","xAI's declared API access policy."),
    (:ClaudeCodeLM,"claude-code","Claude Code subscription access. It uses its declared local login store when no explicit credential is supplied."),
    (:OpenAICodexLM,"openai-codex","Codex subscription access. Account identity is required; the backend uses its streamed Responses mapping."),
)
    doc = "    $name(; kwargs...)\n\nConstruct ProviderLM(\"$provider\"; kwargs...). $detail\n\nThese names are constructor functions, not distinct provider types. See [`ProviderLM`](@ref) for keywords, credential selection and side effects."
    @eval @doc $doc $name
end

for (name, signature, explanation) in (
    (:LMRouter, "LMRouter(config=RouterConfig())", "Select providers from explicit model prefixes, a registry, or ordered route rules. Construction validates configuration but does not call a provider. Clients are created and cached on demand. Ambiguity is an error, never a reason to choose an arbitrary account."),
    (:RouterConfig, "RouterConfig(; registry=nothing, rules=DEFAULT_RULES, env=nothing, api_keys=Dict(), base_urls=Dict(), settings=Dict(), transport=nothing)", "Configure model lookup and per-provider credentials, URLs and settings. env=nothing uses the process environment when clients are resolved. Mapping keys name providers; duplicate alias spellings are rejected. Explicit empty credentials do not fall back to another source. Treat mappings as fixed while the router is in use."),
    (:RouteRule, "RouteRule(prefix, provider; note=\"\")", "Route otherwise unresolved model names beginning with prefix to a provider. Rules are ordered; explicit provider prefixes and registry matches have priority."),
    (:Resolution, "Resolution(; requested, model, provider, source, rule=nothing, model_info=nothing)", "Describe a routing decision. model is the unprefixed provider model name; source identifies prefix, catalog or rule. A resolution is not a successful authentication or provider capability probe."),
    (:resolve, "resolve(router, model::AbstractString)", "Return Resolution without creating a client or contacting a provider. Unknown and ambiguous names raise typed configuration errors. Use an explicit provider:model to avoid unintended rule-based routing."),
    (:lm, "lm(router, model::AbstractString)", "Resolve and return a cached ProviderLM, creating it on first use. This can inspect profiles and stored credentials. It does not itself submit a generation request."),
    (:ModelRegistry, "ModelRegistry()\n    ModelRegistry(models)", "Create a lock-protected model catalog from ModelInfo values or canonical dictionaries. This is local metadata; construction does not ask providers to validate entries."),
    (:register!, "register!(registry, model::ModelInfo; replace=true)", "Validate and store a ModelInfo by canonical provider and ID. Duplicate entries are replaced unless replace=false. The current method returns the stored model value; do not rely on chaining a registry result."),
    (:list_models, "list_models(registry::ModelRegistry; provider=nothing)\n    list_models(client::ProviderLM)", "For a registry, return matching local ModelInfo values. For a provider client, contact its models endpoint and parse ModelInfo values. A catalog entry does not prove account permission for every feature."),
    (:estimate, "estimate(pricing::InferencePricing, usage::Usage)", "Sum input, output, cache-read and cache-write counts times their rates per million. Missing rates or counters contribute zero; extra pricing dimensions and provider-specific overlap adjustments are not included. If input counters include cached tokens, independent products can double-count them. Treat this as a simple calculation, not an invoice, accurate subtotal or bound; apply the provider's billing rules explicitly."),
    (:providers, "providers()", "Return the shipped ProviderDefinition values without network access. Their order follows the provider table; do not mutate nested policy data."),
    (:known_providers, "known_providers()", "Return sorted canonical provider identifiers from the shipped table. This is not a live availability check."),
    (:canonical_provider, "canonical_provider(name::AbstractString)", "Normalize underscores to hyphens in a provider identifier. It does not lowercase arbitrary input, verify registration or select a provider."),
    (:ProviderDefinition, "ProviderDefinition(; id, dialect, access, compat=nothing, placeholder_key=nothing, console_url=nothing, note=\"\")", "Describe a shipped provider's protocol, access policy and optional preset. Inspect through providers(); changing this value does not register a new router backend."),
    (:EndpointSupport, "EndpointSupport(; complete=true, stream=true, live=false, files=false, batches=false, images=false, speech=false, video=false, responses_api=false, models=false, caches=false, extra=())", "Declare adapter endpoint support. These flags are configuration metadata, not a live model/account test. Extra supported endpoint names are strings in extra."),
    (:supports_endpoint, "supports_endpoint(support::EndpointSupport, name::Symbol)", "Read a declared endpoint capability, including names in extra; return false for unknown names. Does not probe the network."),
    (:AccessPolicy, "AccessPolicy(; provider, supports=EndpointSupport(), auth_modes=(), enterprise_variants=(), env_keys=(), credential_policy=\"key\", auth_scheme=(\"bearer\",), headers=(), host=nothing, login_hint=nothing, backend=\"api\", backend_options=Dict(), system_prefix=nothing, base_url=nothing)", "Bind credential policy, backend identity, endpoint support and host configuration. Most applications should use shipped providers instead of constructing policies. Overriding access can change identity and routing; it is not merely a way to add headers."),
    (:OpenAIChatCompat, "OpenAIChatCompat(; kwargs...)", "Configure Chat protocol differences: instruction_role, max_tokens_field, stream_usage, tool_result_name, assistant_after_tool_result, thinking_format, thinking_replay, assistant_reasoning_content, strict_tools, builtin_tools, tool_result_media, cache_control, user_field, forced_tool_choice, json_schema, reasoning_efforts, routing, extensions and model_overrides. Fields default to nothing except model_overrides=(); defaults/model rules are resolved later. Prefer preset for a known server; unsupported settings are rejected."),
    (:OpenAIResponsesCompat, "OpenAIResponsesCompat(; kwargs...)", "Configure Responses differences: developer_role, max_output_tokens_field, reasoning_format, tool_result_name, strict_tools, cache_control, commentary_phase, edit_image_field, builtin_tools, tool_result_media, routing and extensions. Unspecified fields default to nothing and are resolved from protocol/model defaults. Prefer a named preset."),
    (:AnthropicCompat, "AnthropicCompat(; kwargs...)", "Configure Messages differences: thinking_format, thinking_replay, cache_control, structured_output, parallel_tool_calls, sampling_params, tool_result_media, reasoning_efforts, model_prefixes and extensions. Fields default to nothing; actual support depends on the selected backend/model."),
    (:preset, "preset(CompatType, name::AbstractString)", "Return a validated named OpenAIChatCompat, OpenAIResponsesCompat or AnthropicCompat from the shipped table. This does not create a client or contact a server. Passing compat=\"name\" to a client also selects that preset's host when applicable."),
    (:describe, "describe(report::AuthReport)\n    describe(resolution::Resolution)", "Return a human-readable authentication explanation or routing description. Built-in credential values are withheld from authentication reports; application-supplied settings and paths still need review before sharing."),
)
    doc = "    " * signature * "\n\n" * explanation
    @eval @doc $doc $name
end
