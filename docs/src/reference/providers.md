# Providers and routing

```@meta
CurrentModule = LM15
```

Use a direct client for one explicit provider or LMRouter for model-name routing.
A configured endpoint is not a successful live capability test. See
[provider selection](../guides/providers.md).

## Requests and clients

```@docs
complete
ProviderLM
OpenAILM
OpenAIChatLM
AnthropicLM
GeminiLM
XaiLM
ClaudeCodeLM
OpenAICodexLM
LMRouter
RouterConfig
RouteRule
Resolution
resolve
lm
```

## Models and metadata

```@docs
ModelRegistry
register!
list_models
ModelInfo
ModelOrigin
InferenceModelInfo
InferencePricing
estimate
```

## Access and compatibility

```@docs
ProviderDefinition
AccessPolicy
EndpointSupport
supports_endpoint
providers
known_providers
canonical_provider
OpenAIChatCompat
OpenAIResponsesCompat
AnthropicCompat
preset
```

## Human-readable descriptions

```@docs
describe
```

## TypeSafe, the connection budget, managed routers

```@docs
TypeSafeLM
Timeouts
with_auth
```
