# Choose a provider and model

A provider chooses the endpoint and account policy. A model name chooses an offering
within that provider. Compatibility with a protocol does not guarantee every feature.
The snippets here illustrate configuration; they are not live availability tests.

## Direct clients

```julia
using LM15
client = OpenAILM(api_key=ENV["OPENAI_API_KEY"])
req = Request("gpt-4.1-mini", user("Hello"))
```

`OpenAILM` uses Responses. `OpenAIChatLM` uses Chat Completions; `AnthropicLM` uses
Messages; `GeminiLM` uses Gemini's content protocol. These are constructor functions
for `ProviderLM`, not separate subtypes. Direct clients normally take an unprefixed
provider model name. Constructing an ordinary explicit-key client does not validate
the key with a remote service.

## A router

```julia
router = LMRouter(RouterConfig(api_keys=Dict("openai" => ApiKey(ENV["OPENAI_API_KEY"]))))
req = Request("openai:gpt-4.1-mini", user("Hello"))
resolution = resolve(router, req.model)
println(describe(resolution))
```

Resolution checks an explicit prefix, then a registry, then ordered rules. It does
not acquire credentials or submit a generation request. `lm(router, model)` creates
or returns the cached client and can inspect credential/profile files. Prefer
explicit prefixes in production; a family-name rule is a convenience, not proof
of your intended account.

## Local compatible servers

```julia
client = OpenAIChatLM(compat="ollama", api_key="ollama")
req = Request("your-installed-model", user("Hello"))
```

The placeholder above is for that local server, not a real online credential. Named
presets choose their server addresses; they do not silently default to OpenAI.
A custom `base_url` must be a clean HTTP(S) root without embedded credentials,
query parameters or fragments. Do not assume every compatible server accepts tools,
reasoning, JSON schemas or media.

## Catalogs and declared capabilities

`known_providers()` and `providers()` read the shipped table. `list_models(registry)`
is local; `list_models(client)` contacts a provider. Register `ModelInfo` entries
when you want application-owned aliases. Ambiguous aliases raise an error.

```julia
using LM15
registry = ModelRegistry()
register!(registry, ModelInfo(id="example-model", provider="openai",
    api_family="openai_responses", aliases=("writer",)))
router = LMRouter(RouterConfig(registry=registry, env=Dict{String,String}()))
resolve(router, "writer")
```

This example registers local metadata, not an assertion that `example-model` exists.

To inspect a declared endpoint, find the provider definition and call
`supports_endpoint(definition.access.supports, :batches)` or another endpoint symbol.
Treat the result as adapter metadata. Model-specific rules, account permissions,
compatibility overrides and current provider behavior can still restrict it.

## Compatibility overrides

Prefer `preset(OpenAIChatCompat, "ollama")` or the client's named preset. The
compatibility types represent specific protocol differences; they are not a generic
“ignore unsupported settings” switch. Inspect the [reference](../reference/providers.md)
and [status boundaries](../project/status.md) before authoring overrides.

See [credentials](credentials.md) for account selection and [migration](migration.md)
for importing existing Chat-style requests.
