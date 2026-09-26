# lm15 for Julia

One request and response model for every major AI model provider, in Julia.
Write a `Request` once and send it to OpenAI, Anthropic, Gemini, xAI, Groq,
DeepSeek, OpenRouter, Z.AI, Moonshot, Meta, DeepInfra, Together, Fireworks,
Parasail, a cloud (Azure, Bedrock, Vertex) or a model on your own machine:
change the model string, keep the program.

```julia
using LM15
router = LMRouter()
answer = complete(router, Request("anthropic:claude-haiku-4-5", user("What eats acorns at night?")))
println(text(answer))
```

- **Low-level on purpose.** Typed requests, responses, stream events, tools,
  media, errors and exact JSON serialization. No hidden tool loop, no retries
  you did not ask for, no prompt templates: what you build on top decides those.
- **The same behavior in every language.** lm15 exists for Python, TypeScript,
  Rust, Go and Julia, graded by one shared
  [contract](https://github.com/lm15-dev/lm15-contract): the same request
  produces the same wire bytes and the same response in all five.
- **Scientific Julia welcome.** Optional integrations for Tables.jl and
  Unitful.jl load only when those packages are loaded.

Documentation: **[lm15.dev](https://lm15.dev/docs/)** (cross-language guides) ·
the Julia manual in [docs/](docs/src/index.md) · `?complete` in the REPL.

## Install

Requires Julia 1.10 or newer.

```julia
using Pkg
Pkg.add(url="https://github.com/lm15-dev/lm15-jl")
```

**1.0.0 is the first release of lm15 for Julia.** It is not yet in Julia's
General registry (see [RELEASING.md](RELEASING.md)). Python's lm15 is stable;
TypeScript, Rust and Go are release candidates.

Set the key of the provider you call (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`GEMINI_API_KEY`, …). The router reads it from the environment. A client made
directly (`OpenAILM(api_key=...)`) never reads the environment.

## Guide

### Ask, stream, continue

```julia
router = LMRouter()
req = Request("gpt-4.1-mini", user("What eats acorns at night?");
    system="Answer in one sentence.", config=Config(; max_tokens=200))

answer = complete(router, req)
println(text(answer), " ", answer.finish_reason, " ", answer.usage.output_tokens)

# Streaming: text as it arrives, then the same Response complete returns.
final = stream(router, req) do rs
    foreach(print, text_chunks(rs))
    response(rs)
end

# A conversation is the messages so far, plus the reply.
req = Request(req; messages=(req.messages..., final.message, user("And by day?")))
```

A model string is either `provider:model` (`"gemini:gemini-2.5-flash"`,
`"groq:openai/gpt-oss-20b"`, `"ollama:qwen3.5:0.8b"`) or a bare name the router
recognizes (`"gpt-4.1-mini"`, `"claude-haiku-4-5"`).

### Tools

lm15 returns the model's tool calls; your program runs them and answers.

```julia
weather = FunctionTool(; name="get_weather", description="Current weather for a city.",
    parameters=Dict("type" => "object", "properties" => Dict("city" => Dict("type" => "string")),
        "required" => ["city"]))
req = Request("claude-haiku-4-5", user("Weather in Montreal?"); tools=(weather,))
answer = complete(router, req)
results = [tool_result(call, look_up_weather(call.input["city"])) for call in tool_calls(answer)]
req = Request(req; messages=(req.messages..., answer.message, tool_message(results...)))
answer = complete(router, req)   # the model answers with the results
```

No tool is derived from a Julia function: you write its schema, and you decide
whether a call runs.

### Images, documents, audio

```julia
photo = image(path="camera-trap.jpg")   # or url=, data=, file_id=
req = Request("gemini-2.5-flash", user("What animal is this?", photo))
```

### Judgments with probabilities

```julia
format = judgments("ageing" => yes_no("Will it improve with age?"),
                   "style" => choice("Dominant style?", ["fruit", "oak", "mineral"]))
answer = complete(router, Request("typesafe:jev-latest", user("Blackcurrant, cedar, firm tannins.");
    config=Config(; response_format=format, probabilities="required")))
data(answer), probabilities(answer)
```

A distribution comes from TypeSafe or from a vLLM server that scores tokens;
elsewhere you get the pick only (and `probabilities="required"` refuses). See
[judgments](docs/src/guides/judgments.md).

### Errors

Every failure is an `LM15Error` subtype you can dispatch on (`RateLimitError`,
`AuthError`, `ContextLengthError`, `UnsupportedFeatureError`, …) with the
provider's own code and message, the request id, and rate-limit evidence when
the provider sent it.

```julia
try
    complete(router, req)
catch e
    e isa RateLimitError && e.retry_after !== nothing && sleep(e.retry_after)
end
```

When a wire cannot take a setting as asked (a `seed` on Anthropic, `top_k` on
OpenAI), lm15 adapts it and records what it did in `answer.adaptations`;
`plan(router, req)` lists those records with no network call.

### Sign in once, use everywhere

Besides API keys, lm15 can use an account you sign in to (an xAI, Claude,
ChatGPT, GitHub Copilot, Kimi Code or Meta subscription, or an OpenRouter
login), saved in one file every lm15 language shares. Sign-in is
**provisional**: whether each provider permits it, and how it is billed, is the
provider's call.

```julia
using LM15, LM15.Interactive
connect() do lm                       # pick a connection and a model in the terminal
    println(text(complete(lm, "Explain drought stress in oaks.")))
end
```

On a server, attach the saved connections to a router instead:
`LMRouter(RouterConfig(; auth=local_auth()))`. See
[sign-in](docs/src/guides/sign-in.md).

### Azure, AWS and Google Cloud

The cloud doors (`azure:`, `bedrock-anthropic:`, `vertex:` …) find the identity
your machine already has, the way each cloud's own SDK does, and say which one:

```julia
explain_auth("vertex")        # which identity and project, without a network call
deployed = LMRouter(RouterConfig(; credentials=Dict("vertex" => "platform")))  # that identity or fail
keyed = LMRouter(RouterConfig(; api_keys=Dict("vertex" => ENV["MY_VERTEX_KEY"]),
    settings=Dict("vertex" => Dict("location" => "europe-west4"))))
```

### More

Reasoning controls, prompt caching, built-in provider tools, files and batches,
image and speech generation, video, realtime sessions, the model catalog, and
reading an OpenAI Chat Completions request into a `Request` are in the
[manual](docs/src/index.md) and the [cross-language guides](https://lm15.dev/docs/).

## Stability

The chat core is stable within 1.x: requests, responses, streaming, tools,
structured output and judgments, errors, credentials and model listing. These
ship as **provisional** and may still change in 1.x with a notice in the
contract: files, batches, media generation, stored caches, realtime sessions,
Chat Completions ingest and sign-in.

## Conformance

This package is graded by [lm15-contract](https://github.com/lm15-dev/lm15-contract)
at the commit in `CONTRACT_PIN`: **1,788 of 1,788 checks pass** (2026-09-26), the
same count as Python, TypeScript, Rust and Go. The checks compare the exact
requests lm15 builds and the responses it reads against recorded provider traffic.
The sign-in store is shared: runs that alternate Julia with Python, TypeScript,
Rust and Go on one file, and two processes of different languages renewing the
same login at once, pass.

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
cd ../lm15-contract && python3 harness/check.py --shim julia --direction all
```

Real calls: `julia --project=. examples/live_smoke.jl OUT` sends a small set of
checks (chat, streaming, structured output, a tool loop, model listing) to every
provider you have a key for and writes receipts. The latest run is in
[receipts/](receipts/2026-09-26-live-smoke/SUMMARY.md).

## Stated deviations

Where Julia differs from the family surface (contract `playbooks/api-family.md`):

- **Functions, not methods.** `complete(router, req)`, `text(answer)`,
  `tool_calls(answer)`, `status(auth, "xai")`: Julia dispatches on the first
  argument where the other languages write `router.complete(req)`.
- **`connect` lives in `LM15.Interactive`**, which you import where a person is
  present (the contract requires a separately imported interactive family;
  it also avoids clashing with `Sockets.connect`).
- **Ordered choices are vectors of pairs.** A Julia `Dict` has no order, so
  `choice` and `score` take `["key" => "description", …]`.
- **Connection budget.** `Timeouts(; connect=10, read=600, write=600, pool=600)`
  and `max_connections=100` as ratified; HTTP.jl counts whole seconds (fractions
  round up) and has no timeout on waiting for a free connection, so `pool` is
  recorded but not enforced.
- **No browser build.** Julia has no browser runtime lm15 targets.

## Development

```bash
julia --project=. -e 'using Pkg; Pkg.test()'     # unit tests, contract vectors, Aqua
bash docs/build.sh                                # the manual, built without network
```

## License

MIT. See [LICENSE](LICENSE).
