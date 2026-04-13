<p align="center">
  <img src="https://raw.githubusercontent.com/lm15-dev/.github/main/assets/banners/banner-1200x300.png" alt="lm15" width="600">
</p>

# LM15.jl

One interface for OpenAI, Anthropic, and Gemini. Zero dependencies.

Julia implementation — conforms to the [lm15 spec](https://github.com/lm15-dev/spec).

```julia
using LM15

result = call("gpt-4.1-mini", "Hello.")
println(text(result))
```

## Install

```julia
using Pkg
Pkg.add(url="https://github.com/lm15-dev/lm15-jl")
```

Set at least one provider key:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GEMINI_API_KEY=...
```

## Usage

### Blocking

```julia
result = call("gpt-4.1-mini", "Hello.")
println(text(result))
println(usage(result))
println(finish_reason(result))
```

### Streaming

```julia
for chunk in stream(call("gpt-4.1-mini", "Write a haiku."))
    chunk.type == "text" && print(chunk.text)
end
```

### Tools (auto-execute)

```julia
weather_tool = FunctionTool("get_weather", "Get weather by city",
    parameters=Dict{String,Any}(
        "type"=>"object",
        "properties"=>Dict{String,Any}("city"=>Dict{String,Any}("type"=>"string")),
        "required"=>["city"]),
    fn_=args -> "22°C in $(args["city"])")

result = call("gpt-4.1-mini", "Weather in Montreal?", tools=[weather_tool])
println(text(result))
```

### Multimodal

```julia
using LM15

result = call("gemini-2.5-flash", "Describe this.",
    messages=[Message(role="user", parts=[
        TextPart("Describe this image."),
        ImageURL("https://example.com/cat.jpg"),
    ])])
```

### Reasoning

```julia
result = call("claude-sonnet-4-5", "Prove √2 is irrational.", reasoning=true)
println(thinking(result))
println(text(result))
```

### Conversation

```julia
conv = Conversation(system="You are helpful.")
user!(conv, "My name is Max.")
# ... pass conv.messages to call()
```

### Cost tracking

```julia
configure!(track_costs=true)

result = call("gpt-4.1-mini", "Explain TCP.")
println(cost(result))

m = model("claude-sonnet-4")
println(text(call(m, "What is TCP?")))
println(text(call(m, "What is UDP?")))
println(total_cost(m))
```

`configure!(track_costs=true)` fetches pricing from models.dev.
You can also call `enable_cost_tracking!()` directly, or use
`estimate_cost(usage, spec)` / `estimate_cost(usage, rates, provider)` manually.

### Reusable model

```julia
gpt = model("gpt-4.1-mini", system="You are terse.")
r1 = call(gpt, "Hello!")
r2 = call(gpt, "What did I say?")  # remembers conversation
```

### Dump curl / HTTP request

```julia
using LM15

println(dump_curl("gpt-4.1-mini", "Hello.", env=".env"))
println(JSON.serialize(dump_http("gpt-4.1-mini", "Hello.", env=".env")))
```

## Dependencies

**Zero.** Uses only Julia stdlib: `Downloads` for HTTP, `Base64` for encoding, and a built-in JSON parser.

## Architecture

```
call() / model()       ← high-level API
        │
        ▼
LMResult (lazy, streamable)
        │
        ▼
LMRequest → UniversalLM → Adapter → Downloads.request
                             │
                    providers/{openai,anthropic,gemini}.jl
```

## Related

- [lm15 spec](https://github.com/lm15-dev/spec)
- [lm15 Python](https://github.com/lm15-dev/lm15-python)
- [lm15 TypeScript](https://github.com/lm15-dev/lm15-ts)
- [lm15 Go](https://github.com/lm15-dev/lm15-go)
- [lm15 Rust](https://github.com/lm15-dev/lm15-rs)

## License

MIT
