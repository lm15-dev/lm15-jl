# LM15.jl

A Julia implementation of the [LM15 contract](https://github.com/lm15-dev/lm15-contract):
build a `Request`, choose a provider, receive a typed `Response` or stream of events.
It calls provider endpoints directly; there is no LM15 relay service.

## Status: contract checks passing on Linux

All 16 directions of the shared contract harness pass at `CONTRACT_PIN`: **1,380
passing cases, zero failures**. Its two skips are existing corpus gaps for
`openai.computer_use` (no canonical request / no response golden), not exclusions
added by this port. The fixtures and comparator are unchanged.

The normal Julia `Pkg.test()` suite also passes: **874 assertions across 34 test
sets**, including function-derived tools, real local HTTP and WebSocket connections,
stream cleanup, private credential writes, refresh-token reuse, account selection
and cross-process locking. Another **48 assertions** exercise real DataFrames,
Unitful and SciML workflows. The exact counts, commands, platform, source digest
and logs are recorded in [`verification/function-tools/`](verification/function-tools/README.md).

Checks ran with networking isolated and synthetic credentials. **No live provider
calls or real login files were used.** This is not a claim that untested cloud
accounts, macOS/Windows or browser WebAssembly work; the boundaries below remain.

Written surfaces:

- Immutable canonical types, validating constructors, JSON readers/writers, all
  36 contract serialization kinds, errors, usage and model metadata.
- OpenAI Responses, Chat Completions and its presets, Anthropic Messages, Gemini;
  composed subscription/cloud access policies; request building and response parsing.
- Bounded, lazy HTTP streams, SSE decoding, start/end coalescing, incremental
  assembly, partial answers on assembly failures, explicit closing.
- Model routing and catalogs, model listing, Chat Completions request/response
  import, and the migration entry points.
- Files, stored caches and reusable prefixes, batch operations and job handles,
  image/speech generation, video operations and job handles.
- OpenAI Realtime and Gemini Live codecs, scoped WebSocket sessions and live turns.
- Explicit/environment/stored credential selection, shared explicit keys, secret-safe
  display, local OAuth refresh under OS locks, atomic private credential writes,
  PKCE and xAI device-code login.
- AWS, Azure and GCP credential-chain mechanisms, SigV4 and native RS256 signing,
  an offline credential explanation, and the JSONL contract entry point.

### Explicit remaining boundaries

These mechanisms fail with a typed configuration error rather than silently choosing
another account. They are also stated gaps in the Python reference:

- Refresh of AWS `login_session` credentials requiring a DPoP proof: renew with
  `aws login`. Fresh cached credentials are read.
- Azure Service Fabric managed identity with TLS thumbprint pinning: supply an
  explicit credential provider.
- GCP external-account credentials backed by an AWS subject-token source, and the
  `external_account_authorized_user` / `gdch_service_account` credential-file types:
  supply an explicit bearer-token provider.

Bedrock Converse/event-stream and Vertex Chat are future dialects, not silently
substituted for the four dialects declared by the current provider table. Unsupported
provider features, such as xAI video listing, remain explicit refusals.

## Platforms

Native macOS, Linux and Windows are the targets. HTTP/WebSocket support uses HTTP.jl;
credential locks use `flock` on Unix and `LockFileEx` on Windows. Windows replacement
uses `MoveFileExW`. **Cross-platform execution has not been verified yet.**

**No browser-WASM build is claimed.** Julia does not currently provide a mature,
supported browser runtime for this package and its native dependencies. A browser
calling a Julia server is a different deployment, not a substitute for that promise.

The project declares Julia 1.10 or later. Verification here used Julia 1.12.5 on
Linux x86-64. CI is configured for Linux/macOS/Windows and Julia 1.10/1.12, but those
other matrix jobs have not been executed from this checkout.

## Install from this checkout

In your application's Julia environment:

```julia
using Pkg
Pkg.develop(path="/path/to/lm15-jl")
using LM15
```

Dependencies are JSON.jl, OrderedCollections.jl, HTTP.jl and OpenSSL_jll, plus Julia
standard libraries. Established JSON, TLS and WebSocket implementations are preferred
over bespoke parsers, RSA arithmetic or networking protocols. The package's own
`LM15.JSON` module is a small boundary around JSON.jl, not a custom parser.

## One request

```julia
using LM15

router = LMRouter() # consults explicit configuration, then the declared auth policy
req = Request("openai:gpt-4.1-mini", user("Hello!");
    config=Config(max_tokens=100))

answer = complete(router, req)
println(text(answer))
println(answer.usage)
```

Explicit credentials and direct providers:

```julia
router = LMRouter(RouterConfig(api_keys=Dict("openai" => ApiKey(my_key))))
lm = AnthropicLM(api_key=my_key)
lm = OpenAIChatLM(compat="ollama", api_key="ollama")

# Callable credentials are resolved once for each built request.
lm = OpenAILM(api_key=() -> BearerToken(current_access_token()))
```

The router resolves environment credentials. Ordinary direct API-key clients take
an explicit key, token or callable; they do not silently select ambient API keys.
Cloud clients use their declared SDK-style chain and configured profile when no
credential is supplied. Subscription clients use their named login store, honoring
custom credential paths. Path/profile configuration is not suppressed along with
ordinary API-key lookup: doing that could select a different account.

A callable runs once per built request and must return a credential value, not
another callable.

A named preset selects its server's address, never an accidental OpenAI fallback.
Binding an `AccessPolicy` also changes the provider identity, credential policy and
preset together: `AnthropicLM(access=policy, api_key=...)` is not just a header override.
Cloud clients accept `settings=Dict("region" => "us-east-1")` and typed credentials
such as `AwsCredentials(access_key_id=..., secret_access_key=...)`.

### Mixed content and copy-and-change

```julia
message = user("Describe this image", image(path="photo.jpg"))
req = Request("openai:gpt-4.1-mini", message)
shorter = Request(req; config=Config(req.config; max_tokens=100))
```

Strings and parts can be mixed in message factories; message collections may be
vectors or tuples. `Request(req; ...)`, `Config(config; ...)` and the other same-type
constructors make validated, shallow copies. Originals are unchanged. Image/audio/
document factories infer known media types from a path's extension without opening
the file. Unknown extensions use the part's default; an explicit `media_type` wins.

Notebook displays show the answer and usage, not raw provider payloads or enormous
base64 strings. Those remain available through their explicit fields.

## Streaming

```julia
answer = stream(router, req) do rs
    for fragment in text_chunks(rs)
        print(fragment)
    end
    response(rs)
end
```

The block's result is returned, and its stream is closed even if the block throws.
For manual ownership, use `ResponseStream(stream(router, req), req)` with `try/finally`
and `close(rs)`. `ResponseStream(events, req) do rs ... end` is also available for
caller-owned event sources.

`events(rs)` exposes typed `StreamEvent`s. `response(rs)` drains what remains.
Closing before the end event cannot turn a partial answer into a completed response.
Always close a stream when leaving early: breaking iteration alone does not release
a blocked network read. The producer uses a bounded channel and starts lazily.

## Tools and JSON

Describe a tool directly from a typed Julia function:

```julia
square_tool = @tool "Square an integer" function square(n::Int)
    return (answer=big(n)^2,)
end
req = Request("openai:gpt-4.1-mini", user("What is 19 squared?"); tools=square_tool)
answer = complete(router, req)

# Only this explicit step runs the function. Ask for approval first if needed.
outputs = [execute_tool(square_tool, call) for call in tool_calls(answer)]
if !isempty(outputs)
    req = Request(req; messages=[req.messages..., answer.message, tool_message(outputs...)])
end

serialized = to_json(req)
restored = from_json(Request, serialized)
```

`@tool` derives the input schema; `execute_tool` checks and converts arguments,
then runs one selected function. It never retries or starts an automatic tool loop.
Numbers and named-tuple outputs become JSON text. Functions never enter serialized
requests: `FunctionTool(square_tool)` shows the ordinary canonical specification.
Handwritten `FunctionTool(name=..., parameters=...)` remains available.

[Function tools](docs/function-tools.md) covers existing methods, defaults, approval,
custom types and precision. [Runnable scientific examples](examples/scientific/README.md)
use real DataFrames, Unitful quantities and a SciML solver. Tables and Unitful
integrations load only when those optional packages are loaded; scientific objects
are never blindly flattened, stripped of units or uploaded in full.

Canonical JSON uses the same snake_case keys as every other port. Empty optional
fields are omitted only by their own typed serializer; opaque schemas, tool input,
continuation state and provider data are not cleaned. Integer/float distinctions
survive JSON parsing; typed counters are range-checked to Julia's `Int`.

## Resources and jobs

```julia
file = file_upload(lm, FileUploadRequest(filename="notes.pdf", path="notes.pdf"))
page = file_list(lm)
file_delete(lm, file.id)

cached = cache(router, Request(req.model, user("Reusable background")))
answer = complete(router, request(cached, "My question"))

job = batch(lm, (req, req))
wait!(job; poll_every=30, timeout=3600)
entries = results(job)

video = video_generate(lm, VideoGenerationRequest(model="sora-2", prompt="Clouds"))
wait!(video; timeout=300)
part = result(video)
```

These calls require the selected provider to support the surface. Stored caches,
generation and batch submissions can incur charges. No read-only property contacts
the provider. `wait!` waits; `result`/`results` do not implicitly wait. A deadline or
interruption does not cancel the provider's work or billing.

## Live sessions

```julia
answer = live(lm, LiveConfig(model="gpt-realtime")) do session
    send_text!(session, "Hello")
    result(turn(session))
end
println(answer.text)
```

The scoped `do` form owns the socket and closes it on exit. Use `recv` for full-duplex
applications. A turn iterator continues through tool calls; materializing `result`
returns at a tool call so the application can answer it. Usage sums preserve unknown
counters rather than treating them as zero.

## Transport configuration

```julia
lm = OpenAILM(api_key=my_key,
    transport=HTTPTransport(connect_timeout=10, read_timeout=300))
```

Both timeouts are positive whole seconds, as required by HTTP.jl. No rounding of
fractional waits, implicit retries or credential-forwarding redirects is performed.
For a custom streaming connection, implement
`LM15.open_response(f, transport::YourTransport, wire::WireRequest)` and call
`f(head::HttpResponse, body::IO)` while the body is open. Your transport owns and
closes the body, including on exceptions. The same method powers complete and
streaming calls. A callable returning a buffered `HttpResponse` is still supported.

HTTP.jl's low-level diagnostic logs are filtered during LM15 network operations:
they can include complete requests, secret headers and token-exchange bodies. The
trade-off is losing those raw transport logs; application logging inside callbacks
and LM15's secret-safe cleanup warnings still work. Network exceptions are raised
outside the underlying catch block so Julia's exception stack cannot reveal secret
headers or query keys as a hidden cause. That also omits the raw HTTP failure trace.

A custom Gemini `base_url` does not imply an upload address. Set `upload_base_url`
explicitly to enable uploads through that gateway; LM15 will neither guess a route
nor quietly upload files to the public Google host.

## Stated Julia choices and costs

- Functions take the client first: `complete(lm, req)`. Mutating operations use
  Julia's `!`: `push!`, `register!`, `refresh!`, `wait!`, `cancel!`, and live sends.
- Immutable structs and tuples give shallow immutability. Opaque dictionaries retain
  their identity and remain caller-owned; do not mutate them concurrently with a call.
- `nothing` means absent; reported zero is distinct. Optional text access returns
  `nothing` when the response is not representable as text alone.
- Tools accept explicit JSON schemas or `@tool`-derived typed interfaces. Ordinary
  multiple dispatch implements tool schema/conversion extensions and transports.
  Provider selection remains data-driven; protocol selection still uses string checks.
- Tool execution is explicit and separate from specification. Unsupported types need
  a deliberate conversion; schemas do not infer domain limits from function bodies.
  Scientific integrations are optional, not mandatory runtime dependencies.
- Abstract error families reproduce catch semantics: `error isa InvalidRequestError`
  includes context-length and unsupported-model failures; `error isa AuthError`
  includes device-login expiry. The family names remain constructible. Concrete
  storage types are internal; `class_name(error)` reports the canonical class name.
- Streams use tasks and bounded channels, not eager whole-response buffering.
  Custom `AbstractTransport`s may stream; legacy callable transports are buffered.
  Stream sources are closed once. Cleanup failures are reported separately and never
  overwrite a completed answer or the original failure.
- Non-streaming responses, multipart uploads and materialized media are buffered in
  memory. Local file paths are read at request-building time, not at construction.
- Job wait deadlines bound the polling loop, not an HTTP call already in flight;
  that call remains bounded by the transport timeout. `WaitTimeout` carries the last
  snapshot. Jobs are not automatically retried, cancelled, or resubmitted.
- Credential locks cooperate with other LM15 processes, not foreign CLI writers;
  the refresh re-read mitigates but cannot eliminate foreign-writer races. OS locks
  on network filesystems and Windows file permissions need platform-specific review.
- Live sessions use scoped ownership (`live(f, lm, config)` / `do`), not an unscoped
  socket-returning constructor. This is intentional Julia resource management.
- Assistant citations that have no native replay representation on an OpenAI wire
  are refused rather than dropped. Incomplete base64 audio packets are refused
  rather than padded with guessed bits. These stricter boundaries need reconciliation
  with any reference behavior that silently discards or repairs the same material.
- Model-family reasoning/cache detection follows copied tables. New model names can
  outgrow them; provider rejection or explicit overrides are preferable to guesses.

## Development and verification

`tools/import_tables.py` copies literal compat/access declarations from the Python
reference using its syntax tree. It does not import Python, invoke credentials, or
run the reference. The resulting JSON files ship inside the Julia package; Python
is not a runtime dependency.

`bin/vet.jl` speaks the shared JSONL protocol through the public build/parse functions:

```sh
julia --project=. bin/vet.jl
```

Run all Julia tests and all shared contract directions:

```sh
julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()'
python3 tools/check_contract.py --direction all
```

The contract launcher uses the sibling harness's own runners, comparator and pin
check. It supplies Julia's launch command without editing the contract's global
shim registry, and requires a Linux network sandbox. Unit tests need only local
loopback servers, not provider credentials. See the verification record for the
isolated-depot workaround used on this machine.

`tools/probe_requests.py` additionally compares 42 combinations outside the corpus
against the Python reference. There are 38 identical results and four documented
strict refusals where Python drops requested content or cache state. Every difference
is retained in the report; the probe exits nonzero to keep those findings visible.
No fixture expectations or comparator rules were weakened.

Source and test code use JuliaFormatter's Blue style, with a development-only
formatter environment (not a runtime dependency):

```sh
julia --project=tools/format -e 'using Pkg; Pkg.instantiate()'
julia --project=tools/format tools/format/format.jl
```

Before a cross-platform release: run the remaining CI matrix, review the recorded
reference differences, and exercise real cloud credentials and provider calls only
where those credentials and any spending are explicitly authorized. The uncommon
unsupported login mechanisms listed above still require explicit credential providers.
