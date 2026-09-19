# Configure generation and inspect usage

Generation settings belong in `Config`; transport and credentials belong on the
client/router. The examples below only create and inspect local data.

```jldoctest
julia> original = Request("example-model", user("Explain Julia briefly."));

julia> shorter = Request(original; config=Config(original.config; max_tokens=80));

julia> original.config.max_tokens === nothing
true

julia> shorter.config.max_tokens
80
```

## Unspecified is different from zero

An optional field containing `nothing` is unspecified. Reported zero, explicit false
and an empty opaque object can have different meanings. Do not replace all missing
values with zero to simplify downstream calculations.

`max_tokens`, `temperature`, `top_p`, `top_k` and `stop` describe generation controls.
The canonical constructor checks ranges, but the selected model may support fewer
controls. `Reasoning(effort="medium")` configures a reasoning request; it does not
promise a fixed reasoning-token count. `ToolChoice` constrains offered tools, and
`CacheConfig` describes caching intent.

See the source-backed [`Config`](@ref) reference for every keyword and default.
Avoid duplicating configuration in provider-specific fields unless you deliberately
need a supported extension.

## Usage is telemetry, not an invoice

```jldoctest
julia> usage = Usage(input_tokens=10, output_tokens=4);

julia> usage.total_tokens
14

julia> Usage(input_tokens=10).output_tokens === nothing
true
```

A provider can report a total independently of individual counters. Live usage sums
preserve unknown components instead of treating them as zero.

`estimate(pricing, usage)` currently sums input/output/cache counts times their
rates independently. Missing rates or counters contribute zero. It does not adjust
for provider-specific overlap: if input counts already include cached tokens, this
can double-count them. Other pricing dimensions may be absent entirely. Treat it as
a simple calculation, not an accurate subtotal, bill, upper bound or spend cap.
Apply the provider's pricing rules, keep account-side budgets and review resource charges.

## Shallow copies

Canonical structs and tuple fields do not deep-freeze nested dictionaries. A new
Request can share its schema, tool input or continuation dictionaries with the old
one. Do not mutate shared dictionaries while a request is being built or sent.

See [JSON output](json.md), [lifecycle](lifecycle.md) and [privacy](privacy.md).
