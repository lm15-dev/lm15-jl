# Request JSON and read it carefully

Structured output controls a provider request. It does not turn every response into
a valid instance of your application's Julia type. These are configuration snippets,
not live calls or schema-validation results.

```julia
using LM15
format = Dict("type" => "json_schema", "name" => "measurement", "strict" => true,
    "schema" => Dict(
        "type" => "object",
        "properties" => Dict("value" => Dict("type" => "number")),
        "required" => ["value"],
        "additionalProperties" => false,
    ))
req = Request("example-model", user("Return a measurement object.");
    config=Config(response_format=format))
```

For a less constrained JSON-object request, use
`Dict("type" => "json_object")`. Providers and models support different subsets;
a valid canonical request can still be refused by the adapter or remote endpoint.

## Parsing is not validation

After a real `complete(client, req)`, inspect the finish reason, then use
`parse_json(answer)` for text-representable JSON. It raises on invalid JSON or
non-text content. It does not run a general JSON Schema validator or guarantee that
all requested fields were returned.

Check the parsed object against the representation you expect and perform
application-domain checks. For arbitrary schemas, choose an explicit
validator in your application. A model's numerical claim is not independently
verified simply because its JSON is well formed.

## Serialize the conversation

`to_json(req)` and `from_json(Request, serialized)` preserve the canonical data
representation. They do not save callable functions, open sockets or a tool's
approval policy. Bind executable Julia tools separately after loading a conversation.

Opaque schemas, tool inputs and continuation dictionaries retain their contents;
empty values inside them are not stripped. Canonical fields have their own omission
rules. Parsed integer values stay exact as BigInt; declared Julia counters and tool
arguments are range-checked when converted.

`Response.provider_data` is excluded by default when serializing a Response. Enable
`include_provider_data=true` only when that information belongs in the stored data.
Nested responses in BatchEntry retain provider data, and other resource types have
their own serialization rules. This flag is not a general redaction policy.
Do not mistake explicit serialization for redacted display: credentials and raw
provider data can expose secrets.

See [conversion rules](conversions.md), [configuration](configuration.md) and
[serialization reference](../reference/content.md).
