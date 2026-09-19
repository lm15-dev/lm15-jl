# Files and reusable context

These examples describe remote operations. They require a configured provider with
the relevant support and may create billable storage. They are not run by the docs.

## Upload, inspect, use and delete

```julia
using LM15
client = OpenAILM(api_key=ENV["OPENAI_API_KEY"])
file = file_upload(client, FileUploadRequest(filename="notes.pdf", path="notes.pdf",
    media_type="application/pdf"))
ready = file_wait_ready(client, file.id; timeout=120)
ready.readiness == "ready" || error("The provider could not process this file")
req = Request(get(ENV, "LM15_MODEL", "gpt-4.1-mini"),
    user("Summarize the notes", document(file_id=file.id)))
# complete(client, req) is a separate, potentially paid model call.
```

The upload reads the local path and buffers multipart content. A ready file ID is
provider-specific, not a universally portable URL. Keep it with its provider/account
context. Processing can end in failed status; terminal does not always mean usable.

`file_list` returns a page with `items` and `next_cursor`. Follow the cursor explicitly.
`file_get` fetches metadata. `file_download` materializes content where supported.
`file_delete` changes remote state; delete only after deciding the application no
longer needs the resource. Cleanup is an application policy, not a surprise in a
read-only accessor.

For Gemini through a custom gateway, configure `upload_base_url` explicitly. LM15
will not quietly upload to the public Google host merely because a custom content
host was supplied.

## Implicit prefixes versus stored caches

`CachedPrefix` represents reusable context. An implicit prefix is sent again with
cache intent. A stored cache refers to a resource already held by the provider.
Their cost and lifecycle differ.

A stored-cache example for a compatible Gemini account:

```julia
using LM15
client = GeminiLM(api_key=ENV["GEMINI_API_KEY"])
model = ENV["LM15_MODEL"]
prefix = Request(model, user("Background context approved for reuse."))
cached = cache(client, prefix; ttl_seconds=3600, label="example-context")
next_request = request(cached, "What are the main points?")
# complete(client, next_request) submits a separate generation request.
```

The prefix must not carry generation settings. Put those on the follow-up request.
Calling `cache` can create a remote resource on stored-cache paths; do not classify
it as universally local. A supplied TTL must be positive and provider limits apply.

`cache_get`, `cache_list`, `cache_update` and `cache_delete` explicitly manage stored
resources. Expiry or deletion can invalidate an existing handle. No background renewal
or silent resource recreation is performed.

See [resource reference](../reference/resources.md) and [costs](privacy.md).
