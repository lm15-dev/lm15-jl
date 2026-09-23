# Send images, audio and documents

Messages can mix text and media parts. Choose a single source for each media part:
inline bytes/base64, a URL, a provider file ID, or a local path.
These snippets construct data; a later provider call sends or uploads content.

```julia
using LM15
message = user("Describe this image", image(path="photo.jpg"))
req = Request("example-model", message)
```

A path is not opened at construction. Known filename extensions suggest a MIME
type, but LM15 does not inspect the file's bytes or convert its format. An explicit
`media_type` wins. A wrong MIME declaration can pass construction and still be
rejected by the provider. Unknown extensions use the part's default type.

```julia
message = user("Summarize this document",
    document(path="report.pdf", media_type="application/pdf"))
```

## Inline data and references

Use `image(data=png_bytes)` for an encoded image byte vector, not a Julia pixel
matrix. Use `audio(data=encoded_audio, media_type="audio/wav")` for actual audio
file bytes. No recording, OCR, image conversion or resampling happens automatically.

A `url` or `file_id` is a reference. `bytes(part)` can decode inline data or read a
local path, but deliberately refuses to download arbitrary URLs or resolve file IDs.
Use a supported provider download operation or your application's explicit fetch.

For plotting packages, save a small image in a chosen format before constructing an
ImagePart. A numerical array instead needs a deliberate representation such as
`array_content`; its values are not automatically an image.

## Read the response without losing content

`text(answer)` returns `nothing` when the answer is not text-representable. Inspect
`answer.message.parts`, `parts_of(ImagePart, answer)`, `tool_calls(answer)` or
`citations(answer)` as appropriate. A refusal is not an ordinary answer.

Replay the original assistant message when continuing. Copying only its visible
text can lose citations, media or continuation state. A provider may refuse content
that has no supported replay representation; LM15 does not silently discard it.

## Tool output

`tool_result` accepts text and presentational parts, not nested tool calls or
reasoning parts; `tool_content` turns a Julia value into those parts.
Check the selected provider's media policy before returning a large image or audio
payload to it. Encoding media locally is not permission to send it.

See [files](files-caches.md), [conversion rules](conversions.md), and
[the content reference](../reference/content.md).
