# Requests, configuration and content

```@meta
CurrentModule = LM15
```

## Core values

```@docs
Request
Response
Message
Config
Usage
Reasoning
CacheConfig
```

## Parts and replay state

Canonical parts have a fixed wire vocabulary. Represent application types through
explicit tool codecs rather than inventing new protocol variants. Preserve opaque
continuation state when replaying a returned assistant message.

```@docs
Part
TextPart
ThinkingPart
RefusalPart
CitationPart
ImagePart
AudioPart
VideoPart
DocumentPart
BinaryPart
ContinuationState
TokenLogprob
TopLogprob
```

## Content construction and access

```@docs
user
assistant
developer
text
thinking
refusal
citation
image
audio
video
document
binary
bytes
parts_of
citations
continuation_data
kind
```

## Serialization

Explicit serialization can expose private fields even when display is redacted.
`parse_json` parses text; it does not perform arbitrary schema validation.

```@docs
validate
to_dict
from_dict
to_json
from_json
parse_json
```

## Judgments and data parts

A judgment is a question whose answer is one of the keys you declare; the answer is a
[`DataPart`](@ref), with a measured distribution where the provider can measure one
(MAP-14). See [judgments](../guides/judgments.md).

```@docs
DataPart
data
data_part
probabilities
Judgment
judgments
choice
yes_no
score
judgments_in_schema
request_judgments
expected_level
```

## Adaptations

```@docs
Adaptation
plan
apply_client_side_stop
```
