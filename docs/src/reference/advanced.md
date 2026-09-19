# Advanced integration interfaces

```@meta
CurrentModule = LM15
```

Most applications should use complete, stream and the resource/job entry points.
These lower-level interfaces support custom transports, imports and conformance
work. Builders can acquire credentials and read paths even though they do not send
the final model request. Explicit fields can contain secrets.

## HTTP boundary

```@docs
WireRequest
HttpResponse
AbstractTransport
HTTPTransport
open_response
build_request
parse_response
normalize_error
build_models_request
parse_models_response
```

## Resource builders and parsers

```@docs
build_file_request
parse_file_response
build_cache_request
parse_cache_response
build_generation_request
parse_generation_response
build_batch_requests
parse_batch_response
parse_batch_entries
build_video_requests
parse_video_response
parse_video_part
```

## Live codecs

```@docs
live_setup_frames
encode_live_event
decode_live_frame
```

## Chat-style migration

```@docs
request_from_openai_chat
response_from_openai_chat
openai_chat_model_string
resolve_openai_chat
complete_from_openai_chat
stream_from_openai_chat
```

## Cloud exchange and signing

```@docs
token_exchange_build
token_exchange_parse
sigv4_sign
```
