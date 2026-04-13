"""
    LM15

One interface for OpenAI, Anthropic, and Gemini. Zero dependencies.

Julia implementation — conforms to the [lm15 spec](https://github.com/lm15-dev/spec).

# Quick start
```julia
using LM15

result = call("gpt-4.1-mini", "Hello.")
println(text(result))
```
"""
module LM15

using Downloads: Downloads
using Base64: base64decode, base64encode

include("json.jl")
include("types.jl")
include("errors.jl")
include("transport.jl")
include("capabilities.jl")
include("providers/common.jl")
include("providers/openai.jl")
include("providers/anthropic.jl")
include("providers/gemini.jl")
include("client.jl")
include("result.jl")
include("model.jl")
include("conversation.jl")
include("middleware.jl")
include("cost.jl")
include("factory.jl")
include("api.jl")

# ── Exports ─────────────────────────────────────────────────────────

export
    # Types
    Part, TextPart, ThinkingPart, RefusalPart, CitationPart,
    ImageURL, ImageBase64, AudioBase64, DocumentURL, ToolCallPart, ToolResultPart,
    DataSource, decode_bytes, Message, UserMessage, AssistantMessage, ToolResultMessage,
    Tool, FunctionTool, BuiltinTool, ToolCallInfo,
    Config, LMRequest, LMResponse, Usage, StreamEvent, StreamChunk,
    ErrorInfo, PartDelta,
    EmbeddingRequest, EmbeddingResponse,
    FileUploadRequest, FileUploadResponse,
    # Errors
    LM15Error, TransportError, ProviderError, AuthError, BillingError,
    RateLimitError, InvalidRequestError, ContextLengthError,
    TimeoutError, ServerError, UnsupportedModelError,
    # Client
    UniversalLM, register!,
    # Result
    LMResult,
    # Model
    Model, HistoryEntry,
    # Conversation
    Conversation, user!, assistant!, prefill!, clear!,
    # Middleware
    with_retries, with_cache, with_history,
    # Cost
    CostBreakdown, estimate_cost,
    # Factory
    build_default, providers,
    # Capabilities
    resolve_provider,
    # API
    call, model, prepare, send, configure!,
    # Accessors
    text, thinking, tool_calls, finish_reason, usage, image, audio, response

end # module
