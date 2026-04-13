# Model → provider resolution.

const PROVIDER_PATTERNS = [
    ("claude", "anthropic"),
    ("gemini", "gemini"),
    ("gpt", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("chatgpt", "openai"),
    ("dall-e", "openai"),
    ("tts", "openai"),
    ("whisper", "openai"),
]

function resolve_provider(model::String)::String
    lower = lowercase(model)
    for (prefix, provider) in PROVIDER_PATTERNS
        startswith(lower, prefix) && return provider
    end
    throw(UnsupportedModelError("unable to resolve provider for model '$model'"))
end
