abstract type Compat end
Base.@kwdef struct OpenAIChatCompat <: Compat
    instruction_role::Maybe{String} = nothing
    max_tokens_field::Maybe{String} = nothing
    stream_usage::Maybe{String} = nothing
    tool_result_name::Maybe{String} = nothing
    assistant_after_tool_result::Maybe{String} = nothing
    thinking_format::Maybe{String} = nothing
    thinking_replay::Maybe{String} = nothing
    assistant_reasoning_content::Maybe{String} = nothing
    strict_tools::Maybe{String} = nothing
    builtin_tools::Maybe{String} = nothing
    tool_result_media::Maybe{String} = nothing
    cache_control::Maybe{String} = nothing
    user_field::Maybe{String} = nothing
    forced_tool_choice::Maybe{String} = nothing
    reasoning_off::Maybe{String} = nothing
    json_schema::Maybe{String} = nothing
    token_scoring::Maybe{String} = nothing
    reasoning_efforts::Maybe{Tuple} = nothing
    routing::Maybe{JsonObject} = nothing
    extensions::Maybe{JsonObject} = nothing
    model_overrides::Tuple = ()
end
Base.@kwdef struct OpenAIResponsesCompat <: Compat
    developer_role::Maybe{String} = nothing
    max_output_tokens_field::Maybe{String} = nothing
    reasoning_format::Maybe{String} = nothing
    tool_result_name::Maybe{String} = nothing
    strict_tools::Maybe{String} = nothing
    cache_control::Maybe{String} = nothing
    commentary_phase::Maybe{String} = nothing
    edit_image_field::Maybe{String} = nothing
    builtin_tools::Maybe{String} = nothing
    tool_result_media::Maybe{String} = nothing
    routing::Maybe{JsonObject} = nothing
    extensions::Maybe{JsonObject} = nothing
end
Base.@kwdef struct AnthropicCompat <: Compat
    thinking_format::Maybe{String} = nothing
    thinking_replay::Maybe{String} = nothing
    cache_control::Maybe{String} = nothing
    structured_output::Maybe{String} = nothing
    parallel_tool_calls::Maybe{String} = nothing
    sampling_params::Maybe{String} = nothing
    tool_result_media::Maybe{String} = nothing
    reasoning_efforts::Maybe{Tuple} = nothing
    model_prefixes::Maybe{Tuple} = nothing
    extensions::Maybe{JsonObject} = nothing
end
const COMPAT_DATA = JSON.parse(read(joinpath(@__DIR__, "data", "compat.json"), String))
const COMPAT_TABLES = Dict(
    OpenAIChatCompat=>"OPENAI_CHAT",
    OpenAIResponsesCompat=>"OPENAI_RESPONSES",
    AnthropicCompat=>"ANTHROPIC",
)
const PRESET_ALIASES = Dict(
    "openai_chat"=>"openai",
    "chat"=>"openai",
    "chat_completions"=>"openai",
    "responses"=>"openai",
    "openai_responses"=>"openai",
    "lm_studio"=>"lmstudio",
    "dashscope_qwen"=>"qwen",
    "z_ai"=>"zai",
)
function preset_key(name)
    key = replace(lowercase(name), '-'=>'_', ' '=>'_', '.'=>'_')
    return get(PRESET_ALIASES, key, key)
end
function compat_from_dict(::Type{T}, d) where {T<:Compat}
    kw = Dict{Symbol,Any}()
    for (key, v) in d
        name=Symbol(key)
        name in fieldnames(T) || throw(ArgumentError("unknown compatibility setting $key"))
        kw[name] = v isa AbstractVector ? Tuple(v) : v
    end
    return validate(T(; kw...))
end
function preset(::Type{T}, name::AbstractString) where {T<:Compat}
    data = get(COMPAT_DATA[COMPAT_TABLES[T] * "_PRESETS"], preset_key(name), nothing)
    data === nothing && throw(ArgumentError("unknown $T preset"))
    return compat_from_dict(T, data)
end
function preset_url(::Type{T}, name) where {T<:Compat}
    url = get(COMPAT_DATA[COMPAT_TABLES[T] * "_PRESET_BASE_URLS"], preset_key(name), nothing)
    url === nothing &&
        throw(NotConfiguredError("named preset has no default address; pass base_url explicitly"))
    return url
end
const CHAT_DEFAULTS = (
    instruction_role="system",
    max_tokens_field="max_completion_tokens",
    stream_usage="include",
    tool_result_name="omit",
    assistant_after_tool_result="omit",
    thinking_format="reasoning_effort",
    thinking_replay="as_text",
    assistant_reasoning_content="omit",
    strict_tools="omit",
    builtin_tools="reject",
    tool_result_media="reject",
    cache_control="openai",
    user_field="user",
    forced_tool_choice="send",
    reasoning_off="send",
    json_schema="send",
    token_scoring="none",
)
const RESPONSES_DEFAULTS = (
    developer_role="developer",
    max_output_tokens_field="max_output_tokens",
    reasoning_format="responses_reasoning",
    tool_result_name="omit",
    strict_tools="omit",
    cache_control="openai",
    commentary_phase="omit",
    edit_image_field="array",
    builtin_tools="openai",
    tool_result_media="native",
)
const ANTHROPIC_DEFAULTS = (
    thinking_format="anthropic",
    thinking_replay="signed",
    cache_control="anthropic",
    structured_output="send",
    parallel_tool_calls="send",
    sampling_params="send",
    tool_result_media="native",
)
const COMPAT_VALUES=Dict(
    :instruction_role=>("developer", "system"),
    :developer_role=>("developer", "system"),
    :max_tokens_field=>("max_tokens", "max_completion_tokens"),
    :max_output_tokens_field=>("max_tokens", "max_completion_tokens", "max_output_tokens"),
    :stream_usage=>("include", "omit"),
    :tool_result_name=>("include", "omit"),
    :assistant_after_tool_result=>("insert", "omit"),
    :assistant_reasoning_content=>("include_empty", "omit"),
    :strict_tools=>("include", "omit"),
    :tool_result_media=>("native", "images", "reject"),
    :cache_control=>("none", "openai", "openai_implicit", "anthropic"),
    :user_field=>("user", "user_id", "safety_identifier"),
    :forced_tool_choice=>("send", "reject"),
    :reasoning_off=>("send", "lowest"),
    :token_scoring=>("none", "logprob_token_ids"),
    :json_schema=>("send", "reject"),
    :structured_output=>("send", "reject"),
    :parallel_tool_calls=>("send", "reject"),
    :sampling_params=>("send", "reject"),
    :commentary_phase=>("omit", "tag"),
    :edit_image_field=>("array", "indexed"),
    :reasoning_format=>(
        "none",
        "responses_reasoning",
        "reasoning_effort",
        "openrouter",
        "deepseek",
        "qwen",
        "qwen_chat_template",
        "zai",
    ),
)
function validate(c::Compat)
    for name in fieldnames(typeof(c))
        value=getfield(c, name)
        value===nothing && continue
        value isa AbstractDict && check_json(value)
        value=="auto" && continue
        allowed=get(COMPAT_VALUES, name, nothing)
        if name===:thinking_format
            allowed=if c isa AnthropicCompat
                ("anthropic", "deepseek", "adaptive", "effort")
            else
                (
                    "none",
                    "reasoning_effort",
                    "openrouter",
                    "deepseek",
                    "kimi",
                    "qwen",
                    "qwen_chat_template",
                )
            end
        elseif name===:thinking_replay
            allowed=c isa AnthropicCompat ? ("signed", "unsigned") : ("native", "as_text", "omit")
        elseif name===:builtin_tools
            allowed=c isa OpenAIChatCompat ? ("reject", "groq") : ("openai", "verbatim")
        elseif name===:cache_control && c isa AnthropicCompat
            allowed=("none", "anthropic")
        end
        allowed===nothing ||
            value in allowed ||
            throw(ArgumentError("unknown compatibility value for $name"))
        if name===:reasoning_efforts
            all(v->v in VOCABULARIES[:effort] && v!="off", value) ||
                throw(ArgumentError("invalid compatibility reasoning effort"))
        elseif name===:model_prefixes
            !isempty(value) && all(v->v isa AbstractString && !isempty(v), value) ||
                throw(ArgumentError("model_prefixes must be non-empty strings"))
        elseif name===:model_overrides
            for entry in value
                entry isa Union{Tuple,AbstractVector,Pair} && length(entry) == 2 ||
                    throw(ArgumentError("model override must be a prefix and a settings object"))
                prefix, knobs = entry
                prefix isa AbstractString && !isempty(prefix) ||
                    throw(ArgumentError("model override prefix must be non-empty"))
                knobs isa AbstractDict ||
                    throw(ArgumentError("model override settings must be an object"))
                allowed_knobs = (
                    "instruction_role",
                    "max_tokens_field",
                    "stream_usage",
                    "thinking_format",
                    "thinking_replay",
                    "assistant_reasoning_content",
                    "strict_tools",
                    "cache_control",
                    "user_field",
                    "forced_tool_choice",
                    "json_schema",
                    "reasoning_efforts",
                    "tool_result_media",
                    "reasoning_off",
                )
                all(k -> k in allowed_knobs, keys(knobs)) ||
                    throw(ArgumentError("unknown per-model compatibility setting"))
                compat_from_dict(OpenAIChatCompat, knobs)
            end
        end
    end
    return c
end
function resolved_compat(c::Compat, model=""; override=nothing)
    validate(c)
    d = Dict{Symbol,Any}(n=>getfield(c, n) for n in fieldnames(typeof(c)))
    if c isa OpenAIChatCompat
        for (prefix, knobs) in c.model_overrides
            startswith(model, prefix) || continue
            merge!(d, Dict(Symbol(k)=>v for (k, v) in knobs))
            break
        end
    end
    if override !== nothing
        compat_from_dict(typeof(c), override)
        for (key, value) in override
            value === nothing && continue
            name = Symbol(key)
            if name === :extensions && get(d, name, nothing) !== nothing
                d[name] = merge(d[name], value)
            else
                d[name] = value
            end
        end
    end
    # Recheck the effective policy, not just its parent: a per-model or
    # per-request override must not bypass the policy's closed vocabularies.
    effective = compat_from_dict(typeof(c), Dict(string(k) => v for (k, v) in d))
    d = Dict{Symbol,Any}(name => getfield(effective, name) for name in fieldnames(typeof(c)))
    defaults = if c isa OpenAIChatCompat
        CHAT_DEFAULTS
    elseif c isa OpenAIResponsesCompat
        RESPONSES_DEFAULTS
    else
        ANTHROPIC_DEFAULTS
    end
    for (key, value) in pairs(defaults)
        get(d, key, nothing) in (nothing, "auto") && (d[key]=value)
    end
    return (; (k=>v for (k, v) in d)...)
end
