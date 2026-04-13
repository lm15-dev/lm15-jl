# High-level API surface.

const _defaults = Dict{String,Any}()
const _client_cache = Dict{String,UniversalLM}()

"""Configure module-level defaults."""
function configure!(; env=nothing, api_key=nothing, track_costs=false)
    empty!(_defaults)
    empty!(_client_cache)
    env !== nothing && (_defaults["env"] = env)
    api_key !== nothing && (_defaults["api_key"] = api_key)

    if track_costs
        enable_cost_tracking!()
    else
        disable_cost_tracking!()
    end
    nothing
end

function _get_client(; api_key=nothing, provider=nothing, env=nothing)
    resolved_key = api_key !== nothing ? api_key : get(_defaults, "api_key", nothing)
    resolved_env = env !== nothing ? env : get(_defaults, "env", nothing)
    cache_key = string(resolved_key, "|", provider, "|", resolved_env)

    haskey(_client_cache, cache_key) && return _client_cache[cache_key]

    client = build_default(api_key=resolved_key, env=resolved_env)
    _client_cache[cache_key] = client
    client
end

"""
    call(model, prompt; kwargs...) -> LMResult

One-shot call to any model.

# Examples
```julia
result = call("gpt-4.1-mini", "Hello.")
println(text(result))

# Streaming
for chunk in stream(call("gpt-4.1-mini", "Write a haiku."))
    chunk.type == "text" && print(chunk.text)
end
```
"""
function call(model_name::String, prompt::String; kwargs...)
    lm = _get_client(
        api_key=get(kwargs, :api_key, nothing),
        provider=get(kwargs, :provider, nothing),
        env=get(kwargs, :env, nothing),
    )

    provider = get(kwargs, :provider, nothing)
    provider_str = provider !== nothing ? provider : ""

    messages = [UserMessage(prompt)]
    prefill = get(kwargs, :prefill, nothing)
    prefill !== nothing && push!(messages, AssistantMessage(prefill))

    provider_cfg = Dict{String,Any}()
    get(kwargs, :prompt_caching, false) && (provider_cfg["prompt_caching"] = true)
    output = get(kwargs, :output, nothing)
    output !== nothing && (provider_cfg["output"] = output)

    reasoning = get(kwargs, :reasoning, nothing)
    reasoning_cfg = if reasoning === true
        Dict{String,Any}("enabled" => true)
    elseif reasoning isa Dict
        merge(Dict{String,Any}("enabled" => true), reasoning)
    else
        nothing
    end

    tools_input = get(kwargs, :tools, Tool[])
    callable_registry = Dict{String,Function}()
    for t in tools_input
        t.type == "function" && t.fn_ !== nothing && (callable_registry[t.name] = t.fn_)
    end

    config = Config(
        max_tokens=get(kwargs, :max_tokens, nothing),
        temperature=get(kwargs, :temperature, nothing),
        top_p=get(kwargs, :top_p, nothing),
        stop=get(kwargs, :stop, nothing),
        reasoning=reasoning_cfg,
        provider=isempty(provider_cfg) ? nothing : provider_cfg,
    )

    request = LMRequest(model=model_name, messages=messages,
        system=get(kwargs, :system, nothing), tools=tools_input, config=config)

    LMResult(
        request=request,
        start_stream=req -> stream_events(lm, req, provider=provider_str),
        callable_registry=callable_registry,
        on_tool_call=get(kwargs, :on_tool_call, nothing),
        max_tool_rounds=get(kwargs, :max_tool_rounds, 8),
        retries=get(kwargs, :retries, 0),
    )
end

"""Create a reusable Model object."""
function model(model_name::String; kwargs...)
    lm = _get_client(
        api_key=get(kwargs, :api_key, nothing),
        env=get(kwargs, :env, nothing),
    )
    Model(lm, model_name;
        system=get(kwargs, :system, nothing),
        tools=get(kwargs, :tools, Tool[]),
        provider=get(kwargs, :provider, nothing),
        retries=get(kwargs, :retries, 0),
        prompt_caching=get(kwargs, :prompt_caching, false),
        temperature=get(kwargs, :temperature, nothing),
        max_tokens=get(kwargs, :max_tokens, nothing),
        max_tool_rounds=get(kwargs, :max_tool_rounds, 8),
        on_tool_call=get(kwargs, :on_tool_call, nothing),
    )
end

"""Build an LMRequest without sending it."""
function prepare(model_name::String, prompt::String; kwargs...)
    LMRequest(model=model_name, messages=[UserMessage(prompt)],
        system=get(kwargs, :system, nothing), tools=get(kwargs, :tools, Tool[]),
        config=Config())
end

"""Upload a file and return a Part."""
function upload(model_name::String, path::String; provider=nothing, api_key=nothing, env=nothing, media_type=nothing)
    prov = provider !== nothing ? provider : try resolve_provider(model_name) catch; "" end
    m = model(model_name, provider=prov, api_key=api_key, env=env)
    LM15.upload(m, path; media_type=media_type)
end

"""Send a pre-built LMRequest."""
function send(request::LMRequest; kwargs...)
    provider = get(kwargs, :provider, nothing)
    provider_str = provider !== nothing ? provider : ""
    lm = _get_client(api_key=get(kwargs, :api_key, nothing), env=get(kwargs, :env, nothing))
    LMResult(
        request=request,
        start_stream=req -> stream_events(lm, req, provider=provider_str),
    )
end
