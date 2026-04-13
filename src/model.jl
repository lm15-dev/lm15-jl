# Model — reusable stateful object with conversation memory.

struct HistoryEntry
    request::LMRequest
    response::LMResponse
end

mutable struct Model
    lm::UniversalLM
    model_name::String
    system::Union{String,Nothing}
    tools::Vector{Tool}
    provider::Union{String,Nothing}
    retries::Int
    prompt_caching::Bool
    temperature::Union{Float64,Nothing}
    max_tokens::Union{Int,Nothing}
    max_tool_rounds::Int
    on_tool_call::Union{Function,Nothing}
    conversation::Vector{Message}
    history::Vector{HistoryEntry}
    pending_tool_calls::Vector{Part}
end

function Model(lm::UniversalLM, model_name::String;
               system=nothing, tools=Tool[], provider=nothing, retries=0,
               prompt_caching=false, temperature=nothing, max_tokens=nothing,
               max_tool_rounds=8, on_tool_call=nothing)
    Model(lm, model_name, system, tools, provider, retries, prompt_caching,
          temperature, max_tokens, max_tool_rounds, on_tool_call,
          Message[], HistoryEntry[], Part[])
end

function clear_history!(m::Model)
    empty!(m.history)
    empty!(m.conversation)
    empty!(m.pending_tool_calls)
end

function prepare(m::Model, prompt::String)
    messages = vcat(m.conversation, [UserMessage(prompt)])
    build_config(m) |> cfg -> LMRequest(
        model=m.model_name, messages=messages,
        system=m.system, tools=m.tools, config=cfg)
end

function call(m::Model, prompt::String; kwargs...)
    messages = vcat(m.conversation, [UserMessage(prompt)])

    prefill = get(kwargs, :prefill, nothing)
    prefill !== nothing && push!(messages, AssistantMessage(prefill))

    cfg = build_config(m; kwargs...)
    request = LMRequest(model=m.model_name, messages=messages,
        system=get(kwargs, :system, m.system), tools=m.tools, config=cfg)

    prov = get(kwargs, :provider, m.provider)
    provider_str = prov !== nothing ? prov : ""

    callable_registry = Dict{String,Function}()
    for t in m.tools
        t.type == "function" && t.fn_ !== nothing && (callable_registry[t.name] = t.fn_)
    end

    result = LMResult(
        request=request,
        start_stream=req -> stream_events(m.lm, req, provider=provider_str),
        on_finished=(req, resp) -> begin
            push!(m.history, HistoryEntry(req, resp))
            m.pending_tool_calls = tool_calls(resp)
            m.conversation = vcat(req.messages, [resp.message])
        end,
        callable_registry=callable_registry,
        on_tool_call=get(kwargs, :on_tool_call, m.on_tool_call),
        max_tool_rounds=get(kwargs, :max_tool_rounds, m.max_tool_rounds),
        retries=m.retries,
    )
    result
end

function submit_tools(m::Model, results::Vector{Pair{String,String}}; provider=nothing)
    isempty(m.pending_tool_calls) && throw(ProviderError("no pending tool calls"))

    parts = Part[]
    for tc in m.pending_tool_calls
        id = something(tc.id, "")
        for (rid, val) in results
            if rid == id
                push!(parts, ToolResultPart(id, [TextPart(val)], name=tc.name))
            end
        end
    end

    tool_msg = Message(role="tool", parts=parts)
    msgs = vcat(m.conversation, [tool_msg])
    cfg = build_config(m)
    request = LMRequest(model=m.model_name, messages=msgs, system=m.system, tools=m.tools, config=cfg)

    prov_str = provider !== nothing ? provider : something(m.provider, "")

    callable_registry = Dict{String,Function}()
    for t in m.tools
        t.type == "function" && t.fn_ !== nothing && (callable_registry[t.name] = t.fn_)
    end

    LMResult(
        request=request,
        start_stream=req -> stream_events(m.lm, req, provider=prov_str),
        on_finished=(req, resp) -> begin
            push!(m.history, HistoryEntry(req, resp))
            m.pending_tool_calls = tool_calls(resp)
            m.conversation = vcat(req.messages, [resp.message])
        end,
        callable_registry=callable_registry,
        on_tool_call=m.on_tool_call,
        max_tool_rounds=m.max_tool_rounds,
        retries=m.retries,
    )
end

function build_config(m::Model; kwargs...)
    provider_cfg = Dict{String,Any}()
    m.prompt_caching && (provider_cfg["prompt_caching"] = true)
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

    Config(
        max_tokens=get(kwargs, :max_tokens, m.max_tokens),
        temperature=get(kwargs, :temperature, m.temperature),
        top_p=get(kwargs, :top_p, nothing),
        stop=get(kwargs, :stop, nothing),
        reasoning=reasoning_cfg,
        provider=isempty(provider_cfg) ? nothing : provider_cfg,
    )
end
