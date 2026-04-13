# Result — lazy stream-backed response with auto tool execution.

mutable struct LMResult
    request::LMRequest
    start_stream::Function  # (LMRequest) -> Vector{StreamEvent}
    on_finished::Union{Function,Nothing}
    callable_registry::Dict{String,Function}
    on_tool_call::Union{Function,Nothing}
    max_tool_rounds::Int
    retries::Int
    response::Union{LMResponse,Nothing}
    consumed::Bool
end

function LMResult(; request, start_stream, on_finished=nothing,
                    callable_registry=Dict{String,Function}(),
                    on_tool_call=nothing, max_tool_rounds=8, retries=0)
    LMResult(request, start_stream, on_finished, callable_registry,
             on_tool_call, max_tool_rounds, retries, nothing, false)
end

# ── Accessors ───────────────────────────────────────────────────────

function response(r::LMResult)
    r.response === nothing && consume!(r)
    r.response
end

text(r::LMResult) = let resp = response(r); resp !== nothing ? text(resp) : nothing end
thinking(r::LMResult) = let resp = response(r); resp !== nothing ? thinking(resp) : nothing end
tool_calls(r::LMResult) = let resp = response(r); resp !== nothing ? tool_calls(resp) : Part[] end
finish_reason(r::LMResult) = let resp = response(r); resp !== nothing ? resp.finish_reason : nothing end
usage(r::LMResult) = let resp = response(r); resp !== nothing ? resp.usage : Usage() end
image(r::LMResult) = let resp = response(r); resp !== nothing ? image(resp) : nothing end
audio(r::LMResult) = let resp = response(r); resp !== nothing ? audio(resp) : nothing end

function Base.parse(::Type{T}, r::LMResult) where T
    t = text(r)
    t === nothing && error("response contains no text")
    JSON.parse(t)
end

# ── Streaming ───────────────────────────────────────────────────────

function stream(r::LMResult)::Vector{StreamChunk}
    consume!(r)
end

function consume!(r::LMResult)
    r.consumed && return StreamChunk[]
    r.consumed = true
    run_stream!(r)
end

function run_stream!(r::LMResult)::Vector{StreamChunk}
    current_request = r.request
    all_chunks = StreamChunk[]
    rounds = 0

    while true
        state = RoundState(current_request)

        # Stream with retries
        events = stream_with_retries(r, current_request)

        for event in events
            if event.type == "error"
                err = event.error
                throw(err !== nothing ? error_for_code(err.code, err.message) : ProviderError("stream error"))
            end
            append!(all_chunks, apply!(state, event))
        end

        resp = materialize(state)
        r.response = resp

        # Yield tool_call chunks
        tcs = tool_calls(resp)
        for tc in tcs
            push!(all_chunks, StreamChunk("tool_call", nothing, tc.name, tc.input, nothing))
        end

        # Auto-execute tools
        if resp.finish_reason == "tool_call" && !isempty(tcs) && rounds < r.max_tool_rounds
            executed = execute_tools(r, tcs)
            if length(executed) == length(tcs)
                for outcome in executed
                    push!(all_chunks, StreamChunk("tool_result", outcome.preview, outcome.name, nothing, nothing))
                end
                tool_parts = [e.part for e in executed]
                tool_msg = Message(role="tool", parts=tool_parts)
                msgs = vcat(current_request.messages, [resp.message, tool_msg])
                current_request = LMRequest(
                    model=current_request.model, messages=msgs,
                    system=current_request.system, tools=current_request.tools,
                    config=current_request.config)
                rounds += 1
                continue
            end
        end

        # Finalize
        r.on_finished !== nothing && r.on_finished(current_request, resp)
        push!(all_chunks, StreamChunk("finished", nothing, nothing, nothing, resp))
        return all_chunks
    end
end

function stream_with_retries(r::LMResult, request::LMRequest)
    last_err = nothing
    for attempt in 0:r.retries
        try
            return r.start_stream(request)
        catch e
            (attempt == r.retries || !is_transient(e)) && rethrow()
            last_err = e
            sleep(0.2 * (1 << attempt))
        end
    end
    throw(last_err)
end

struct ExecutedTool
    name::String
    part::Part
    preview::String
end

function execute_tools(r::LMResult, tcs::Vector{Part})
    results = ExecutedTool[]
    for tc in tcs
        id = something(tc.id, "")
        name = something(tc.name, "tool")
        input = something(tc.input, Dict{String,Any}())
        info = ToolCallInfo(id, name, input)

        # Check on_tool_call callback
        if r.on_tool_call !== nothing
            override = r.on_tool_call(info)
            if override !== nothing
                content = [TextPart(string(override))]
                push!(results, ExecutedTool(name, ToolResultPart(id, content, name=name), string(override)))
                continue
            end
        end

        # Check callable registry
        if haskey(r.callable_registry, name)
            fn = r.callable_registry[name]
            output = try string(fn(input)) catch e "error: $e" end
            content = [TextPart(output)]
            push!(results, ExecutedTool(name, ToolResultPart(id, content, name=name), output))
            continue
        end

        # Can't execute — return partial
        return results
    end
    results
end

# ── RoundState ──────────────────────────────────────────────────────

mutable struct RoundState
    request::LMRequest
    started_id::String
    started_model::String
    finish_reason::Union{String,Nothing}
    usage_val::Union{Usage,Nothing}
    text_parts::Vector{String}
    thinking_parts::Vector{String}
    audio_chunks::Vector{String}
    tool_call_raw::Dict{Int,String}
    tool_call_meta::Dict{Int,Dict{String,Any}}
end

RoundState(request::LMRequest) = RoundState(request, "", "", nothing, nothing, String[], String[], String[], Dict{Int,String}(), Dict{Int,Dict{String,Any}}())

function apply!(state::RoundState, event::StreamEvent)::Vector{StreamChunk}
    chunks = StreamChunk[]

    if event.type == "start"
        event.id !== nothing && (state.started_id = event.id)
        event.model !== nothing && (state.started_model = event.model)
    elseif event.type == "end"
        event.finish_reason !== nothing && (state.finish_reason = event.finish_reason)
        event.usage !== nothing && (state.usage_val = event.usage)
    elseif event.type == "delta"
        if event.delta !== nothing
            dt = event.delta.type
            if dt == "text"
                txt = something(event.delta.text, "")
                push!(state.text_parts, txt)
                push!(chunks, StreamChunk("text", txt, nothing, nothing, nothing))
            elseif dt == "thinking"
                txt = something(event.delta.text, "")
                push!(state.thinking_parts, txt)
                push!(chunks, StreamChunk("thinking", txt, nothing, nothing, nothing))
            elseif dt == "audio"
                data = something(event.delta.data, "")
                push!(state.audio_chunks, data)
                push!(chunks, StreamChunk("audio", data, nothing, nothing, nothing))
            elseif dt == "tool_call"
                idx = something(event.part_index, 0)
                raw_input = something(event.delta.input, "")
                push_tool_call!(state, idx, raw_input)
            end
        end
        if event.delta_raw !== nothing
            if get(event.delta_raw, "type", nothing) == "tool_call"
                idx = something(event.part_index, 0)
                meta = get!(state.tool_call_meta, idx, Dict{String,Any}())
                haskey(event.delta_raw, "id") && event.delta_raw["id"] !== nothing && (meta["id"] = event.delta_raw["id"])
                haskey(event.delta_raw, "name") && event.delta_raw["name"] !== nothing && (meta["name"] = event.delta_raw["name"])
                if haskey(event.delta_raw, "input")
                    push_tool_call!(state, idx, string(event.delta_raw["input"]))
                end
            end
        end
    end
    chunks
end

function push_tool_call!(state::RoundState, idx::Int, raw_input::String)
    agg = get(state.tool_call_raw, idx, "") * raw_input
    state.tool_call_raw[idx] = agg
    meta = get!(state.tool_call_meta, idx, Dict{String,Any}())
    try
        parsed = JSON.parse(agg)
        if parsed isa Dict
            meta["input"] = parsed
        end
    catch; end
end

function materialize(state::RoundState)::LMResponse
    parts = Part[]

    !isempty(state.thinking_parts) && push!(parts, ThinkingPart(join(state.thinking_parts)))
    !isempty(state.text_parts) && push!(parts, TextPart(join(state.text_parts)))
    !isempty(state.audio_chunks) && push!(parts, AudioBase64(join(state.audio_chunks), "audio/wav"))

    tool_names = [t.name for t in state.request.tools if t.type == "function"]
    indices = sort(collect(keys(state.tool_call_meta)))
    for (pos, idx) in enumerate(indices)
        meta = state.tool_call_meta[idx]
        payload = get(meta, "input", nothing)
        if !(payload isa Dict)
            payload = try JSON.parse(get(state.tool_call_raw, idx, "{}")) catch; Dict{String,Any}() end
        end
        name = get(meta, "name", nothing)
        if name === nothing
            name = length(tool_names) == 1 ? tool_names[1] : (pos <= length(tool_names) ? tool_names[pos] : "tool")
        end
        id = get(meta, "id", "tool_call_$idx")
        push!(parts, ToolCallPart(string(id), string(name), payload))
    end

    isempty(parts) && push!(parts, TextPart(""))
    has_tc = any(p -> p.type == "tool_call", parts)
    finish = if state.finish_reason !== nothing
        (state.finish_reason == "stop" && has_tc) ? "tool_call" : state.finish_reason
    else
        has_tc ? "tool_call" : "stop"
    end

    model = isempty(state.started_model) ? state.request.model : state.started_model
    usage_val = state.usage_val !== nothing ? state.usage_val : Usage()

    LMResponse(state.started_id, model, Message(role="assistant", parts=parts), finish, usage_val, nothing)
end
