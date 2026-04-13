# Anthropic adapter (Messages API).

const _ANTHROPIC_BUILTIN_MAP = Dict(
    "code_execution" => "code_execution_20250522",
)

function _builtin_to_anthropic(tool)
    wire_type = get(_ANTHROPIC_BUILTIN_MAP, tool.name, tool.name)
    out = Dict{String,Any}("type" => wire_type, "name" => tool.name)
    if tool.builtin_config !== nothing
        for (k, v) in tool.builtin_config
            out[k] = v
        end
    end
    out
end

struct AnthropicAdapter
    api_key::String
    base_url::String
    api_version::String
end

AnthropicAdapter(api_key; base_url="https://api.anthropic.com/v1", api_version="2023-06-01") =
    AnthropicAdapter(api_key, base_url, api_version)

provider_name(::AnthropicAdapter) = "anthropic"
env_keys(::AnthropicAdapter) = ["ANTHROPIC_API_KEY"]

function build_request(a::AnthropicAdapter, request::LMRequest, stream::Bool)
    messages = [Dict{String,Any}("role" => m.role == "tool" ? "user" : m.role,
        "content" => [anthropic_part(p) for p in m.parts]) for m in request.messages]

    max_tokens = something(request.config.max_tokens, 1024)
    payload = Dict{String,Any}("model" => request.model, "messages" => messages,
        "stream" => stream, "max_tokens" => max_tokens)

    request.system !== nothing && (payload["system"] = request.system)
    request.config.temperature !== nothing && (payload["temperature"] = request.config.temperature)

    tools_wire = Dict{String,Any}[]
    for t in request.tools
        if t.type == "function"
            push!(tools_wire, Dict{String,Any}("name"=>t.name, "description"=>t.description,
                "input_schema"=>something(t.parameters, Dict{String,Any}("type"=>"object","properties"=>Dict{String,Any}()))))
        elseif t.type == "builtin"
            push!(tools_wire, _builtin_to_anthropic(t))
        end
    end
    !isempty(tools_wire) && (payload["tools"] = tools_wire)

    if request.config.reasoning !== nothing && get(request.config.reasoning, "enabled", false) == true
        budget = get(request.config.reasoning, "budget", 1024)
        payload["thinking"] = Dict{String,Any}("type"=>"enabled", "budget_tokens"=>budget)
    end

    if request.config.provider !== nothing
        for (k, v) in request.config.provider
            k == "prompt_caching" && continue
            payload[k] = v
        end
    end

    body = Vector{UInt8}(JSON.serialize(payload))
    HttpRequest(method="POST", url="$(a.base_url)/messages",
        headers=Dict("x-api-key"=>a.api_key, "anthropic-version"=>a.api_version, "content-type"=>"application/json"),
        body=body, timeout=stream ? 120.0 : 60.0)
end

function anthropic_part(p::Part)
    if p.type == "text"
        return Dict{String,Any}("type"=>"text", "text"=>something(p.text, ""))
    elseif p.type == "tool_call"
        return Dict{String,Any}("type"=>"tool_use", "id"=>p.id, "name"=>p.name,
            "input"=>something(p.input, Dict{String,Any}()))
    elseif p.type == "tool_result"
        txt = p.content !== nothing ? parts_to_text(p.content) : ""
        out = Dict{String,Any}("type"=>"tool_result", "tool_use_id"=>p.id)
        !isempty(txt) && (out["content"] = txt)
        p.is_error === true && (out["is_error"] = true)
        return out
    elseif p.type == "image" && p.source !== nothing
        src = p.source
        if src.type == "url"
            return Dict{String,Any}("type"=>"image", "source"=>Dict{String,Any}("type"=>"url","url"=>src.url))
        else
            return Dict{String,Any}("type"=>"image", "source"=>Dict{String,Any}("type"=>"base64","media_type"=>src.media_type,"data"=>src.data))
        end
    elseif p.type == "document" && p.source !== nothing
        src = p.source
        if src.type == "url"
            return Dict{String,Any}("type"=>"document", "source"=>Dict{String,Any}("type"=>"url","url"=>src.url))
        else
            return Dict{String,Any}("type"=>"document", "source"=>Dict{String,Any}("type"=>"base64","media_type"=>src.media_type,"data"=>src.data))
        end
    end
    Dict{String,Any}("type"=>"text", "text"=>something(p.text, ""))
end

function normalize_error(a::AnthropicAdapter, status::Int, body::String)
    try
        data = JSON.parse(body)
        err = get(data, "error", Dict())
        msg = get(err, "message", "")
        err_type = get(err, "type", "")
        if err_type == "invalid_request_error" && is_context_msg(msg)
            return ContextLengthError(msg)
        end
        err_type == "authentication_error" && return AuthError(msg)
        err_type == "permission_error" && return AuthError(msg)
        err_type == "rate_limit_error" && return RateLimitError(msg)
        err_type == "billing_error" && return BillingError(msg)
        err_type in ("api_error", "overloaded_error") && return ServerError(msg)
        err_type == "timeout_error" && return TimeoutError(msg)
        return map_http_error(status, msg)
    catch
        return map_http_error(status, body[1:min(200,length(body))])
    end
end

is_context_msg(msg) = let m = lowercase(msg)
    occursin("prompt is too long", m) || occursin("too many tokens", m) ||
    occursin("context window", m) || occursin("context length", m)
end

function parse_response(a::AnthropicAdapter, request::LMRequest, response::HttpResponse)
    data = JSON.parse(String(response.body))
    parts = Part[]
    for block in get(data, "content", Any[])
        block isa Dict || continue
        bt = get(block, "type", "")
        if bt == "text"
            push!(parts, TextPart(get(block, "text", "")))
        elseif bt == "tool_use"
            input = get(block, "input", Dict{String,Any}())
            push!(parts, ToolCallPart(get(block, "id", ""), get(block, "name", ""), input))
        elseif bt == "thinking"
            push!(parts, ThinkingPart(get(block, "thinking", "")))
        end
    end

    isempty(parts) && push!(parts, TextPart(""))
    has_tc = any(p -> p.type == "tool_call", parts)
    finish = has_tc ? "tool_call" : "stop"

    u = get(data, "usage", Dict())
    usage = Usage(
        input_tokens=safe_int(get(u, "input_tokens", 0)),
        output_tokens=safe_int(get(u, "output_tokens", 0)),
        total_tokens=safe_int(get(u, "input_tokens", 0)) + safe_int(get(u, "output_tokens", 0)),
        cache_read_tokens=get(u, "cache_read_input_tokens", nothing),
        cache_write_tokens=get(u, "cache_creation_input_tokens", nothing),
    )

    LMResponse(get(data, "id", ""), get(data, "model", request.model),
        Message(role="assistant", parts=parts), finish, usage, data)
end

function parse_stream_event(a::AnthropicAdapter, request::LMRequest, raw::SSEEvent)
    isempty(raw.data) && return nothing
    p = JSON.parse(raw.data)
    et = get(p, "type", "")

    if et == "message_start"
        msg = get(p, "message", Dict())
        return StreamEvent(type="start", id=get(msg,"id",""), model=get(msg,"model",""))
    elseif et == "content_block_start"
        block = get(p, "content_block", Dict())
        if get(block, "type", "") == "tool_use"
            idx = safe_int(get(p, "index", 0))
            return StreamEvent(type="delta", part_index=idx,
                delta_raw=Dict{String,Any}("type"=>"tool_call","id"=>get(block,"id",nothing),"name"=>get(block,"name",nothing),"input"=>""))
        end
    elseif et == "content_block_delta"
        delta = get(p, "delta", Dict())
        idx = safe_int(get(p, "index", 0))
        dt = get(delta, "type", "")
        if dt == "text_delta"
            return StreamEvent(type="delta", part_index=idx, delta=PartDelta("text", get(delta,"text",""), nothing, nothing))
        elseif dt == "input_json_delta"
            return StreamEvent(type="delta", part_index=idx,
                delta_raw=Dict{String,Any}("type"=>"tool_call","input"=>get(delta,"partial_json","")))
        elseif dt == "thinking_delta"
            return StreamEvent(type="delta", part_index=idx, delta=PartDelta("thinking", get(delta,"thinking",""), nothing, nothing))
        end
    elseif et == "message_stop"
        return StreamEvent(type="end", finish_reason="stop")
    elseif et == "error"
        e = get(p, "error", Dict())
        code = get(e, "type", "provider")
        msg = get(e, "message", "")
        return StreamEvent(type="error", error=ErrorInfo(string(code), string(msg), nothing))
    end
    nothing
end
