# OpenAI adapter (Responses API).

struct OpenAIAdapter
    api_key::String
    base_url::String
end

OpenAIAdapter(api_key; base_url="https://api.openai.com/v1") = OpenAIAdapter(api_key, base_url)

provider_name(::OpenAIAdapter) = "openai"
env_keys(::OpenAIAdapter) = ["OPENAI_API_KEY"]

function build_request(a::OpenAIAdapter, request::LMRequest, stream::Bool)
    messages = [message_to_openai_input(m) for m in request.messages]
    payload = Dict{String,Any}("model" => request.model, "input" => messages, "stream" => stream)

    request.system !== nothing && (payload["instructions"] = request.system)
    request.config.max_tokens !== nothing && (payload["max_output_tokens"] = request.config.max_tokens)
    request.config.temperature !== nothing && (payload["temperature"] = request.config.temperature)

    tools = [Dict{String,Any}("type"=>"function", "name"=>t.name, "description"=>t.description,
             "parameters"=>something(t.parameters, Dict{String,Any}("type"=>"object","properties"=>Dict{String,Any}())))
             for t in request.tools if t.type == "function"]
    !isempty(tools) && (payload["tools"] = tools)

    if request.config.provider !== nothing
        for (k, v) in request.config.provider
            k == "prompt_caching" && continue
            payload[k] = v
        end
    end

    body = Vector{UInt8}(JSON.serialize(payload))
    HttpRequest(method="POST", url="$(a.base_url)/responses",
        headers=Dict("Authorization"=>"Bearer $(a.api_key)", "Content-Type"=>"application/json"),
        body=body, timeout=stream ? 120.0 : 60.0)
end

function normalize_error(a::OpenAIAdapter, status::Int, body::String)
    try
        data = JSON.parse(body)
        err = get(data, "error", Dict())
        msg = get(err, "message", "")
        code = get(err, "code", "")
        err_type = get(err, "type", "")
        code == "context_length_exceeded" && return ContextLengthError(msg)
        (code == "insufficient_quota" || err_type == "insufficient_quota") && return BillingError(msg)
        (code == "invalid_api_key" || err_type == "authentication_error") && return AuthError(msg)
        (code == "rate_limit_exceeded" || err_type == "rate_limit_error") && return RateLimitError(msg)
        return map_http_error(status, msg)
    catch
        return map_http_error(status, body[1:min(200,length(body))])
    end
end

function parse_response(a::OpenAIAdapter, request::LMRequest, response::HttpResponse)
    data = JSON.parse(String(response.body))

    # In-band error
    if haskey(data, "error") && data["error"] isa Dict
        throw(ServerError(get(data["error"], "message", "server error")))
    end

    parts = Part[]
    for item in get(data, "output", Any[])
        item isa Dict || continue
        if get(item, "type", "") == "message"
            for c in get(item, "content", Any[])
                c isa Dict || continue
                ct = get(c, "type", "")
                if ct in ("output_text", "text")
                    push!(parts, TextPart(get(c, "text", "")))
                elseif ct == "refusal"
                    push!(parts, RefusalPart(get(c, "refusal", "")))
                end
            end
        elseif get(item, "type", "") == "function_call"
            args_str = get(item, "arguments", "{}")
            args = try JSON.parse(args_str) catch; Dict{String,Any}() end
            push!(parts, ToolCallPart(get(item, "call_id", ""), get(item, "name", ""), args))
        end
    end

    isempty(parts) && push!(parts, TextPart(get(data, "output_text", "")))
    has_tc = any(p -> p.type == "tool_call", parts)
    finish = has_tc ? "tool_call" : "stop"

    u = get(data, "usage", Dict())
    u_in = get(u, "input_tokens_details", Dict())
    u_out = get(u, "output_tokens_details", Dict())
    usage = Usage(
        input_tokens=safe_int(get(u, "input_tokens", 0)),
        output_tokens=safe_int(get(u, "output_tokens", 0)),
        total_tokens=safe_int(get(u, "total_tokens", 0)),
        reasoning_tokens=get(u_out, "reasoning_tokens", nothing),
        cache_read_tokens=get(u_in, "cached_tokens", nothing),
    )

    LMResponse(
        get(data, "id", ""),
        get(data, "model", request.model),
        Message(role="assistant", parts=parts),
        finish, usage, data)
end

function parse_stream_event(a::OpenAIAdapter, request::LMRequest, raw::SSEEvent)
    isempty(raw.data) && return nothing
    raw.data == "[DONE]" && return StreamEvent(type="end", finish_reason="stop")

    p = JSON.parse(raw.data)
    et = get(p, "type", "")

    if et == "response.created"
        id = get_nested(p, "response", "id", default="")
        return StreamEvent(type="start", id=id, model=request.model)
    elseif et in ("response.output_text.delta", "response.refusal.delta")
        return StreamEvent(type="delta", part_index=0, delta=PartDelta("text", get(p, "delta", ""), nothing, nothing))
    elseif et == "response.output_item.added"
        item = get(p, "item", Dict())
        if get(item, "type", "") == "function_call"
            idx = safe_int(get(p, "output_index", 0))
            return StreamEvent(type="delta", part_index=idx,
                delta_raw=Dict{String,Any}("type"=>"tool_call", "id"=>get(item,"call_id",nothing),
                    "name"=>get(item,"name",nothing), "input"=>get(item,"arguments","")))
        end
    elseif et == "response.function_call_arguments.delta"
        idx = safe_int(get(p, "output_index", 0))
        return StreamEvent(type="delta", part_index=idx,
            delta_raw=Dict{String,Any}("type"=>"tool_call", "id"=>get(p,"call_id",nothing),
                "name"=>get(p,"name",nothing), "input"=>get(p,"delta","")))
    elseif et == "response.completed"
        resp = get(p, "response", Dict())
        u = get(resp, "usage", Dict())
        u_in = get(u, "input_tokens_details", Dict())
        u_out = get(u, "output_tokens_details", Dict())
        usage = Usage(
            input_tokens=safe_int(get(u, "input_tokens", 0)),
            output_tokens=safe_int(get(u, "output_tokens", 0)),
            total_tokens=safe_int(get(u, "total_tokens", 0)),
            reasoning_tokens=get(u_out, "reasoning_tokens", nothing),
            cache_read_tokens=get(u_in, "cached_tokens", nothing),
        )
        has_fc = any(i -> get(i, "type", "") == "function_call", get(resp, "output", Any[]))
        finish = has_fc ? "tool_call" : "stop"
        return StreamEvent(type="end", finish_reason=finish, usage=usage)
    elseif et in ("response.error", "error")
        err = get(p, "error", Dict())
        code = err isa Dict ? get(err, "code", get(err, "type", "provider")) : "provider"
        msg = err isa Dict ? get(err, "message", "") : ""
        return StreamEvent(type="error", error=ErrorInfo(string(code), string(msg), nothing))
    end
    nothing
end

function do_embeddings(a::OpenAIAdapter, request::EmbeddingRequest)
    payload = Dict{String,Any}("model" => request.model, "input" => request.inputs)
    body = Vector{UInt8}(JSON.serialize(payload))
    req = HttpRequest(method="POST", url="$(a.base_url)/embeddings",
        headers=Dict("Authorization"=>"Bearer $(a.api_key)", "Content-Type"=>"application/json"),
        body=body, timeout=60.0)
    resp = http_request(req)
    resp.status >= 400 && throw(normalize_error(a, resp.status, text(resp)))
    data = JSON.parse(String(resp.body))
    vectors = [Float64.(get(item, "embedding", Float64[])) for item in get(data, "data", Any[])]
    EmbeddingResponse(request.model, vectors, Usage(), data)
end
