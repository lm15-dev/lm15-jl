# Gemini adapter (GenerativeLanguage API).

struct GeminiAdapter
    api_key::String
    base_url::String
end

GeminiAdapter(api_key; base_url="https://generativelanguage.googleapis.com/v1beta") =
    GeminiAdapter(api_key, base_url)

provider_name(::GeminiAdapter) = "gemini"
env_keys(::GeminiAdapter) = ["GEMINI_API_KEY", "GOOGLE_API_KEY"]

model_path(model) = startswith(model, "models/") ? model : "models/$model"

function gemini_part(p::Part)
    if p.type == "text"
        return Dict{String,Any}("text" => something(p.text, ""))
    elseif p.type in ("image", "audio", "video", "document") && p.source !== nothing
        mime = something(p.source.media_type, "application/octet-stream")
        if p.source.type == "url"
            return Dict{String,Any}("fileData" => Dict{String,Any}("mimeType"=>mime, "fileUri"=>p.source.url))
        elseif p.source.type == "base64"
            return Dict{String,Any}("inlineData" => Dict{String,Any}("mimeType"=>mime, "data"=>p.source.data))
        end
    elseif p.type == "tool_call"
        fc = Dict{String,Any}("name"=>something(p.name, ""), "args"=>something(p.input, Dict{String,Any}()))
        p.id !== nothing && (fc["id"] = p.id)
        return Dict{String,Any}("functionCall" => fc)
    elseif p.type == "tool_result"
        txt = p.content !== nothing ? parts_to_text(p.content) : ""
        fr = Dict{String,Any}("name"=>something(p.name, "tool"), "response"=>Dict{String,Any}("result"=>txt))
        p.id !== nothing && (fr["id"] = p.id)
        return Dict{String,Any}("functionResponse" => fr)
    end
    Dict{String,Any}("text" => something(p.text, ""))
end

function build_request(a::GeminiAdapter, request::LMRequest, stream::Bool)
    contents = [Dict{String,Any}(
        "role" => m.role == "assistant" ? "model" : "user",
        "parts" => [gemini_part(p) for p in m.parts]
    ) for m in request.messages]

    payload = Dict{String,Any}("contents" => contents)

    if request.system !== nothing
        payload["systemInstruction"] = Dict{String,Any}("parts" => [Dict{String,Any}("text" => request.system)])
    end

    cfg = Dict{String,Any}()
    request.config.temperature !== nothing && (cfg["temperature"] = request.config.temperature)
    request.config.max_tokens !== nothing && (cfg["maxOutputTokens"] = request.config.max_tokens)
    request.config.stop !== nothing && (cfg["stopSequences"] = request.config.stop)
    !isempty(cfg) && (payload["generationConfig"] = cfg)

    tools = [Dict{String,Any}("name"=>t.name, "description"=>t.description,
             "parameters"=>something(t.parameters, Dict{String,Any}("type"=>"OBJECT","properties"=>Dict{String,Any}())))
             for t in request.tools if t.type == "function"]
    !isempty(tools) && (payload["tools"] = [Dict{String,Any}("functionDeclarations" => tools)])

    if request.config.provider !== nothing
        for (k, v) in request.config.provider
            k in ("prompt_caching", "output") && continue
            payload[k] = v
        end
    end

    endpoint = stream ? "streamGenerateContent" : "generateContent"
    params = Dict{String,String}()
    stream && (params["alt"] = "sse")

    body = Vector{UInt8}(JSON.serialize(payload))
    HttpRequest(method="POST",
        url="$(a.base_url)/$(model_path(request.model)):$endpoint",
        headers=Dict("x-goog-api-key"=>a.api_key, "Content-Type"=>"application/json"),
        params=params, body=body, timeout=stream ? 120.0 : 60.0)
end

function normalize_error(a::GeminiAdapter, status::Int, body::String)
    try
        data = JSON.parse(body)
        err = get(data, "error", Dict())
        msg = get(err, "message", "")
        err_status = get(err, "status", "")
        is_ctx = let m = lowercase(msg)
            (occursin("token", m) && (occursin("limit", m) || occursin("exceed", m))) || occursin("too long", m) || occursin("context length", m)
        end
        is_ctx && return ContextLengthError(msg)

        err_status == "PERMISSION_DENIED" && return AuthError(msg)
        err_status == "RESOURCE_EXHAUSTED" && return RateLimitError(msg)
        err_status == "FAILED_PRECONDITION" && return BillingError(msg)
        err_status in ("INTERNAL", "UNAVAILABLE") && return ServerError(msg)
        err_status == "DEADLINE_EXCEEDED" && return TimeoutError(msg)
        return map_http_error(status, msg)
    catch
        return map_http_error(status, body[1:min(200,length(body))])
    end
end

function parse_response(a::GeminiAdapter, request::LMRequest, response::HttpResponse)
    data = JSON.parse(String(response.body))

    # In-band error
    pf = get(data, "promptFeedback", nothing)
    if pf isa Dict
        reason = get(pf, "blockReason", "")
        reason != "" && reason != "BLOCK_REASON_UNSPECIFIED" && throw(InvalidRequestError("Prompt blocked: $reason"))
    end

    candidates = get(data, "candidates", Any[])
    candidate = !isempty(candidates) && candidates[1] isa Dict ? candidates[1] : Dict()
    content = get(candidate, "content", Dict())
    raw_parts = get(content, "parts", Any[])

    parts = Part[]
    for p in raw_parts
        p isa Dict || continue
        if haskey(p, "text")
            push!(parts, TextPart(string(p["text"])))
        elseif haskey(p, "functionCall")
            fc = p["functionCall"]
            args = get(fc, "args", Dict{String,Any}())
            push!(parts, ToolCallPart(get(fc, "id", "fc_0"), get(fc, "name", ""), args isa Dict ? args : Dict{String,Any}()))
        elseif haskey(p, "inlineData")
            inline = p["inlineData"]
            mime = get(inline, "mimeType", "application/octet-stream")
            data_str = get(inline, "data", "")
            if startswith(mime, "image/")
                push!(parts, ImageBase64(data_str, mime))
            elseif startswith(mime, "audio/")
                push!(parts, AudioBase64(data_str, mime))
            end
        end
    end

    isempty(parts) && push!(parts, TextPart(""))
    has_tc = any(p -> p.type == "tool_call", parts)

    um = get(data, "usageMetadata", Dict())
    usage = Usage(
        input_tokens=safe_int(get(um, "promptTokenCount", 0)),
        output_tokens=safe_int(get(um, "candidatesTokenCount", 0)),
        total_tokens=safe_int(get(um, "totalTokenCount", 0)),
        cache_read_tokens=get(um, "cachedContentTokenCount", nothing),
        reasoning_tokens=get(um, "thoughtsTokenCount", nothing),
    )

    LMResponse(get(data, "responseId", ""), request.model,
        Message(role="assistant", parts=parts),
        has_tc ? "tool_call" : "stop", usage, data)
end

function parse_stream_event(a::GeminiAdapter, request::LMRequest, raw::SSEEvent)
    isempty(raw.data) && return nothing
    payload = JSON.parse(raw.data)

    if haskey(payload, "error")
        e = payload["error"]
        code = e isa Dict ? get(e, "status", get(e, "code", "provider")) : "provider"
        msg = e isa Dict ? get(e, "message", "") : ""
        return StreamEvent(type="error", error=ErrorInfo(string(code), string(msg), nothing))
    end

    cands = get(payload, "candidates", Any[])
    isempty(cands) && return nothing
    cand = cands[1]
    content = get(cand, "content", Dict())
    parts_list = get(content, "parts", Any[])
    isempty(parts_list) && return nothing
    part = parts_list[1]

    if haskey(part, "text")
        return StreamEvent(type="delta", part_index=0, delta=PartDelta("text", string(part["text"]), nothing, nothing))
    elseif haskey(part, "functionCall")
        fc = part["functionCall"]
        args = JSON.serialize(get(fc, "args", Dict()))
        return StreamEvent(type="delta", part_index=0,
            delta_raw=Dict{String,Any}("type"=>"tool_call", "id"=>get(fc,"id","fc_0"),
                "name"=>get(fc,"name",""), "input"=>args))
    elseif haskey(part, "inlineData")
        inline = part["inlineData"]
        mime = get(inline, "mimeType", "")
        if startswith(mime, "audio/")
            return StreamEvent(type="delta", part_index=0, delta=PartDelta("audio", nothing, get(inline,"data",""), nothing))
        end
    end
    nothing
end

function do_embeddings(a::GeminiAdapter, request::EmbeddingRequest)
    mp = model_path(request.model)
    if length(request.inputs) <= 1
        input = isempty(request.inputs) ? "" : request.inputs[1]
        payload = Dict{String,Any}("model"=>mp, "content"=>Dict{String,Any}("parts"=>[Dict{String,Any}("text"=>input)]))
        body = Vector{UInt8}(JSON.serialize(payload))
        req = HttpRequest(method="POST", url="$(a.base_url)/$mp:embedContent",
            headers=Dict("x-goog-api-key"=>a.api_key, "Content-Type"=>"application/json"),
            body=body, timeout=60.0)
        resp = http_request(req)
        resp.status >= 400 && throw(normalize_error(a, resp.status, text(resp)))
        data = JSON.parse(String(resp.body))
        values = Float64.(get(get(data, "embedding", Dict()), "values", Float64[]))
        return EmbeddingResponse(request.model, [values], Usage(), data)
    end

    requests = [Dict{String,Any}("model"=>mp, "content"=>Dict{String,Any}("parts"=>[Dict{String,Any}("text"=>input)])) for input in request.inputs]
    payload = Dict{String,Any}("requests" => requests)
    body = Vector{UInt8}(JSON.serialize(payload))
    req = HttpRequest(method="POST", url="$(a.base_url)/$mp:batchEmbedContents",
        headers=Dict("x-goog-api-key"=>a.api_key, "Content-Type"=>"application/json"),
        body=body, timeout=60.0)
    resp = http_request(req)
    resp.status >= 400 && throw(normalize_error(a, resp.status, text(resp)))
    data = JSON.parse(String(resp.body))
    vectors = [Float64.(get(e, "values", Float64[])) for e in get(data, "embeddings", Any[])]
    EmbeddingResponse(request.model, vectors, Usage(), data)
end
