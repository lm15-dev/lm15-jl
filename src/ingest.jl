const INGEST_EXTENSION_KEYS=(
    "seed",
    "logit_bias",
    "presence_penalty",
    "frequency_penalty",
    "metadata",
    "verbosity",
    "moderation",
    "provider",
)
const INGEST_CONFIG_KEYS=(
    "model",
    "messages",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "max_completion_tokens",
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "seed",
    "frequency_penalty",
    "presence_penalty",
    "functions",
    "function_call",
    "stop",
    "logprobs",
    "top_logprobs",
    "response_format",
    "service_tier",
    "store",
    "user",
    "safety_identifier",
    "user_id",
    "reasoning_effort",
    "reasoning",
    "thinking",
    "enable_thinking",
    "chat_template_kwargs",
    "reasoning_format",
    "prompt_cache_key",
    "prompt_cache_retention",
    "prompt_cache_options",
)
const INGEST_REFUSED_KEYS=(
    "n",
    "audio",
    "modalities",
    "prediction",
    "web_search_options",
)
function ingest_keys(d, allowed, where)
    d isa AbstractDict || throw(ArgumentError("$where must be an object"))
    for key in keys(d)
        key in allowed || unsupported("openai-chat", "$where.$key: no canonical mapping")
    end
    return d
end
function ingest_string(v, where)
    return v isa AbstractString ? String(v) : throw(ArgumentError("$where must be a string"))
end
function ingest_uri(uri)
    m=match(r"^data:([^;,]+);base64,(.+)$"s, uri)
    m===nothing && throw(ArgumentError("expected data:<media-type>;base64,<payload>"))
    return String(m[1]), String(m[2])
end
function ingest_blocks(content, role)
    content isa AbstractString && return Part[TextPart(content)], false
    content isa AbstractVector || throw(ArgumentError("message content must be text or an array"))
    parts=Part[]
    marked=false
    for (i, block) in enumerate(content)
        block isa AbstractDict || throw(ArgumentError("content block must be an object"))
        tag=get(block, "type", nothing)
        if get(block, "prompt_cache_breakpoint", nothing)!==nothing
            block["prompt_cache_breakpoint"]==obj("mode"=>"explicit") &&
            tag=="text" &&
            i==length(content) || throw(ArgumentError("breakpoint must mark the last text block"))
            marked=true
        end
        if tag=="text"
            ingest_keys(block, ("type", "text", "prompt_cache_breakpoint"), "text")
            push!(parts, TextPart(ingest_string(get(block, "text", nothing), "text")))
        elseif tag=="image_url" && role in ("user", "tool")
            ingest_keys(block, ("type", "image_url", "prompt_cache_breakpoint"), "image")
            spec=ingest_keys(get(block, "image_url", nothing), ("url", "detail"), "image_url")
            url=ingest_string(get(spec, "url", nothing), "image_url.url")
            detail=get(spec, "detail", nothing)
            if startswith(url, "data:")
                mime, data=ingest_uri(url)
                push!(parts, ImagePart(; media_type=mime, data, detail))
            else
                extension=lowercase(split(first(split(url, '?')), '.')[end])
                mime=get(
                    Dict(
                        "jpg"=>"image/jpeg",
                        "jpeg"=>"image/jpeg",
                        "png"=>"image/png",
                        "webp"=>"image/webp",
                        "gif"=>"image/gif",
                    ),
                    extension,
                    "image/png",
                )
                push!(parts, ImagePart(; media_type=mime, url, detail))
            end
        elseif tag=="input_audio" && role=="user"
            ingest_keys(block, ("type", "input_audio"), "audio")
            spec=ingest_keys(get(block, "input_audio", nothing), ("data", "format"), "input_audio")
            format=get(spec, "format", nothing)
            format in ("wav", "mp3") ||
                throw(ArgumentError("input_audio format must be wav or mp3"))
            push!(
                parts,
                AudioPart(;
                    data=ingest_string(get(spec, "data", nothing), "input_audio.data"),
                    media_type=format=="wav" ? "audio/wav" : "audio/mpeg",
                ),
            )
        elseif tag=="file" && role=="user"
            ingest_keys(block, ("type", "file"), "file")
            spec=ingest_keys(
                get(block, "file", nothing), ("file_data", "file_id", "filename"), "file"
            )
            get(spec, "filename", nothing)===nothing ||
                unsupported("openai-chat", "document filename")
            id=get(spec, "file_id", nothing)
            data=get(spec, "file_data", nothing)
            (id===nothing)!=(data===nothing) || throw(ArgumentError("file needs one source"))
            if id!==nothing
                push!(parts, DocumentPart(; file_id=ingest_string(id, "file_id")))
            else
                mime, payload=ingest_uri(ingest_string(data, "file_data"))
                push!(parts, DocumentPart(; media_type=mime, data=payload))
            end
        elseif tag=="refusal" && role=="assistant"
            ingest_keys(block, ("type", "refusal"), "refusal")
            push!(parts, RefusalPart(ingest_string(get(block, "refusal", nothing), "refusal")))
        else
            unsupported("openai-chat", "$tag content in $role message")
        end
    end
    return parts, marked
end
function ingest_calls(raw)
    raw isa AbstractVector || throw(ArgumentError("tool_calls must be an array"))
    parts=Part[]
    for call in raw
        ingest_keys(call, ("id", "type", "function"), "tool_calls")
        get(call, "type", "function")=="function" ||
            unsupported("openai-chat", "non-function tool call")
        f=ingest_keys(get(call, "function", nothing), ("name", "arguments"), "function")
        arguments=get(f, "arguments", "{}")
        input=if arguments isa AbstractString
            JSON.parse(isempty(arguments) ? "{}" : arguments)
        else
            arguments
        end
        input isa AbstractDict || throw(ArgumentError("tool arguments must encode an object"))
        push!(
            parts,
            ToolCallPart(;
                id=ingest_string(get(call, "id", nothing), "tool call id"),
                name=ingest_string(get(f, "name", nothing), "tool name"),
                input,
            ),
        )
    end
    return parts
end
function ingest_annotations(raw, content)
    raw isa AbstractVector || throw(ArgumentError("annotations must be an array"))
    parts=Part[]
    for entry in raw
        ingest_keys(entry, ("type", "url_citation"), "annotation")
        get(entry, "type", nothing)=="url_citation" ||
            unsupported("openai-chat", "non-URL annotation")
        spec=ingest_keys(
            get(entry, "url_citation", nothing),
            ("url", "title", "start_index", "end_index"),
            "url_citation",
        )
        url=ingest_string(get(spec, "url", nothing), "citation URL")
        part=citation_from(
            merge(spec, obj("url"=>url)); source=content isa AbstractString ? content : nothing
        )
        part===nothing || push!(parts, part)
    end
    return parts
end
function ingest_messages(raw)
    raw isa AbstractVector || throw(ArgumentError("messages must be an array"))
    messages=Message[]
    system=nothing
    boundary=nothing
    stable=false
    pending=ToolResultPart[]
    function flush_results()
        return isempty(pending) ||
               (push!(messages, Message("tool", Tuple(pending))); empty!(pending))
    end
    function mark(index)
        boundary===nothing && !stable || throw(ArgumentError("more than one cache breakpoint"))
        return boundary=index
    end
    for (position, row) in enumerate(raw)
        row isa AbstractDict || throw(ArgumentError("message must be an object"))
        role=get(row, "role", nothing)
        role=="tool" || flush_results()
        if role in ("system", "developer", "user")
            get(row, "name", nothing)===nothing || unsupported("openai-chat", "participant name")
            ingest_keys(row, ("role", "content", "name"), "message")
            parts, marked=ingest_blocks(
                get(row, "content", nothing), role=="user" ? "user" : "system"
            )
            if position==1 && role!="user"
                system=if length(parts)==1 && first(parts) isa TextPart
                    first(parts).text
                else
                    Tuple(parts)
                end
                stable=marked
            else
                marked && mark(length(messages))
                push!(messages, Message(role=="user" ? "user" : "developer", Tuple(parts)))
            end
        elseif role=="assistant"
            ingest_keys(
                row,
                (
                    "role",
                    "content",
                    "tool_calls",
                    "refusal",
                    "reasoning_content",
                    "name",
                    "audio",
                    "function_call",
                    "annotations",
                    "provider_specific_fields",
                    "thinking_blocks",
                    "images",
                ),
                "assistant",
            )
            for key in ("name", "audio", "function_call")
                get(row, key, nothing)===nothing || unsupported("openai-chat", "assistant.$key")
            end
            for key in ("provider_specific_fields", "thinking_blocks", "images")
                value=get(row, key, nothing)
                empty_optional(value) ||
                    (value isa AbstractDict && all(empty_optional, values(value))) ||
                    unsupported("openai-chat", "assistant.$key")
            end
            parts=Part[]
            reason=get(row, "reasoning_content", nothing)
            reason===nothing ||
                push!(parts, ThinkingPart(ingest_string(reason, "reasoning_content")))
            content=get(row, "content", nothing)
            if content!==nothing
                blocks, marked=ingest_blocks(content, "assistant")
                marked && throw(ArgumentError("assistant breakpoint is not representable"))
                append!(parts, blocks)
            end
            refusal=get(row, "refusal", nothing)
            refusal===nothing || push!(parts, RefusalPart(ingest_string(refusal, "refusal")))
            get(row, "tool_calls", nothing)===nothing ||
                append!(parts, ingest_calls(row["tool_calls"]))
            get(row, "annotations", nothing)===nothing ||
                append!(parts, ingest_annotations(row["annotations"], content))
            isempty(parts) && push!(parts, TextPart(""))
            push!(messages, assistant(parts))
        elseif role=="tool"
            ingest_keys(row, ("role", "content", "tool_call_id", "name"), "tool")
            parts, marked=ingest_blocks(get(row, "content", nothing), "tool")
            marked && throw(ArgumentError("tool breakpoint is not representable"))
            push!(
                pending,
                ToolResultPart(;
                    id=ingest_string(get(row, "tool_call_id", nothing), "tool_call_id"),
                    content=Tuple(parts),
                    name=get(row, "name", nothing),
                ),
            )
        elseif role=="function"
            unsupported("openai-chat", "deprecated function message")
        else
            throw(ArgumentError("unknown message role"))
        end
    end
    flush_results()
    return system, Tuple(messages), stable, boundary
end
function ingest_tools(raw, c)
    raw===nothing && return ()
    raw isa AbstractVector || throw(ArgumentError("tools must be an array"))
    tools=Tool[]
    inverse=Dict(v=>k for (k, v) in GROQ_BUILTINS)
    for entry in raw
        entry isa AbstractDict || throw(ArgumentError("tool must be an object"))
        type=get(entry, "type", nothing)
        if type=="function"
            ingest_keys(entry, ("type", "function"), "tool")
            f=ingest_keys(
                get(entry, "function", nothing),
                ("name", "description", "parameters", "strict"),
                "function",
            )
            get(f, "strict", false)===true && unsupported("openai-chat", "strict function tools")
            strict=get(f, "strict", nothing)
            strict===nothing ||
                strict isa Bool ||
                throw(ArgumentError("tool strict must be boolean"))
            parameters=get(f, "parameters", nothing)
            parameters===nothing && (parameters=obj("type"=>"object", "properties"=>obj()))
            push!(
                tools,
                FunctionTool(;
                    name=ingest_string(get(f, "name", nothing), "tool name"),
                    description=get(f, "description", nothing),
                    parameters,
                ),
            )
        elseif c.builtin_tools=="groq" && haskey(inverse, type)
            push!(
                tools,
                BuiltinTool(;
                    name=inverse[type], config=obj((k=>v for (k, v) in entry if k!="type")...)
                ),
            )
        else
            unsupported("openai-chat", "tool type $type")
        end
    end
    return Tuple(tools)
end
function ingest_choice(raw, parallel)
    raw===nothing && parallel===nothing && return nothing
    mode="auto"
    allowed=String[]
    if raw isa AbstractString
        raw in ("auto", "none", "required") || throw(ArgumentError("unknown tool choice"))
        mode=String(raw)
    elseif raw isa AbstractDict
        type=get(raw, "type", nothing)
        if type=="function"
            ingest_keys(raw, ("type", "function"), "tool_choice")
            f=ingest_keys(get(raw, "function", nothing), ("name",), "tool_choice.function")
            mode="required"
            push!(allowed, ingest_string(get(f, "name", nothing), "forced tool name"))
        elseif type=="allowed_tools"
            ingest_keys(raw, ("type", "allowed_tools"), "tool_choice")
            spec=ingest_keys(get(raw, "allowed_tools", nothing), ("mode", "tools"), "allowed_tools")
            mode=ingest_string(get(spec, "mode", nothing), "allowed_tools.mode")
            entries=get(spec, "tools", nothing)
            entries isa AbstractVector && !isempty(entries) ||
                throw(ArgumentError("allowed tools must be non-empty"))
            for entry in entries
                ingest_keys(entry, ("type", "function"), "allowed tool")
                get(entry, "type", nothing)=="function" ||
                    unsupported("openai-chat", "builtin tool allowlist")
                f=ingest_keys(get(entry, "function", nothing), ("name",), "allowed tool function")
                push!(allowed, ingest_string(get(f, "name", nothing), "allowed tool name"))
            end
        else
            unsupported("openai-chat", "tool choice type $type")
        end
    elseif raw!==nothing
        throw(ArgumentError("tool_choice must be a string or object"))
    end
    return ToolChoice(; mode, allowed=Tuple(allowed), parallel)
end
function ingest_format(raw)
    raw isa AbstractDict || throw(ArgumentError("response_format must be an object"))
    type=get(raw, "type", nothing)
    if type in ("text", "json_object")
        ingest_keys(raw, ("type",), "response_format")
        return type=="text" ? nothing : obj("type"=>"json_object")
    end
    type=="json_schema" || throw(ArgumentError("unknown response format"))
    ingest_keys(raw, ("type", "json_schema"), "response_format")
    spec=ingest_keys(
        get(raw, "json_schema", nothing), ("name", "schema", "strict", "description"), "json_schema"
    )
    get(spec, "description", nothing)===nothing || unsupported("openai-chat", "schema description")
    out=obj("type"=>"json_schema", "schema"=>get(spec, "schema", nothing))
    name=get(spec, "name", nothing)
    name in (nothing, "response") || (out["name"]=name)
    get(spec, "strict", nothing)===nothing || (out["strict"]=spec["strict"])
    return out
end
function ingest_reasoning(body, c)
    keys=(
        "reasoning_effort",
        "reasoning",
        "thinking",
        "enable_thinking",
        "chat_template_kwargs",
        "reasoning_format",
    )
    spellings=Dict(
        "reasoning_effort"=>("reasoning_effort",),
        "openrouter"=>("reasoning",),
        "deepseek"=>("thinking", "reasoning_effort"),
        "kimi"=>("thinking", "reasoning_effort"),
        "qwen"=>("enable_thinking",),
        "qwen_chat_template"=>("chat_template_kwargs",),
        "none"=>(),
    )
    allowed=spellings[c.thinking_format]
    c.builtin_tools=="groq" && (allowed=(allowed..., "reasoning_format"))
    for key in keys
        haskey(body, key) &&
            !(key in allowed) &&
            unsupported("openai-chat", "reasoning spelling $key for this preset")
    end
    word=get(body, "reasoning_effort", nothing)
    off=word=="none"
    off && (word=nothing)
    extra=obj()
    summary=nothing
    if haskey(body, "thinking")
        spec=ingest_keys(body["thinking"], ("type",), "thinking")
        type=get(spec, "type", nothing)
        type in ("enabled", "disabled") || throw(ArgumentError("unknown thinking type"))
        if type=="disabled"
            word===nothing || throw(ArgumentError("reasoning level contradicts thinking disabled"))
            off=true
        elseif word===nothing && !off
            unsupported("openai-chat", "enabled thinking without an effort level")
        end
    end
    if haskey(body, "reasoning")
        spec=ingest_keys(body["reasoning"], ("effort", "enabled"), "reasoning")
        if get(spec, "enabled", nothing)===false
            off=true
        elseif get(spec, "effort", nothing)!==nothing
            word=spec["effort"]
        else
            throw(ArgumentError("reasoning needs effort or enabled=false"))
        end
    end
    if haskey(body, "enable_thinking") || haskey(body, "chat_template_kwargs")
        flag=if haskey(body, "chat_template_kwargs")
            spec=ingest_keys(
                body["chat_template_kwargs"],
                ("enable_thinking", "preserve_thinking"),
                "chat_template_kwargs",
            )
            get(spec, "enable_thinking", nothing)
        else
            body["enable_thinking"]
        end
        flag isa Bool || throw(ArgumentError("enable_thinking must be a boolean"))
        flag && unsupported("openai-chat", "thinking enabled without a canonical effort level")
        off=true
    end
    if haskey(body, "reasoning_format")
        body["reasoning_format"]=="parsed" ||
            unsupported("openai-chat", "reasoning format other than parsed")
        word===nothing ? (extra["reasoning_format"]="parsed") : (summary="auto")
    end
    return if off
        Reasoning(; effort="off")
    elseif word===nothing
        nothing
    else
        Reasoning(; effort=word, summary)
    end,
    extra
end
function ingest_cache(body, c, stable, boundary)
    keys=("prompt_cache_key", "prompt_cache_retention", "prompt_cache_options")
    marked=stable || boundary!==nothing
    any(k->haskey(body, k), keys) || marked || return nothing
    c.cache_control in ("openai", "openai_implicit") ||
        unsupported("openai-chat", "cache settings on this preset")
    marked &&
        c.cache_control!="openai" &&
        unsupported("openai-chat", "explicit cache mark on this preset")
    retention=nothing
    explicit=false
    key=get(body, "prompt_cache_key", nothing)
    if haskey(body, "prompt_cache_retention")
        body["prompt_cache_retention"]=="24h" ||
            unsupported("openai-chat", "cache retention other than 24h")
        retention="long"
    end
    if haskey(body, "prompt_cache_options")
        spec=ingest_keys(body["prompt_cache_options"], ("mode", "ttl"), "prompt_cache_options")
        get(spec, "ttl", nothing)===nothing || unsupported("openai-chat", "cache TTL")
        mode=get(spec, "mode", nothing)
        mode=="implicit" &&
            unsupported("openai-chat", "implicit cache mode has no canonical control")
        mode=="explicit" || throw(ArgumentError("unknown prompt_cache_options mode"))
        explicit=true
    end
    if explicit && !marked
        key===nothing && retention===nothing ||
            throw(ArgumentError("cache off cannot have a key or retention"))
        return CacheConfig(; mode="off")
    end
    return CacheConfig(;
        prefix=stable ? "stable" : nothing, prefix_until_index=boundary, key, retention
    )
end
function request_from_openai_chat(body::AbstractDict; compat=nothing)
    partial=if compat===nothing
        OpenAIChatCompat()
    elseif compat isa AbstractString
        preset(OpenAIChatCompat, compat)
    else
        compat
    end
    partial isa OpenAIChatCompat || throw(ArgumentError("compat must describe Chat Completions"))
    model=ingest_string(get(body, "model", nothing), "model")
    c=resolved_compat(partial, model)
    for key in keys(body)
        key in INGEST_REFUSED_KEYS && unsupported("openai-chat", "request key $key")
        key in (INGEST_CONFIG_KEYS..., INGEST_EXTENSION_KEYS..., "stream", "stream_options") ||
            unsupported("openai-chat", "unrecognized request key $key")
    end
    system, messages, stable, boundary=ingest_messages(get(body, "messages", nothing))
    # The deprecated functions / function_call shape is a spelling of tools /
    # tool_choice, and is translated (MAP-13: a spelling change is never a refusal).
    haskey(body, "functions") && haskey(body, "tools") && throw(ArgumentError("functions and tools cannot both be given"))
    haskey(body, "function_call") && haskey(body, "tool_choice") && throw(ArgumentError("function_call and tool_choice cannot both be given"))
    raw_tools=get(body, "tools", nothing)
    if haskey(body, "functions")
        body["functions"] isa AbstractVector || throw(ArgumentError("functions must be an array"))
        raw_tools=[obj("type"=>"function", "function"=>fn) for fn in body["functions"]]
    end
    raw_choice=get(body, "tool_choice", nothing)
    if haskey(body, "function_call")
        fc=body["function_call"]
        raw_choice=if fc in ("none", "auto")
            fc
        elseif fc isa AbstractDict && haskey(fc, "name")
            obj("type"=>"function", "function"=>obj("name"=>fc["name"]))
        else
            throw(ArgumentError("function_call must be 'none', 'auto', or an object with a name"))
        end
    end
    tools=ingest_tools(raw_tools, c)
    kw=Dict{Symbol,Any}()
    if haskey(body, "max_tokens") && haskey(body, "max_completion_tokens")
        body["max_tokens"]==body["max_completion_tokens"] ||
            throw(ArgumentError("max token spellings disagree"))
    end
    kw[:max_tokens]=get(body, "max_completion_tokens", get(body, "max_tokens", nothing))
    for key in ("temperature", "top_p", "top_k", "seed", "frequency_penalty", "presence_penalty", "service_tier", "store", "stop")
        haskey(body, key) && (kw[Symbol(key)]=body[key])
    end
    logprobs=get(body, "logprobs", nothing)
    if logprobs===true
        kw[:logprobs]=get(body, "top_logprobs", 0)
    else
        (logprobs===nothing || logprobs===false) || throw(ArgumentError("logprobs must be boolean"))
        haskey(body, "top_logprobs") && throw(ArgumentError("top_logprobs needs logprobs=true"))
    end
    haskey(body, "response_format") && (kw[:response_format]=ingest_format(body["response_format"]))
    kw[:tool_choice]=ingest_choice(raw_choice, get(body, "parallel_tool_calls", nothing))
    userkeys=[k for k in ("user", "safety_identifier", "user_id") if haskey(body, k)]
    length(userkeys)<=1 || throw(ArgumentError("multiple end-user identifiers"))
    "user_id" in userkeys &&
        c.user_field!="user_id" &&
        unsupported("openai-chat", "user_id spelling on this preset")
    isempty(userkeys) || (kw[:user_id]=body[only(userkeys)])
    reasoning, ext=ingest_reasoning(body, c)
    kw[:reasoning]=reasoning
    for key in INGEST_EXTENSION_KEYS
        # seed and the two penalties are canonical Config fields (promoted 2026-09-14).
        key in ("seed", "frequency_penalty", "presence_penalty") && continue
        haskey(body, key) && (ext[key]=body[key])
    end
    kw[:extensions]=ext
    kw[:cache]=ingest_cache(body, c, stable, boundary)
    return Request(model, messages; system, tools, config=Config(; kw...))
end
function request_from_openai_chat(l::ProviderLM, body::AbstractDict)
    l.compat isa OpenAIChatCompat ||
        unsupported(l.provider, "Chat Completions request ingest on this dialect")
    return request_from_openai_chat(body; compat=l.compat)
end
const LITELLM_PROVIDER_PREFIXES=Dict{String,String}(ROUTING_DATA["LITELLM_PROVIDER_PREFIXES"])
function openai_chat_model_string(model::AbstractString)
    occursin(':', model) && return String(model)
    bits=split(model, '/'; limit=2)
    length(bits)==1 && return String(model)
    haskey(LITELLM_PROVIDER_PREFIXES, bits[1]) || throw(
        UnknownModelError(
            "unknown or ambiguous foreign provider prefix; use provider:model";
            model=String(model),
        ),
    )
    return LITELLM_PROVIDER_PREFIXES[bits[1]]*":"*bits[2]
end
function resolve_openai_chat(router::LMRouter, model)
    resolution=resolve(router, openai_chat_model_string(model))
    return if resolution.source=="rule" && resolution.provider=="openai"
        resolve(router, "openai-chat:"*resolution.model)
    else
        resolution
    end
end
function request_from_openai_chat(router::LMRouter, model, messages; kwargs...)
    for (key, _) in kwargs
        key in (
            :api_key,
            :api_base,
            :base_url,
            :timeout,
            :num_retries,
            :max_retries,
            :headers,
            :extra_headers,
            :extra_body,
            :extra_query,
            :cache,
            :caching,
            :mock_response,
            :drop_params,
            :custom_llm_provider,
        ) && throw(
            NotConfiguredError(
                "$key configures the client, not the request; use RouterConfig or an explicit Request",
            ),
        )
    end
    resolution=resolve_openai_chat(router, model)
    client=lm(router, "$(resolution.provider):$(resolution.model)")
    body=obj(
        "model"=>resolution.model, "messages"=>messages, (string(k)=>v for (k, v) in kwargs)...
    )
    request=request_from_openai_chat(
        body; compat=client.compat isa OpenAIChatCompat ? client.compat : nothing
    )
    return request, client
end
function complete_from_openai_chat(router::LMRouter, model, messages; kwargs...)
    request, client=request_from_openai_chat(router, model, messages; kwargs...)
    return complete(client, request)
end
function stream_from_openai_chat(router::LMRouter, model, messages; kwargs...)
    request, client=request_from_openai_chat(router, model, messages; kwargs...)
    return stream(client, request)
end
