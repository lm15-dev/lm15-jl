const OPENAI_BUILTINS=Dict(
    "web_search"=>"web_search_preview",
    "code_execution"=>"code_interpreter",
    "file_search"=>"file_search",
    "computer_use"=>"computer_use_preview",
)
const GROQ_BUILTINS=Dict("web_search"=>"browser_search", "code_execution"=>"code_interpreter")
const ANTHROPIC_BUILTINS=Dict(
    "web_search"=>"web_search_20250305", "code_execution"=>"code_execution_20250522"
)
const GEMINI_BUILTINS=Dict("web_search"=>"googleSearch", "code_execution"=>"codeExecution")
function openai_input(p::Part, provider)
    p isa TextPart && return obj("type"=>"input_text", "text"=>p.text)
    if p isa ImagePart
        p.file_id!==nothing && return obj("type"=>"input_image", "file_id"=>p.file_id)
        d=obj("type"=>"input_image", "image_url"=>p.url===nothing ? media_uri(p) : p.url)
        p.detail===nothing || (d["detail"]=p.detail)
        return d
    elseif p isa AudioPart
        p.url!==nothing && return obj("type"=>"input_audio", "audio_url"=>p.url)
        p.file_id!==nothing && return obj("type"=>"input_audio", "file_id"=>p.file_id)
        format=last(split(p.media_type, '/'))
        format in ("mpeg", "mp3") && (format="mp3")
        return obj("type"=>"input_audio", "audio"=>media_base64(p), "format"=>format)
    elseif p isa Union{DocumentPart,BinaryPart}
        p.url!==nothing && return obj("type"=>"input_file", "file_url"=>p.url)
        p.file_id!==nothing && return obj("type"=>"input_file", "file_id"=>p.file_id)
        ext=first(split(last(split(p.media_type, '/')), '+'))
        return obj("type"=>"input_file", "filename"=>"file.$ext", "file_data"=>media_uri(p))
    elseif p isa VideoPart
        p.url!==nothing && return obj("type"=>"input_video", "video_url"=>p.url)
        p.file_id!==nothing && return obj("type"=>"input_video", "file_id"=>p.file_id)
        return obj("type"=>"input_video", "video_data"=>media_uri(p))
    elseif p isa Union{CitationPart,ThinkingPart}
        return obj("type"=>"input_text", "text"=>parts_to_text((p,); provider))
    end
    return unsupported(provider, "$(kind(p)) in Responses input")
end
function chat_image(p::ImagePart, provider)
    p.file_id===nothing || unsupported(provider, "file-id image on Chat Completions")
    d=obj("url"=>p.url===nothing ? media_uri(p) : p.url)
    p.detail===nothing || (d["detail"]=p.detail)
    return obj("type"=>"image_url", "image_url"=>d)
end
function chat_content(parts, provider; force_array=false)
    length(parts)==1 && first(parts) isa TextPart && !force_array && return first(parts).text
    out=Any[]
    for p in parts
        if p isa TextPart
            push!(out, obj("type"=>"text", "text"=>p.text))
        elseif p isa ImagePart
            push!(out, chat_image(p, provider))
        else
            unsupported(provider, "$(kind(p)) on Chat Completions input")
        end
    end
    return out
end
function tool_output(p, c, provider; chat=false)
    check_result_media(provider, p, c.tool_result_media)
    if all(x->!(x isa MediaPart), p.content)
        return error_text(p, parts_to_text(p.content; provider))
    end
    blocks=Any[]
    for part in p.content
        push!(
            blocks,
            if chat
                (
                    if part isa ImagePart
                        chat_image(part, provider)
                    else
                        obj("type"=>"text", "text"=>parts_to_text((part,); provider))
                    end
                )
            else
                openai_input(part, provider)
            end,
        )
    end
    p.is_error && mark_error!(blocks, chat ? "text" : "input_text", "text")
    return blocks
end
function breakpoint_index(request, control)
    cache=request.config.cache
    return if cache===nothing ||
        cache.mode=="off" ||
        control!="openai" ||
        cache.prefix_until_index===nothing
        nothing
    else
        min(cache.prefix_until_index, length(request.messages)-1)
    end
end
function stable_prefix(request, control)
    return request.config.cache!==nothing &&
           request.config.cache.mode!="off" &&
           request.config.cache.prefix=="stable" &&
           control=="openai"
end
function openai_cache!(payload, r, c, provider)
    cache=r.config.cache
    cache===nothing && return nothing
    cache.resource===nothing || unsupported(provider, "stored cache resource")
    c.cache_control in ("openai", "openai_implicit") || return nothing
    if cache.mode=="off"
        c.cache_control=="openai" &&
            has_cache_options(r.model) &&
            (payload["prompt_cache_options"]=obj("mode"=>"explicit"))
        return nothing
    end
    cache.key===nothing || (payload["prompt_cache_key"]=cache.key)
    cache.retention=="long" && (payload["prompt_cache_retention"]="24h")
    if c.cache_control=="openai" &&
        has_cache_options(r.model) &&
        (
            breakpoint_index(r, c.cache_control)!==nothing ||
            (stable_prefix(r, c.cache_control) && r.system!==nothing)
        )
        payload["prompt_cache_options"]=obj("mode"=>"explicit")
    end
end
function openai_messages(l, r, c; chat=false)
    out=Any[]
    boundary=breakpoint_index(r, c.cache_control)
    for (pos, msg) in enumerate(r.messages)
        marked=pos-1==boundary
        marked &&
            msg.role in ("assistant", "tool") &&
            unsupported(l.provider, "cache breakpoint on $(msg.role) message")
        if msg.role=="tool"
            for part in msg.parts
                entry=if chat
                    obj(
                        "role"=>"tool",
                        "tool_call_id"=>part.id,
                        "content"=>tool_output(part, c, l.provider; chat=true),
                    )
                else
                    obj(
                        "type"=>"function_call_output",
                        "call_id"=>part.id,
                        "output"=>tool_output(part, c, l.provider),
                    )
                end
                c.tool_result_name=="include" && part.name!==nothing && (entry["name"]=part.name)
                push!(out, entry)
            end
            continue
        end
        if msg.role=="assistant"
            content=Any[]
            calls=Any[]
            reasoning=String[]
            for part in msg.parts
                if part isa ToolCallPart
                    push!(
                        calls,
                        if chat
                            obj(
                                "id"=>part.id,
                                "type"=>"function",
                                "function"=>obj(
                                    "name"=>part.name, "arguments"=>JSON.serialize(part.input)
                                ),
                            )
                        else
                            obj(
                                "type"=>"function_call",
                                "call_id"=>part.id,
                                "name"=>part.name,
                                "arguments"=>JSON.serialize(part.input),
                            )
                        end,
                    )
                elseif part isa TextPart
                    push!(content, chat ? part.text : obj("type"=>"output_text", "text"=>part.text))
                elseif part isa RefusalPart
                    push!(content, chat ? part.text : obj("type"=>"refusal", "refusal"=>part.text))
                elseif part isa ThinkingPart
                    if chat
                        c.thinking_replay=="native" && push!(reasoning, part.text)
                        c.thinking_replay=="as_text" &&
                            !isempty(part.text) &&
                            push!(content, part.text)
                    else
                        state=continuation_data(part, "openai", "reasoning_item")
                        if state!==nothing && !isempty(state)
                            item=obj("type"=>"reasoning")
                            for key in ("id", "encrypted_content")
                                haskey(state, key) && (item[key]=state[key])
                            end
                            item["summary"]=if isempty(part.text)
                                Any[]
                            else
                                [obj("type"=>"summary_text", "text"=>part.text)]
                            end
                            push!(out, item)
                        elseif !isempty(part.text)
                            push!(content, obj("type"=>"output_text", "text"=>part.text))
                        end
                    end
                elseif part isa CitationPart
                    unsupported(
                        l.provider, "assistant citation replay has no native block on this wire"
                    )
                else
                    unsupported(l.provider, "assistant $(kind(part)) replay")
                end
            end
            if chat
                item=obj(
                    "role"=>"assistant", "content"=>isempty(content) ? nothing : join(content, "\n")
                )
                isempty(calls) || (item["tool_calls"]=calls)
                if c.thinking_replay=="native" &&
                    (!isempty(reasoning) || c.assistant_reasoning_content=="include_empty")
                    item["reasoning_content"]=join(filter(!isempty, reasoning), "\n")
                end
                push!(out, item)
            else
                if !isempty(content)
                    item=obj("role"=>"assistant", "content"=>content)
                    c.commentary_phase=="tag" && !isempty(calls) && (item["phase"]="commentary")
                    push!(out, item)
                end
                append!(out, calls)
            end
            continue
        end
        role=msg.role=="developer" ? (chat ? c.instruction_role : c.developer_role) : msg.role
        content=if chat
            chat_content(msg.parts, l.provider; force_array=marked)
        else
            [openai_input(p, l.provider) for p in msg.parts]
        end
        if marked
            (
                content isa AbstractVector &&
                !isempty(content) &&
                get(last(content), "type", nothing)==(chat ? "text" : "input_text")
            ) || unsupported(l.provider, "cache breakpoint must be on the final text block")
            last(content)["prompt_cache_breakpoint"]=obj("mode"=>"explicit")
        end
        push!(out, obj("role"=>role, "content"=>content))
    end
    return out
end
function openai_builtin(t, c; chat=false, provider="openai")
    if chat
        c.builtin_tools=="groq" && haskey(GROQ_BUILTINS, t.name) ||
            unsupported(provider, "builtin tool $(t.name) on Chat Completions")
        wire=GROQ_BUILTINS[t.name]
    else
        wire=c.builtin_tools=="verbatim" ? t.name : get(OPENAI_BUILTINS, t.name, t.name)
    end
    return merge(obj("type"=>wire), something(t.config, obj()))
end
function openai_choice(r, c, provider; chat=false)
    tc=r.config.tool_choice
    tc===nothing && return nothing
    isempty(tc.allowed) && return tc.mode
    named=Dict(t.name=>t for t in r.tools)
    entries=Any[]
    for name in tc.allowed
        tool=named[name]
        if tool isa BuiltinTool
            chat && unsupported(provider, "builtin tool forcing on Chat Completions")
            push!(
                entries,
                obj("type"=>c.builtin_tools=="verbatim" ? name : get(OPENAI_BUILTINS, name, name)),
            )
        else
            push!(
                entries,
                if chat
                    obj("type"=>"function", "function"=>obj("name"=>name))
                else
                    obj("type"=>"function", "name"=>name)
                end,
            )
        end
    end
    length(entries)==1 && tc.mode=="required" && return only(entries)
    return if chat
        obj("type"=>"allowed_tools", "allowed_tools"=>obj("mode"=>tc.mode, "tools"=>entries))
    else
        obj("type"=>"allowed_tools", "mode"=>tc.mode, "tools"=>entries)
    end
end
function structured_openai(f; chat=false)
    f["type"]=="json_object" &&
        return chat ? obj("type"=>"json_object") : obj("format"=>obj("type"=>"json_object"))
    inner=obj("name"=>get(f, "name", "response"), "schema"=>f["schema"])
    haskey(f, "strict") && (inner["strict"]=f["strict"])
    return if chat
        obj("type"=>"json_schema", "json_schema"=>inner)
    else
        obj("format"=>merge(obj("type"=>"json_schema"), inner))
    end
end
function openai_reasoning!(d, reasoning, c, l; chat=false)
    reasoning===nothing && return nothing
    format=chat ? c.thinking_format : c.reasoning_format
    format=="none" && unsupported(l.provider, "reasoning dial on a wire with no reasoning field")
    reasoning.thinking_budget===nothing || unsupported(l.provider, "thinking token budget")
    reasoning.summary in ("concise", "detailed") &&
        format!="responses_reasoning" &&
        unsupported(l.provider, "reasoning summary detail")
    off=reasoning.effort=="off"
    word=off ? "none" : reasoning.effort
    if chat && !off && c.reasoning_efforts!==nothing
        word in c.reasoning_efforts || unsupported(l.provider, "reasoning effort $word")
    end
    if format=="responses_reasoning"
        d["reasoning"]=obj("effort"=>word)
        reasoning.summary===nothing || (d["reasoning"]["summary"]=reasoning.summary)
    elseif format in ("reasoning_effort", "kimi")
        if format=="kimi" && off
            (d["thinking"]=obj("type"=>"disabled"))
        else
            (d["reasoning_effort"]=word)
        end
    elseif format=="openrouter"
        d["reasoning"]=off ? obj("enabled"=>false) : obj("effort"=>word)
    elseif format=="deepseek"
        d["thinking"]=obj("type"=>off ? "disabled" : "enabled")
        off || (d["reasoning_effort"]=word)
    elseif format in ("qwen", "zai")
        d["enable_thinking"]=!off
    elseif format=="qwen_chat_template"
        d["chat_template_kwargs"]=if off
            obj("enable_thinking"=>false)
        else
            obj("enable_thinking"=>true, "preserve_thinking"=>true)
        end
    else
        unsupported(l.provider, "unknown reasoning mapping")
    end
    return chat &&
           c.builtin_tools=="groq" &&
           reasoning.summary=="auto" &&
           (d["reasoning_format"]="parsed")
end
function openai_payload(l, r; stream=false, chat=false)
    c=effective_compat(l, r)
    config=r.config
    if l.provider=="xai"
        config.reasoning!==nothing &&
            config.reasoning.effort=="off" &&
            unsupported(l.provider, "reasoning off")
        config.logprobs===nothing || unsupported(l.provider, "logprobs")
        tc=config.tool_choice
        tc===nothing ||
            isempty(tc.allowed) ||
            (length(tc.allowed)==1 && tc.mode=="required") ||
            unsupported(l.provider, "tool allowlists")
        tc!==nothing &&
            tc.mode=="required" &&
            config.response_format!==nothing &&
            unsupported(l.provider, "forced tools with structured output")
    end
    rows=openai_messages(l, r, c; chat)
    d=obj("model"=>r.model, (chat ? "messages" : "input")=>rows)
    if !chat || stream
        d["stream"]=stream
    end
    chat && stream && c.stream_usage=="include" && (d["stream_options"]=obj("include_usage"=>true))
    if r.system!==nothing
        content=system_text(r.system; provider=l.provider)
        stable=stable_prefix(r, c.cache_control)
        if chat || stable
            block=if stable
                [
                    obj(
                        "type"=>chat ? "text" : "input_text",
                        "text"=>content,
                        "prompt_cache_breakpoint"=>obj("mode"=>"explicit"),
                    ),
                ]
            else
                content
            end
            pushfirst!(
                rows, obj("role"=>chat ? c.instruction_role : c.developer_role, "content"=>block)
            )
        else
            d["instructions"]=content
        end
    end
    config.max_tokens===nothing ||
        (d[chat ? c.max_tokens_field : c.max_output_tokens_field]=config.max_tokens)
    for key in (:temperature, :top_p, :service_tier, :store)
        value=getfield(config, key)
        value===nothing || (d[string(key)]=value)
    end
    config.user_id===nothing || (d[chat ? c.user_field : "safety_identifier"]=config.user_id)
    config.top_k===nothing || unsupported(l.provider, "top_k")
    if !isempty(config.stop)
        chat || unsupported(l.provider, "stop sequences on Responses")
        d["stop"]=collect(config.stop)
    end
    if config.logprobs!==nothing
        if chat
            d["logprobs"]=true
            config.logprobs>0 && (d["top_logprobs"]=config.logprobs)
        else
            d["top_logprobs"]=config.logprobs
            d["include"]=["message.output_text.logprobs"]
        end
    end
    if !isempty(r.tools)
        tools=Any[]
        for t in r.tools
            if t isa FunctionTool
                f=obj("name"=>t.name, "description"=>t.description, "parameters"=>t.parameters)
                c.strict_tools=="include" && (f["strict"]=false)
                push!(
                    tools,
                    if chat
                        obj("type"=>"function", "function"=>f)
                    else
                        merge(obj("type"=>"function"), f)
                    end,
                )
            else
                push!(tools, openai_builtin(t, c; chat, provider=l.provider))
            end
        end
        d["tools"]=tools
    end
    tc=config.tool_choice
    if tc!==nothing
        chat &&
            c.forced_tool_choice=="reject" &&
            (tc.mode!="auto" || !isempty(tc.allowed)) &&
            unsupported(l.provider, "forced tool choice")
        d["tool_choice"]=openai_choice(r, c, l.provider; chat)
        tc.parallel===nothing || (d["parallel_tool_calls"]=tc.parallel)
    end
    if config.response_format!==nothing
        chat &&
            c.json_schema=="reject" &&
            config.response_format["type"]=="json_schema" &&
            unsupported(l.provider, "JSON schema enforcement")
        d[chat ? "response_format" : "text"]=structured_openai(config.response_format; chat)
    end
    openai_reasoning!(d, config.reasoning, c, l; chat)
    openai_cache!(d, r, c, l.provider)
    c.routing===nothing || (d["provider"]=c.routing)
    for (key, value) in something(config.extensions, obj())
        key in (
            "prompt_caching",
            "cache",
            "compat",
            "openai_compat",
            chat ? "openai_chat_compat" : "openai_responses_compat",
        ) || (d[key]=value)
    end
    if l.access.backend=="chatgpt-codex"
        # An explicit cap or store=true is refused, never stripped: dropping a cap
        # means unbounded spend (MAP-13 rule 4; the other SDKs refuse the same).
        r.config.max_tokens===nothing || throw(UnsupportedFeatureError(
            "$(l.provider): config.max_tokens: this backend has no output cap; dropping it risks unbounded paid generation";
            provider=l.provider))
        r.config.store===true && throw(UnsupportedFeatureError(
            "$(l.provider): config.store: this backend cannot store a retrievable response; the program may depend on retrieval";
            provider=l.provider))
        get!(d, "instructions", something(l.access.system_prefix, "You are a helpful assistant."))
        d["store"]=false
        d["stream"]=true
        for key in ("max_output_tokens", "max_completion_tokens", "max_tokens")
            delete!(d, key)
        end
    end
    return d
end

function anthropic_source(p::MediaPart)
    p.url!==nothing && return obj("type"=>"url", "url"=>p.url)
    p.file_id!==nothing && return obj("type"=>"file", "file_id"=>p.file_id)
    return obj("type"=>"base64", "media_type"=>p.media_type, "data"=>media_base64(p))
end
function anthropic_part(l, p, c)
    p isa TextPart && return obj("type"=>"text", "text"=>p.text)
    p isa Union{ImagePart,DocumentPart} &&
        return obj("type"=>kind(p), "source"=>anthropic_source(p))
    p isa MediaPart && unsupported(l.provider, "$(kind(p)) on Messages")
    p isa ToolCallPart &&
        return obj("type"=>"tool_use", "id"=>p.id, "name"=>p.name, "input"=>p.input)
    if p isa ToolResultPart
        check_result_media(l.provider, p, c.tool_result_media)
        blocks=[anthropic_part(l, part, c) for part in p.content]
        content=length(blocks)==1 && first(blocks)["type"]=="text" ? first(blocks)["text"] : blocks
        d=obj("type"=>"tool_result", "tool_use_id"=>p.id, "content"=>content)
        p.is_error && (d["is_error"]=true)
        return d
    elseif p isa ThinkingPart
        redacted=continuation_data(p, "anthropic", "redacted_thinking")
        redacted===nothing || return merge(obj("type"=>"redacted_thinking"), redacted)
        signed=continuation_data(p, "anthropic", "thinking_signature")
        if signed!==nothing && !empty_optional(get(signed, "signature", nothing))
            return obj("type"=>"thinking", "thinking"=>p.text, "signature"=>signed["signature"])
        elseif c.thinking_replay=="unsigned" && !isempty(p.text)
            return obj("type"=>"thinking", "thinking"=>p.text)
        end
        return obj("type"=>"text", "text"=>p.text)
    end
    return obj("type"=>"text", "text"=>parts_to_text((p,); provider=l.provider))
end
function anthropic_payload(l, r; stream=false)
    c=effective_compat(l, r)
    config=r.config
    c.model_prefixes===nothing ||
        any(p->startswith(r.model, p), c.model_prefixes) ||
        throw(UnsupportedModelError("model would be silently substituted"; provider=l.provider))
    messages=Any[]
    for m in r.messages
        blocks=if m.role=="developer"
            [
                obj(
                    "type"=>"text",
                    "text"=>"[developer]\n"*parts_to_text(m.parts; provider=l.provider),
                ),
            ]
        else
            [anthropic_part(l, p, c) for p in m.parts]
        end
        push!(messages, obj("role"=>m.role=="assistant" ? "assistant" : "user", "content"=>blocks))
    end
    reasoning=config.reasoning
    active=reasoning!==nothing && reasoning.effort!="off"
    adaptive=active && (
        c.thinking_format in ("deepseek", "adaptive", "effort") || anthropic_adaptive(r.model)
    )
    budget=if active && !adaptive
        something(reasoning.thinking_budget, EFFORT_BUDGETS[reasoning.effort])
    else
        nothing
    end
    if active
        reasoning.summary in ("concise", "detailed") &&
            unsupported(l.provider, "reasoning summary detail")
        c.reasoning_efforts===nothing ||
            reasoning.effort in c.reasoning_efforts ||
            unsupported(l.provider, "reasoning effort $(reasoning.effort)")
        adaptive &&
            reasoning.thinking_budget!==nothing &&
            unsupported(l.provider, "thinking budget on adaptive model")
        adaptive &&
            c.thinking_format=="anthropic" &&
            reasoning.effort=="minimal" &&
            unsupported(l.provider, "minimal adaptive reasoning")
    end
    d=obj(
        "model"=>r.model,
        "messages"=>messages,
        "stream"=>stream,
        "max_tokens"=>something(config.max_tokens, 1024)+something(budget, 0),
    )
    cache=config.cache
    usecache=cache!==nothing && cache.mode!="off" && c.cache_control=="anthropic"
    marker=obj("type"=>"ephemeral")
    cache!==nothing && cache.retention=="long" && (marker["ttl"]="1h")
    # The implicit-cache fallback applies to prefix marks, not references to a
    # stored object or an affinity key. None of the Messages bindings has a wire
    # slot for these intents, even when its compat disables cache_control.
    if cache !== nothing
        cache.key===nothing || unsupported(l.provider, "cache affinity key")
        cache.resource===nothing || unsupported(l.provider, "stored cache resource")
    end
    if usecache
        idx=if cache.prefix_until_index===nothing
            (cache.prefix=="history" ? length(messages)-1 : nothing)
        else
            min(cache.prefix_until_index, length(messages)-1)
        end
        idx===nothing || (last(messages[idx + 1]["content"])["cache_control"]=copy(marker))
    end
    if r.system!==nothing
        s=system_text(r.system; provider=l.provider)
        d["system"]=usecache ? [obj("type"=>"text", "text"=>s, "cache_control"=>copy(marker))] : s
    end
    for name in (:temperature, :top_p, :top_k)
        value=getfield(config, name)
        value===nothing && continue
        c.sampling_params=="reject" && unsupported(l.provider, "sampling setting $name")
        d[string(name)]=value
    end
    isempty(config.stop) || (d["stop_sequences"]=collect(config.stop))
    if !isempty(r.tools)
        d["tools"]=[
            if t isa FunctionTool
                obj("name"=>t.name, "description"=>t.description, "input_schema"=>t.parameters)
            else
                merge(
                    obj("type"=>get(ANTHROPIC_BUILTINS, t.name, t.name), "name"=>t.name),
                    something(t.config, obj()),
                )
            end for t in r.tools
        ]
    end
    tc=config.tool_choice
    if tc!==nothing
        choice=obj("type"=>tc.mode=="required" ? "any" : tc.mode)
        if !isempty(tc.allowed)
            if length(tc.allowed)==1 && tc.mode=="required"
                choice=obj("type"=>"tool", "name"=>only(tc.allowed))
            elseif Set(tc.allowed)!=Set(t.name for t in r.tools)
                unsupported(l.provider, "proper-subset tool allowlist")
            end
        end
        c.parallel_tool_calls=="reject" &&
            tc.parallel!==nothing &&
            unsupported(l.provider, "parallel tool setting")
        tc.parallel===false && tc.mode!="none" && (choice["disable_parallel_tool_use"]=true)
        d["tool_choice"]=choice
    end
    if reasoning!==nothing
        if reasoning.effort=="off"
            c.thinking_format!="anthropic" && (d["thinking"]=obj("type"=>"disabled"))
        elseif adaptive
            c.thinking_format=="effort" ||
                (d["thinking"]=obj("type"=>c.thinking_format=="deepseek" ? "enabled" : "adaptive"))
            d["output_config"]=obj("effort"=>reasoning.effort)
        else
            d["thinking"]=obj("type"=>"enabled", "budget_tokens"=>budget)
        end
    end
    if config.response_format!==nothing
        (c.structured_output!="reject" && config.response_format["type"]=="json_schema") ||
            unsupported(l.provider, "structured output shape")
        output=get!(d, "output_config", obj())
        output["format"]=obj("type"=>"json_schema", "schema"=>config.response_format["schema"])
    end
    config.store===nothing || unsupported(l.provider, "response storage setting")
    config.logprobs===nothing || unsupported(l.provider, "token log probabilities")
    config.service_tier===nothing || (d["service_tier"]=config.service_tier)
    config.user_id===nothing || (d["metadata"]=obj("user_id"=>config.user_id))
    for (key, value) in something(config.extensions, obj())
        key=="prompt_caching" || (d[key]=value)
    end
    if l.access.system_prefix!==nothing
        prefix=obj("type"=>"text", "text"=>l.access.system_prefix)
        existing=get(d, "system", nothing)
        d["system"]=if existing===nothing
            [prefix]
        elseif existing isa AbstractVector
            [prefix; existing]
        else
            [prefix, obj("type"=>"text", "text"=>existing)]
        end
    end
    return d
end

function gemini_part(l, p, names)
    if p isa MediaPart
        source=p.url===nothing ? p.file_id : p.url
        return if source===nothing
            obj("inlineData"=>obj("mimeType"=>p.media_type, "data"=>media_base64(p)))
        else
            obj("fileData"=>obj("mimeType"=>p.media_type, "fileUri"=>source))
        end
    elseif p isa ToolCallPart
        d=obj("functionCall"=>obj("name"=>p.name, "args"=>p.input, "id"=>p.id))
    elseif p isa ToolResultPart
        name=p.name===nothing ? get(names, p.id, nothing) : p.name
        name===nothing && unsupported(l.provider, "tool result without a known function name")
        visible=Tuple(v for v in p.content if !(v isa MediaPart))
        media=[v for v in p.content if v isa MediaPart]
        all(v->v isa Union{ImagePart,DocumentPart}, media) ||
            unsupported(l.provider, "this media kind in a function response")
        output=parts_to_text(visible; provider=l.provider)
        response=if p.is_error
            obj("error"=>output)
        elseif isempty(visible) && !isempty(media)
            obj()
        else
            obj("result"=>output)
        end
        fr=obj("name"=>name, "response"=>response, "id"=>p.id)
        isempty(media) || (fr["parts"]=[gemini_part(l, v, names) for v in media])
        return obj("functionResponse"=>fr)
    elseif p isa Union{TextPart,ThinkingPart}
        d=obj("text"=>p.text)
    else
        return obj("text"=>parts_to_text((p,); provider=l.provider))
    end
    state=continuation_data(p, "gemini", "thought_signature")
    if state!==nothing && !empty_optional(get(state, "value", nothing))
        d["thoughtSignature"]=state["value"]
        p isa ThinkingPart && (d["thought"]=true)
    end
    return d
end
function gemini_messages(l, messages; from=1)
    names=Dict{String,String}()
    out=Any[]
    for (index, m) in enumerate(messages)
        if index>=from
            parts=if m.role=="developer"
                [obj("text"=>"[developer]\n"*parts_to_text(m.parts; provider=l.provider))]
            else
                [gemini_part(l, p, names) for p in m.parts]
            end
            push!(out, obj("role"=>m.role=="assistant" ? "model" : "user", "parts"=>parts))
        end
        # Names come only from preceding turns, never future calls.
        for p in m.parts
            p isa ToolCallPart && (names[p.id]=p.name)
        end
    end
    return out
end
contains_key(d, k) =
    if d isa AbstractDict
        haskey(d, k) || any(v->contains_key(v, k), values(d))
    elseif d isa AbstractVector
        any(v->contains_key(v, k), d)
    else
        false
    end
function gemini_tools(tools)
    functions=[
        obj("name"=>t.name, "description"=>t.description, "parameters"=>t.parameters) for
        t in tools if t isa FunctionTool
    ]
    out=Any[]
    isempty(functions) || push!(out, obj("functionDeclarations"=>functions))
    append!(
        out,
        [
            obj(get(GEMINI_BUILTINS, t.name, t.name)=>something(t.config, obj())) for
            t in tools if t isa BuiltinTool
        ],
    )
    return out
end
function gemini_payload(l, r)
    config=r.config
    cache=config.cache
    resource=nothing
    from=1
    if cache!==nothing && cache.mode!="off"
        cache.key===nothing || unsupported(l.provider, "cache affinity key")
        cache.retention in (nothing, "short") ||
            unsupported(l.provider, "in-request cache retention")
        resource=cache.resource
        resource===nothing ||
            cache.prefix_until_index===nothing ||
            (from=min(cache.prefix_until_index, length(r.messages)-1)+2)
    end
    from>length(r.messages) && throw(ArgumentError("stored cache request needs a suffix message"))
    d=obj("contents"=>gemini_messages(l, r.messages; from))
    if resource!==nothing
        d["cachedContent"]=if startswith(resource, "cachedContents/")
            resource
        else
            "cachedContents/"*resource
        end
    elseif r.system!==nothing
        d["systemInstruction"]=obj(
            "parts"=>[obj("text"=>system_text(r.system; provider=l.provider))]
        )
    end
    gen=obj()
    for (field, wire) in (
        (:temperature, "temperature"),
        (:top_p, "topP"),
        (:top_k, "topK"),
        (:max_tokens, "maxOutputTokens"),
    )
        v=getfield(config, field)
        v===nothing || (gen[wire]=v isa AbstractFloat && isinteger(v) ? Int(v) : v)
    end
    isempty(config.stop) || (gen["stopSequences"]=collect(config.stop))
    if config.logprobs!==nothing
        gen["responseLogprobs"]=true
        config.logprobs>0 && (gen["logprobs"]=config.logprobs)
    end
    if config.response_format!==nothing
        f=config.response_format
        gen["responseMimeType"]="application/json"
        f["type"]=="json_schema" && (
            gen[contains_key(f["schema"], "additionalProperties") ? "responseJsonSchema" : "responseSchema"]=f["schema"]
        )
    end
    reasoning=config.reasoning
    if reasoning!==nothing
        thinking=obj()
        level=gemini_level(r.model)
        if reasoning.effort=="off"
            level && unsupported(l.provider, "reasoning off on Gemini 3")
            thinking["thinkingBudget"]=0
        else
            reasoning.summary in ("concise", "detailed") &&
                unsupported(l.provider, "reasoning summary detail")
            reasoning.summary===nothing || (thinking["includeThoughts"]=true)
            if reasoning.thinking_budget!==nothing
                thinking["thinkingBudget"]=reasoning.thinking_budget
            elseif level
                reasoning.effort in ("xhigh", "max") &&
                    unsupported(l.provider, "reasoning effort $(reasoning.effort) on Gemini 3")
                thinking["thinkingLevel"]=reasoning.effort
            else
                thinking["thinkingBudget"]=EFFORT_BUDGETS[reasoning.effort]
            end
        end
        gen["thinkingConfig"]=thinking
    end
    isempty(gen) || (d["generationConfig"]=gen)
    if resource===nothing
        isempty(r.tools) || (d["tools"]=gemini_tools(r.tools))
        tc=config.tool_choice
        if tc!==nothing
            tc.parallel===false && unsupported(l.provider, "parallel=false")
            byname=Dict(t.name=>t for t in r.tools)
            any(n->byname[n] isa BuiltinTool, tc.allowed) &&
                unsupported(l.provider, "builtin tool forcing")
            cfg=obj("mode"=>if tc.mode=="none"
                "NONE"
            elseif tc.mode=="required"
                "ANY"
            elseif isempty(tc.allowed)
                "AUTO"
            else
                "VALIDATED"
            end)
            isempty(tc.allowed) || (cfg["allowedFunctionNames"]=collect(tc.allowed))
            d["toolConfig"]=obj("functionCallingConfig"=>cfg)
        end
    elseif config.tool_choice!==nothing
        unsupported(l.provider, "tool choice alongside a stored cache resource")
    end
    config.user_id===nothing || unsupported(l.provider, "end-user attribution")
    config.store===nothing || (d["store"]=config.store)
    config.service_tier===nothing || (d["serviceTier"]=config.service_tier)
    ext=something(config.extensions, obj())
    output=get(ext, "output", nothing)
    output in ("image", "audio") &&
        (get!(d, "generationConfig", obj())["responseModalities"]=[uppercase(output)])
    for (k, v) in ext
        k in ("output", "prompt_caching") || (d[k]=v)
    end
    return d
end
function build_payload(l, r; stream=false)
    validate(r)
    l.dialect=="openai-responses" && return openai_payload(l, r; stream)
    l.dialect=="openai-chat" && return openai_payload(l, r; stream, chat=true)
    l.dialect=="anthropic" && return anthropic_payload(l, r; stream)
    return gemini_payload(l, r)
end
function build_request(l::ProviderLM, r::Request; stream=false)
    require_surface(l, stream ? :stream : :complete)
    payload=build_payload(l, r; stream)
    endpoint=if l.dialect=="openai-responses"
        "responses"
    elseif l.dialect=="openai-chat"
        "chat/completions"
    elseif l.dialect=="anthropic"
        "messages"
    else
        "generateContent"
    end
    params=obj()
    if l.dialect=="gemini"
        model=startswith(r.model, "models/") ? r.model : "models/"*r.model
        url=l.base_url*"/"*percent_encode(model; safe="/:@")*":"*(
            stream ? "streamGenerateContent" : "generateContent"
        )
        stream && (params["alt"]="sse")
    else
        url=l.base_url*"/"*endpoint
    end
    return emit(
        l; url, payload, params, headers=base_headers(l; request=r), endpoint, stream, model=r.model
    )
end
