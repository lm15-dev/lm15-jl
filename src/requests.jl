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
    p isa DataPart && return obj("type"=>"input_text", "text"=>data_part_text(p))
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
    length(parts)==1 && first(parts) isa DataPart && !force_array && return data_part_text(first(parts))
    out=Any[]
    for p in parts
        if p isa TextPart
            push!(out, obj("type"=>"text", "text"=>p.text))
        elseif p isa DataPart
            push!(out, obj("type"=>"text", "text"=>data_part_text(p)))
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
# MAP-13: the wire carries the mark on a text block of a user/developer message
# only; a mark asked elsewhere walks back to the nearest eligible message ("cache
# up to here"), else it is dropped and implicit caching still applies.
function adapted_breakpoint(request, control)
    asked=breakpoint_index(request, control)
    asked===nothing && return nothing
    for index in asked:-1:0
        msg=request.messages[index+1]
        (msg.role in ("assistant", "tool") || isempty(msg.parts) || !(last(msg.parts) isa TextPart)) && continue
        index==asked || adapt!(
            "config.cache.prefix_until_index", "substituted",
            "message $(asked) is a $(request.messages[asked+1].role) message or does not end with text; the Responses wire marks text blocks of user/developer messages only, so the mark moved to the nearest eligible message before it";
            asked, applied=index,
        )
        return index
    end
    adapt!(
        "config.cache.prefix_until_index", "dropped",
        "no user/developer message ending with text at or before message $(asked); the Responses wire marks text blocks only (implicit caching still applies)";
        asked,
    )
    return nothing
end
function stable_prefix(request, control)
    return request.config.cache!==nothing &&
           request.config.cache.mode!="off" &&
           request.config.cache.prefix=="stable" &&
           control=="openai"
end
# MAP-10: a message part a dialect has no content slot for in that role raises
# before any wire (changes/2026-09-24-message-media.md); `feature` is its path.
const NO_MESSAGE_SLOT=Dict(
    "anthropic"=>(role, kind)->kind in ("audio", "video", "binary"),
    "openai-responses"=>(role, kind)->role=="assistant",
    "openai-chat"=>(role, kind)->role=="assistant",
)
function check_message_media(l, r)
    gap=get(NO_MESSAGE_SLOT, l.dialect, nothing)
    gap===nothing && return nothing
    for (i, m) in enumerate(r.messages), (j, p) in enumerate(m.parts)
        p isa MediaPart && gap(m.role, kind(p)) && refuse(
            l.provider, "messages[$(i-1)].parts[$(j-1)]",
            "the program depends on this $(m.role) $(kind(p)) part; no native $(l.dialect) content slot carries it (MAP-10)",
        )
    end
end
function openai_cache!(payload, r, c, provider; boundary=breakpoint_index(r, c.cache_control))
    cache=r.config.cache
    cache===nothing && return nothing
    # MAP-6 rule 7: a stored-cache resource where there is no such tier raises;
    # dropping it would send the request without the prefix it holds.
    cache.resource===nothing || refuse(provider, "config.cache.resource", "this provider has no stored-cache tier; sending without it would drop the prompt prefix the resource holds")
    if !(c.cache_control in ("openai", "openai_implicit"))
        # MAP-13: the key and the lifetime have no home on this server.
        cache.key===nothing || adapt!("config.cache.key", "dropped", "this server has no cache affinity field; implicit caching still applies"; asked=cache.key)
        cache.retention=="long" && adapt!("config.cache.retention", "dropped", "this server has no in-request cache lifetime knob; implicit caching still applies"; asked="long")
        return nothing
    end
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
            boundary!==nothing ||
            (stable_prefix(r, c.cache_control) && r.system!==nothing)
        )
        payload["prompt_cache_options"]=obj("mode"=>"explicit")
    end
end
function openai_messages(l, r, c; chat=false, boundary=breakpoint_index(r, c.cache_control))
    out=Any[]
    for (pos, msg) in enumerate(r.messages)
        marked=pos-1==boundary
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
        marked && (last(content)["prompt_cache_breakpoint"]=obj("mode"=>"explicit"))
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
    if chat && format=="none"
        adapt!("config.reasoning", "dropped", "this server has no reasoning dial on its wire (compat thinking_format='none'); the model reasons at its own default; pass the server's own knob through extensions"; asked=obj("effort"=>reasoning.effort))
        return nothing
    end
    if chat && reasoning.effort=="off" && c.reasoning_off=="lowest"
        # The model cannot stop reasoning and this server accepts the off word and
        # reasons anyway (compat reasoning_off): send the lowest level and say so
        # (MAP-13 §4.2, xAI's rule).
        lowest=c.reasoning_efforts===nothing || isempty(c.reasoning_efforts) ? "low" : first(c.reasoning_efforts)
        adapt!("config.reasoning.effort", "substituted", "this model cannot stop reasoning and the server accepts 'none' and reasons anyway (a paid no-op); the lowest level was sent"; asked="off", applied=lowest)
        reasoning=reconstruct(reasoning; effort=lowest)
    end
    off=reasoning.effort=="off"
    word=off ? "none" : reasoning.effort
    summary=reasoning.summary
    if !off
        reasoning.thinking_budget===nothing || adapt!(
            "config.reasoning.thinking_budget", "dropped",
            chat ? "the Chat Completions wire has no thinking token budget; effort carries the intent" :
                   "this wire has no thinking token budget; effort carries the intent (Anthropic's manual class and Gemini take a budget)";
            asked=reasoning.thinking_budget,
        )
        if summary in ("concise", "detailed") && !(!chat && format=="responses_reasoning")
            adapt!(
                "config.reasoning.summary", "substituted",
                chat ? "the Chat Completions wire has no summary detail levels; 'auto' is what it shows" :
                       "this wire has no summary detail levels; 'auto' is what it shows";
                asked=summary, applied="auto",
            )
            chat && (summary="auto")
        end
        if chat && c.reasoning_efforts!==nothing && !(word in c.reasoning_efforts)
            nearest=nearest_effort(word, c.reasoning_efforts)
            adapt!(
                "config.reasoning.effort", "clamped",
                "this server has no '$(word)' level (it accepts $(join(c.reasoning_efforts, ", "))) and would have accepted the word silently";
                asked=word, applied=nearest,
            )
            word=nearest
        end
    end
    if format=="responses_reasoning"
        d["reasoning"]=obj("effort"=>word)
        !off && summary!==nothing && (d["reasoning"]["summary"]=summary)
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
    end
    return !off && chat &&
           c.builtin_tools=="groq" &&
           summary=="auto" &&
           (d["reasoning_format"]="parsed")
end
# xAI's own adaptations before the Chat Completions build (Python XaiLM._payload).
function xai_prepare(l, r)
    config=r.config
    if config.reasoning!==nothing && config.reasoning.effort=="off"
        adapt!("config.reasoning.effort", "substituted", "Grok reasoning models have no off switch and api.x.ai ignores disable fields (158 reasoning tokens on an explicit off, live 2026-09-01); the lowest level was sent"; asked="off", applied="low")
        config=reconstruct(config; reasoning=reconstruct(config.reasoning; effort="low"))
    end
    if config.logprobs!==nothing
        adapt!("config.logprobs", "dropped", "grok-4.20 and newer ignore logprobs/top_logprobs (docs.x.ai, live 2026-09-01); Response.logprobs will be absent (OpenAI and Gemini carry them)"; asked=config.logprobs)
        config=reconstruct(config; logprobs=nothing)
    end
    tools=r.tools
    tc=config.tool_choice
    if tc!==nothing && !isempty(tc.allowed) && !(length(tc.allowed)==1 && tc.mode=="required")
        tools=Tuple(t for t in r.tools if t.name in tc.allowed)
        adapt!("config.tool_choice.allowed", "client_side", "api.x.ai ignores tool_choice allowlists (live 2026-09-02); only the allowed tools were sent, which is what the allowlist means"; asked=collect(tc.allowed), applied=[t.name for t in tools])
        tc=reconstruct(tc; allowed=())
        config=reconstruct(config; tool_choice=tc)
    end
    tc!==nothing && tc.mode=="required" && config.response_format!==nothing && throw(UnsupportedFeatureError(
        "xai: a forced tool (mode='required') cannot be combined with response_format — api.x.ai returns JSON text and drops the call (verified live 2026-09-02)";
        provider=l.provider, feature="config.tool_choice.mode"))
    return reconstruct(r; tools, config)
end
# A server that ignores every tool_choice but auto (Z.AI): "none" and an allowlist
# have a client-side form; "required" cannot be forced and is refused.
function forced_choice_prepare(l, r)
    tc=r.config.tool_choice
    tc.mode=="required" && throw(UnsupportedFeatureError(
        "$(l.provider): tool_choice mode='required' is silently ignored by this server (only 'auto' is honoured) and a forced call cannot be reproduced client-side";
        provider=l.provider, feature="config.tool_choice.mode"))
    tools=if tc.mode=="none"
        adapt!("config.tool_choice.mode", "client_side", "this server ignores tool_choice='none'; no tools were sent, which is the same outcome"; asked="none", applied="no tools sent")
        ()
    else
        kept=Tuple(t for t in r.tools if t isa FunctionTool && t.name in tc.allowed)
        adapt!("config.tool_choice.allowed", "client_side", "this server ignores tool_choice allowlists; only the allowed tools were sent, which is what the allowlist means"; asked=collect(tc.allowed), applied=[t.name for t in kept])
        kept
    end
    return reconstruct(r; tools, config=reconstruct(r.config; tool_choice=reconstruct(tc; mode="auto", allowed=())))
end
function openai_payload(l, r; stream=false, chat=false)
    c=effective_compat(l, r)
    l.provider=="xai" && (r=xai_prepare(l, r))
    if chat && c.forced_tool_choice=="reject" && r.config.tool_choice!==nothing &&
        (r.config.tool_choice.mode!="auto" || !isempty(r.config.tool_choice.allowed))
        r=forced_choice_prepare(l, r)
    end
    config=r.config
    boundary=adapted_breakpoint(r, c.cache_control)
    rows=openai_messages(l, r, c; chat, boundary)
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
    for key in (:seed, :frequency_penalty, :presence_penalty)
        value=getfield(config, key)
        value===nothing && continue
        if chat
            d[string(key)]=value
        else
            adapt!("config.$(key)", "dropped", "the Responses wire has no $(key) field (the Chat Completions dialect carries it)"; asked=value)
        end
    end
    config.user_id===nothing || (d[chat ? c.user_field : "safety_identifier"]=config.user_id)
    config.top_k===nothing || adapt!(
        "config.top_k", "dropped",
        chat ? "the Chat Completions wire has no top_k (Anthropic and Gemini carry it; servers that accept it take it through extensions)" :
               "the Responses wire has no top_k (Anthropic and Gemini carry it)";
        asked=config.top_k,
    )
    if !isempty(config.stop)
        if chat
            d["stop"]=collect(config.stop)
        else
            adapt!("config.stop", "client_side", "the Responses wire has no stop field; the reply is streamed and the connection closed at the first stop sequence (whether the provider then stops generating, and billing, is its own behaviour); the usage report rides only the final frame, so it is not reported when the cut happens (never estimated)"; asked=collect(config.stop), applied=collect(config.stop))
        end
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
        d["tool_choice"]=openai_choice(r, c, l.provider; chat)
        tc.parallel===nothing || (d["parallel_tool_calls"]=tc.parallel)
    end
    if config.response_format!==nothing
        if chat && c.json_schema=="reject" && config.response_format["type"]=="json_schema"
            adapt!("config.response_format", "dropped", "this server accepts response_format type 'json_schema' and does not apply it; use {'type': 'json_object'} and describe the shape in the prompt"; asked=config.response_format)
        else
            # MAP-14: the judgment convention goes verbatim (strict honours
            # anyOf/const/title); a distribution cannot be measured here.
            note_unmeasurable_probabilities(r, l.provider)
            d[chat ? "response_format" : "text"]=structured_openai(config.response_format; chat)
        end
    end
    openai_reasoning!(d, config.reasoning, c, l; chat)
    openai_cache!(d, r, c, l.provider; boundary)
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
    p isa DataPart && return obj("type"=>"text", "text"=>data_part_text(p))
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
# Output ceilings by model class, for the max_tokens the Messages API requires.
function anthropic_default_max_tokens(model)
    lowered=lowercase(model)
    for (marker, ceiling) in (("claude-3-haiku", 4096), ("claude-3-opus", 4096), ("claude-3-sonnet", 4096), ("claude-3-5-", 8192), ("claude-3.5-", 8192))
        occursin(marker, lowered) && return ceiling
    end
    return 16384
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
    # MAP-13: the Messages API cannot restrict to a subset of the declared tools;
    # only the allowed ones are sent, which is what the allowlist means.
    tools=r.tools
    tc=config.tool_choice
    if tc!==nothing && tc.mode!="none" && !isempty(tc.allowed) &&
        !(length(tc.allowed)==1 && tc.mode=="required") && Set(tc.allowed)!=Set(t.name for t in r.tools)
        tools=Tuple(t for t in r.tools if t.name in tc.allowed)
        adapt!("config.tool_choice.allowed", "client_side", "the Messages API cannot restrict to a subset of the declared tools; only the allowed tools were sent, which is what the allowlist means"; asked=collect(tc.allowed), applied=[t.name for t in tools])
    end
    reasoning=config.reasoning
    active=reasoning!==nothing && reasoning.effort!="off"
    adaptive=active && (
        c.thinking_format in ("deepseek", "adaptive", "effort") || anthropic_adaptive(r.model)
    )
    effort=reasoning===nothing ? nothing : reasoning.effort
    thinking_budget=reasoning===nothing ? nothing : reasoning.thinking_budget
    if active
        if c.reasoning_efforts!==nothing && !(effort in c.reasoning_efforts)
            nearest=nearest_effort(effort, c.reasoning_efforts)
            adapt!("config.reasoning.effort", "clamped", "this server has no '$(effort)' level (it accepts $(join(c.reasoning_efforts, ", "))) and would have accepted the word silently"; asked=effort, applied=nearest)
            effort=nearest
        end
        reasoning.summary in ("concise", "detailed") && adapt!("config.reasoning.summary", "substituted", "the Messages API has no summary detail levels; it returns thinking blocks whenever thinking runs, which is 'auto'"; asked=reasoning.summary, applied="auto")
        if adaptive
            if thinking_budget!==nothing
                adapt!("config.reasoning.thinking_budget", "dropped",
                    c.thinking_format=="deepseek" ? "this server ignores budget_tokens; effort is the dial" :
                    c.thinking_format=="adaptive" ? "this server accepts budget_tokens without translating it; effort is the dial (protocols--messages.md)" :
                    "$(r.model) takes thinking.type 'adaptive' with output_config.effort; budget_tokens is rejected by the API (live 2026-09-02)";
                    asked=thinking_budget)
                thinking_budget=nothing
            end
            if c.thinking_format=="anthropic" && effort=="minimal"
                adapt!("config.reasoning.effort", "clamped", "this model class has no 'minimal' level (output_config.effort is low|medium|high|xhigh|max); 'low' is the floor"; asked="minimal", applied="low")
                effort="low"
            end
        end
    end
    budget=active && !adaptive ? something(thinking_budget, EFFORT_BUDGETS[effort]) : nothing
    # The Messages API requires max_tokens; when none was set, the class default is used and recorded.
    visible=config.max_tokens
    if visible===nothing
        visible=anthropic_default_max_tokens(r.model)
        adapt!("config.max_tokens", "defaulted", "the Messages API requires max_tokens and none was set; the class default was used"; applied=visible)
    end
    d=obj(
        "model"=>r.model,
        "messages"=>messages,
        "stream"=>stream,
        "max_tokens"=>visible+something(budget, 0),
    )
    cache=config.cache
    usecache=cache!==nothing && cache.mode!="off" && c.cache_control=="anthropic"
    marker=obj("type"=>"ephemeral")
    cache!==nothing && cache.retention=="long" && (marker["ttl"]="1h")
    if cache !== nothing
        cache.key===nothing || adapt!("config.cache.key", "dropped", "the Messages API has no cache affinity key (OpenAI's prompt_cache_key); marks on blocks are its mechanism"; asked=cache.key)
        cache.retention=="long" && c.cache_control!="anthropic" && adapt!("config.cache.retention", "dropped", "this server caches implicitly and has no cache-control TTL"; asked="long")
        cache.resource===nothing || refuse(l.provider, "config.cache.resource", "this server has no stored-cache tier; sending without it would drop the prompt prefix the resource holds")
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
        if c.sampling_params=="reject"
            adapt!("config.$(name)", "dropped", "this server ignores sampling parameters (the model's sampling is fixed)"; asked=value)
            continue
        end
        if name===:temperature && value>1.0
            adapt!("config.temperature", "clamped", "the Messages API accepts temperature in [0, 1]; the canonical range is [0, 2]"; asked=value, applied=1.0)
            value=1.0
        end
        d[string(name)]=value
    end
    for name in (:seed, :frequency_penalty, :presence_penalty)
        value=getfield(config, name)
        value===nothing || adapt!("config.$(name)", "dropped", "the Messages API has no $(name) field"; asked=value)
    end
    isempty(config.stop) || (d["stop_sequences"]=collect(config.stop))
    if !isempty(tools)
        d["tools"]=[
            if t isa FunctionTool
                obj("name"=>t.name, "description"=>t.description, "input_schema"=>t.parameters)
            else
                merge(
                    obj("type"=>get(ANTHROPIC_BUILTINS, t.name, t.name), "name"=>t.name),
                    something(t.config, obj()),
                )
            end for t in tools
        ]
    end
    if tc!==nothing
        choice=obj("type"=>tc.mode=="required" ? "any" : tc.mode)
        !isempty(tc.allowed) && length(tc.allowed)==1 && tc.mode=="required" &&
            (choice=obj("type"=>"tool", "name"=>only(tc.allowed)))
        if c.parallel_tool_calls=="reject" && tc.parallel!==nothing
            adapt!("config.tool_choice.parallel", "dropped", "this server accepts disable_parallel_tool_use and does not apply it (guide--anthropic-api.md); the model may return several calls"; asked=tc.parallel)
        elseif tc.parallel===false && tc.mode!="none"
            choice["disable_parallel_tool_use"]=true
        end
        d["tool_choice"]=choice
    end
    if reasoning!==nothing
        if reasoning.effort=="off"
            c.thinking_format!="anthropic" && (d["thinking"]=obj("type"=>"disabled"))
        elseif adaptive
            c.thinking_format=="effort" ||
                (d["thinking"]=obj("type"=>c.thinking_format=="deepseek" ? "enabled" : "adaptive"))
            d["output_config"]=obj("effort"=>effort)
        else
            d["thinking"]=obj("type"=>"enabled", "budget_tokens"=>budget)
        end
    end
    if config.response_format!==nothing
        if c.structured_output=="reject"
            adapt!("config.response_format", "dropped", "this server accepts output_config.format and does not apply it; describe the shape in the prompt"; asked=config.response_format)
        else
            config.response_format["type"]=="json_schema" || throw(UnsupportedFeatureError(
                "anthropic: response_format json_object is not supported — the Messages API has no any-JSON mode; give a json_schema (objects need additionalProperties: false)";
                provider=l.provider, feature="config.response_format"))
            # MAP-14 §2: a judgment property carrying type+anyOf has its type moved
            # into every branch (the wire 400s on the combination).
            note_unmeasurable_probabilities(r, l.provider)
            output=get!(d, "output_config", obj())
            output["format"]=obj("type"=>"json_schema", "schema"=>anthropic_schema(config.response_format["schema"], request_judgments(r)))
        end
    end
    config.store===false && adapt!("config.store", "satisfied", "the Messages API has no stored-response object to opt out of; nothing retrievable is kept"; asked=false)
    config.store===true && adapt!("config.store", "dropped", "the Messages API has no stored-response object to opt into (OpenAI and Gemini carry `store`)"; asked=true)
    config.logprobs===nothing || adapt!("config.logprobs", "dropped", "the Messages API does not expose token log probabilities (OpenAI and Gemini carry them); Response.logprobs will be absent"; asked=config.logprobs)
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
    elseif p isa DataPart
        return obj("text"=>data_part_text(p))
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
# Gemini's Schema object: the keys its OpenAPI fields (responseSchema, parameters) parse.
const GEMINI_SCHEMA_FIELDS=Set((
    "type", "format", "title", "description", "nullable", "enum", "maxItems", "minItems",
    "properties", "required", "minProperties", "maxProperties", "minLength", "maxLength",
    "pattern", "example", "anyOf", "propertyOrdering", "default", "items", "minimum", "maximum",
))
"""
MAP-16: can Gemini's OpenAPI field carry `schema`? No when a schema node (the root, a
value of `properties`, `items`, an element of `anyOf`) is a boolean, has a key that is
not a Schema field, has a list `type`, or an `enum` with a non-string element. `example`
and `default` are values, never walked.
"""
function gemini_openapi_schema(schema)
    stack=Any[schema]
    while !isempty(stack)
        node=pop!(stack)
        node isa Bool && return false
        node isa AbstractDict || continue
        for (key, value) in node
            key in GEMINI_SCHEMA_FIELDS || return false
            key=="type" && value isa AbstractVector && return false
            key=="enum" && value isa AbstractVector && any(v->!(v isa AbstractString), value) && return false
            if key=="properties" && value isa AbstractDict
                append!(stack, collect(values(value)))
            elseif key=="items"
                push!(stack, value)
            elseif key=="anyOf"
                value isa AbstractVector ? append!(stack, value) : push!(stack, value)
            end
        end
    end
    return true
end
function gemini_tools(tools)
    functions=[
        obj(
            "name"=>t.name, "description"=>t.description,
            (gemini_openapi_schema(t.parameters) ? "parameters" : "parametersJsonSchema")=>t.parameters,
        ) for t in tools if t isa FunctionTool
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
        cache.key===nothing || adapt!("config.cache.key", "dropped", "GenerateContent has no cache affinity key; implicit caching applies, and a stored cache (lm.cache(prefix), cache.resource) is the explicit tier"; asked=cache.key)
        cache.retention in (nothing, "short") || adapt!("config.cache.retention", "dropped", "GenerateContent takes no lifetime in-request; it belongs to the stored cache (cache_create(..., ttl_seconds=...) / cache_update)"; asked=cache.retention)
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
        (:seed, "seed"),
        (:frequency_penalty, "frequencyPenalty"),
        (:presence_penalty, "presencePenalty"),
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
        if f["type"]=="json_schema"
            # MAP-14 §2: judgment properties go as enum with the descriptions
            # folded into the property description.
            note_unmeasurable_probabilities(r, l.provider)
            schema=gemini_schema(f["schema"], request_judgments(r))
            gen[gemini_openapi_schema(schema) ? "responseSchema" : "responseJsonSchema"]=schema
        end
    end
    reasoning=config.reasoning
    if reasoning!==nothing
        thinking=obj()
        level=gemini_level(r.model)
        effort=reasoning.effort
        if effort=="off" && level
            adapt!("config.reasoning.effort", "substituted", "$(r.model) cannot disable thinking (the Gemini 3 class honours no off switch); the lowest level was sent and the thinking spend is visible in usage"; asked="off", applied="minimal")
            effort="minimal"
        end
        if effort=="off"
            thinking["thinkingBudget"]=0
        else
            summary=reasoning.summary
            if summary in ("concise", "detailed")
                adapt!("config.reasoning.summary", "substituted", "GenerateContent has includeThoughts only, no detail levels; 'auto' shows the thoughts"; asked=summary, applied="auto")
                summary="auto"
            end
            summary===nothing || (thinking["includeThoughts"]=true)
            if reasoning.thinking_budget!==nothing
                thinking["thinkingBudget"]=reasoning.thinking_budget
            elseif level
                if effort in ("xhigh", "max")
                    adapt!("config.reasoning.effort", "clamped", "the Gemini 3 class has thinkingLevel minimal|low|medium|high; 'high' is the ceiling"; asked=effort, applied="high")
                    effort="high"
                end
                thinking["thinkingLevel"]=effort
            else
                thinking["thinkingBudget"]=EFFORT_BUDGETS[effort]
            end
        end
        gen["thinkingConfig"]=thinking
    end
    isempty(gen) || (d["generationConfig"]=gen)
    if resource===nothing
        isempty(r.tools) || (d["tools"]=gemini_tools(r.tools))
        tc=config.tool_choice
        if tc!==nothing
            tc.parallel===false && adapt!("config.tool_choice.parallel", "dropped", "GenerateContent has no parallel-tool-calls knob and may return several calls (OpenAI and Anthropic carry it)"; asked=false)
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
    config.user_id===nothing || adapt!("config.user_id", "dropped", "GenerateContent has no end-user attribution field (OpenAI and Anthropic carry it)"; asked=config.user_id)
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
    check_message_media(l, r)
    l.dialect=="typesafe" && return typesafe_payload(l, r; stream)
    l.dialect=="openai-responses" && return openai_payload(l, r; stream)
    l.dialect=="openai-chat" && return openai_payload(l, r; stream, chat=true)
    l.dialect=="anthropic" && return anthropic_payload(l, r; stream)
    return gemini_payload(l, r)
end
build_request(l::ProviderLM, r::Request; stream=false) = first(build_request_adapted(l, r; stream))
# The wire request and the MAP-13 record of what its build adapted.
function build_request_adapted(l::ProviderLM, r::Request; stream=false)
    stream && l.dialect=="typesafe" &&
        refuse(l.provider, "stream", "systemone answers in one piece; there is no stream to wrap")
    require_surface(l, stream ? :stream : :complete)
    payload, records=collecting(()->build_payload(l, r; stream), l.adaptations, l.provider)
    endpoint=if l.dialect=="typesafe"
        "v1/systemone"
    elseif l.dialect=="openai-responses"
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
    ), records
end
"""
    plan(client, request)

What a call with this request would adapt (MAP-13), with no network and no credential
read. Raises what the call would raise; returns the full record under every policy.
"""
function plan(l::ProviderLM, r::Request; stream=false)
    require_surface(l, stream ? :stream : :complete)
    return last(collecting(()->build_payload(l, r; stream), l.adaptations, l.provider))
end
