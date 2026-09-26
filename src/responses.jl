const OPENAI_ERROR_MAP=Dict(
    "server_error"=>ServerError,
    "rate_limit_exceeded"=>RateLimitError,
    # Azure documents these on Responses error frames even under HTTP 200.
    "no_capacity"=>RateLimitError,
    "too_many_requests"=>RateLimitError,
    "invalid_prompt"=>InvalidRequestError,
    "vector_store_timeout"=>TimeoutError,
    "context_length_exceeded"=>ContextLengthError,
    "invalid_api_key"=>AuthError,
    "insufficient_quota"=>BillingError,
    "1113"=>BillingError,
    "exceeded_current_quota_error"=>BillingError,
    "authentication_error"=>AuthError,
    "rate_limit_error"=>RateLimitError,
    "model_not_found"=>UnsupportedModelError,
    "model_not_available"=>UnsupportedModelError,
    "unsupported_model"=>UnsupportedModelError,
    "DeploymentNotFound"=>UnsupportedModelError,
)
for key in (
    "invalid_image",
    "invalid_image_format",
    "invalid_base64_image",
    "invalid_image_url",
    "image_too_large",
    "image_too_small",
    "image_parse_error",
    "image_content_policy_violation",
    "invalid_image_mode",
    "image_file_too_large",
    "unsupported_image_media_type",
    "empty_image_file",
    "failed_to_download_image",
    "image_file_not_found",
)
    OPENAI_ERROR_MAP[key]=InvalidRequestError
end
const ANTHROPIC_ERROR_MAP=Dict(
    "authentication_error"=>AuthError,
    "permission_error"=>AuthError,
    "billing_error"=>BillingError,
    "rate_limit_error"=>RateLimitError,
    "request_too_large"=>InvalidRequestError,
    "not_found_error"=>InvalidRequestError,
    "resource_not_found_error"=>InvalidRequestError,
    "DeploymentNotFound"=>UnsupportedModelError,
    "invalid_authentication_error"=>AuthError,
    "invalid_request_error"=>InvalidRequestError,
    "api_error"=>ServerError,
    "overloaded_error"=>ServerError,
    "timeout_error"=>TimeoutError,
)
const GEMINI_ERROR_MAP=Dict(
    "INVALID_ARGUMENT"=>InvalidRequestError,
    "FAILED_PRECONDITION"=>BillingError,
    "PERMISSION_DENIED"=>AuthError,
    "UNAUTHENTICATED"=>AuthError,
    "NOT_FOUND"=>InvalidRequestError,
    "RESOURCE_EXHAUSTED"=>RateLimitError,
    "INTERNAL"=>ServerError,
    "UNAVAILABLE"=>ServerError,
    "DEADLINE_EXCEEDED"=>TimeoutError,
)
# MAP-15: the pinned forms of a provider's "no such model" answer that carry no
# model-specific code and no not-found class (lm15-contract
# spec/model-not-found.json, carried verbatim; each form has a live receipt).
const MODEL_NOT_FOUND_FORMS=Tuple(
    (; (Symbol(k)=>v for (k, v) in form if k in ("code", "prefix", "contains", "suffix"))...) for
    form in JSON.parse(read(joinpath(@__DIR__, "data", "model_not_found.json"), String))
)
function pinned_model_not_found(code, message)
    (code isa AbstractString && !isempty(code)) || return false
    text=message isa AbstractString ? message : ""
    return any(MODEL_NOT_FOUND_FORMS) do f
        f.code==code &&
            (!haskey(f, :prefix) || startswith(text, f.prefix)) &&
            (!haskey(f, :contains) || occursin(f.contains, text)) &&
            (!haskey(f, :suffix) || endswith(text, f.suffix))
    end
end
function model_error(message)
    return occursin("model", lowercase(message)) && any(
        s->occursin(s, lowercase(message)),
        (
            "not found",
            "does not exist",
            "not exist",
            "not supported",
            "unsupported",
            "not available",
            "unknown",
        ),
    )
end
function context_error(message)
    return any(
        s->occursin(s, lowercase(message)),
        ("prompt is too long", "too many tokens", "context window", "context length"),
    ) || (
        occursin("token", lowercase(message)) &&
        any(s->occursin(s, lowercase(message)), ("limit", "exceed"))
    )
end
function normalize_error(l::ProviderLM, status, body)
    l.dialect=="typesafe" && return typesafe_error(l, status, body)
    data=try
        JSON.parse(body)
    catch
        obj()
    end
    data=asobject(data)
    raw=get(data, "error", nothing)
    inner=asobject(raw)
    code=nothing
    requestid=string_field(data, "request_id")
    message=""
    T=nothing
    if l.dialect in ("openai-responses", "openai-chat")
        message=raw isa AbstractString ? String(raw) : wire_string(get(inner, "message", ""))
        code=wire_string(
            first_nonempty(
                get(inner, "code", nothing),
                get(inner, "type", nothing),
                l.provider=="xai" ? get(data, "code", nothing) : nothing,
            ),
        )
        typ=wire_string(get(inner, "type", nothing))
        if l.access.backend=="chatgpt-codex" && string_field(data, "detail")!==nothing
            message=data["detail"]
            model_error(message) && (T=UnsupportedModelError)
        elseif code=="context_length_exceeded"
            T=ContextLengthError
        elseif code in (
            "model_not_found", "model_not_available", "unsupported_model", "DeploymentNotFound"
        ) || (status==404 && model_error(message))
            T=UnsupportedModelError
        elseif code in ("insufficient_quota", "1113") ||
            typ in ("insufficient_quota", "exceeded_current_quota_error")
            T=BillingError
        elseif code=="invalid_api_key" || typ=="authentication_error"
            T=AuthError
        elseif code=="rate_limit_exceeded" || typ=="rate_limit_error"
            T=RateLimitError
        end
    elseif l.dialect=="anthropic"
        isempty(inner) && (inner=data)
        message=raw isa AbstractString ? raw : wire_string(get(inner, "message", ""))
        code=wire_string(first_nonempty(get(inner, "type", nothing), get(inner, "code", nothing)))
        T=if context_error(message)
            ContextLengthError
        elseif code=="DeploymentNotFound" ||
            (code in ("not_found_error", "resource_not_found_error") && model_error(message))
            UnsupportedModelError
        else
            get(ANTHROPIC_ERROR_MAP, code, nothing)
        end
    else
        message=raw isa AbstractString ? raw : wire_string(get(inner, "message", ""))
        code=wire_string(get(inner, "status", nothing))
        T=if context_error(message) || occursin("too long", lowercase(message))
            ContextLengthError
        elseif code=="NOT_FOUND" && model_error(message)
            UnsupportedModelError
        else
            get(GEMINI_ERROR_MAP, code, nothing)
        end
    end
    pinned_model_not_found(code, message) && (T=UnsupportedModelError)  # MAP-15
    isempty(message) && (message="HTTP $status") # Do not echo arbitrary error bodies containing credentials.
    requestid=first_nonempty(requestid, string_field(inner, "request_id"))
    hint=get(inner, "retry_after", get(data, "retry_after", nothing))
    retry_after=if hint isa Real && !(hint isa Bool) && isfinite(hint) && hint>=0
        Float64(hint)
    else
        nothing
    end
    metadata=(
        provider=l.provider,
        provider_code=empty_optional(code) ? nothing : code,
        request_id=requestid,
        retry_after=retry_after,
    )
    error=if T===nothing
        map_http_error(status, message; metadata...)
    else
        T(message; status=Int(status), metadata...)
    end
    if error isa AuthError &&
        (l.access.credential_policy=="oauth" || l.credentials_source===:stored)
        error=AuthError(
            message; status=Int(status), metadata..., credential_hint=l.access.login_hint
        )
    elseif error isa AuthError
        # A cloud door refusing an identity is an IAM or token question, not a
        # mistyped key: say which role, or which kind of credential.
        hint=wire_auth_hint(l.access, status, sent_credential_kind(l))
        hint===nothing || (error=with_metadata(error; credential_hint=hint))
    end
    if error isa AuthError
        # AUTH-1 provenance: where the rejected credential came from, never the value.
        origin=try
            credential_origin(l)
        catch e
            e isa InterruptException && rethrow()
            nothing
        end
        origin===nothing || (error=with_metadata(error; credential_origin=origin))
    end
    return error
end
"""What a client sends: "key", "token", or nothing when a callable decides per request."""
function sent_credential_kind(l)
    c=l.credential
    c isa CloudCredentialProvider && return "token"
    c isa BearerToken && return "token"
    c isa ApiKey && return looks_like_access_token(c.value)===nothing ? "key" : "token"
    c isa AbstractString && return looks_like_access_token(c)===nothing ? "key" : "token"
    return nothing
end
function wire_auth_hint(policy, status, sent)
    policy.credential_policy=="gcp-chain" || return nothing
    status==403 && return "give the identity named above the Vertex AI User role (roles/aiplatform.user) on the project and enable the Vertex AI API (aiplatform.googleapis.com); a new project or a new grant can take a few minutes to apply. To use another identity: `gcloud auth application-default login`, or GOOGLE_APPLICATION_CREDENTIALS=<file>"
    status==401 || return nothing
    sent=="key" && return "Google refused this API key: use a Vertex AI key (Cloud console > APIs & Services > Credentials, restricted to the Vertex AI API or bound to a service account); Claude on Vertex takes no keys. If the value is an access token that does not start with `ya29.`, pass BearerToken(value)"
    return "Google refused this access token: it expired (they last an hour; pass a callable, or let lm15's chain refresh it) or it is not an OAuth token. Sign in again with `gcloud auth application-default login`"
end
function error_detail(l, code, message)
    table=if l.dialect=="anthropic"
        ANTHROPIC_ERROR_MAP
    elseif l.dialect=="gemini"
        GEMINI_ERROR_MAP
    else
        OPENAI_ERROR_MAP
    end
    T=get(table, code, GenericProviderError)
    pinned_model_not_found(code, message) && (T=UnsupportedModelError)  # MAP-15
    if l.dialect in ("anthropic", "gemini")
        context_error(message) && (T=ContextLengthError)
        code in ("NOT_FOUND", "not_found_error") &&
            model_error(message) &&
            (T=UnsupportedModelError)
    end
    return ErrorDetail(;
        code=ERROR_CODES[T],
        message=isempty(message) ? code : message,
        provider_code=isempty(code) ? "provider" : code,
    )
end
const CHAT_FINISH=Dict(
    "stop"=>"stop",
    "length"=>"length",
    "tool_calls"=>"tool_call",
    "function_call"=>"tool_call",
    "content_filter"=>"content_filter",
)
const GEMINI_BLOCKED=(
    "SAFETY",
    "RECITATION",
    "LANGUAGE",
    "BLOCKLIST",
    "PROHIBITED_CONTENT",
    "SPII",
    "MALFORMED_FUNCTION_CALL",
    "IMAGE_SAFETY",
    "IMAGE_PROHIBITED_CONTENT",
    "IMAGE_OTHER",
    "NO_IMAGE",
    "IMAGE_RECITATION",
    "UNEXPECTED_TOOL_CALL",
    "TOO_MANY_TOOL_CALLS",
    "MISSING_THOUGHT_SIGNATURE",
    "MALFORMED_RESPONSE",
)
function finish_reason(dialect, raw; has_tool=false)
    has_tool && return "tool_call"
    reason=wire_string(raw)
    dialect=="openai-chat" && return get(CHAT_FINISH, reason, "stop")
    if dialect=="anthropic"
        reason in ("max_tokens", "model_context_window_exceeded") && return "length"
        reason in ("tool_use", "pause_turn") && return "tool_call"
        reason in ("refusal", "safety", "content_filter") && return "content_filter"
    elseif dialect=="gemini"
        reason=="MAX_TOKENS" && return "length"
        reason in ("SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII") &&
            return "content_filter"
    end
    return "stop"
end
function response_finish(d; has_tool=false)
    has_tool && return "tool_call"
    reason=lowercase(wire_string(get(getobject(d, "incomplete_details"), "reason", nothing)))
    get(d, "status", nothing)=="incomplete" && occursin("token", reason) && return "length"
    return if occursin("content_filter", reason) || occursin("safety", reason)
        "content_filter"
    else
        "stop"
    end
end
function usage_from(dialect, raw; live=false)
    d=asobject(raw)
    if dialect=="anthropic"
        return Usage(;
            input_tokens=get(d, "input_tokens", nothing),
            output_tokens=get(d, "output_tokens", nothing),
            cache_read_tokens=get(d, "cache_read_input_tokens", nothing),
            cache_write_tokens=get(d, "cache_creation_input_tokens", nothing),
            reasoning_tokens=get(getobject(d, "output_tokens_details"), "thinking_tokens", nothing),
        )
    elseif dialect=="gemini"
        isempty(d) && return Usage()
        function modality(rows)
            numbers=[
                get(row, "tokenCount", 0) for row in asarray(rows) if
                row isa AbstractDict && get(row, "modality", nothing)=="AUDIO"
            ]
            return isempty(numbers) ? nothing : sum(numbers)
        end
        output=get(
            d,
            live ? "responseTokenCount" : "candidatesTokenCount",
            get(d, live ? "candidatesTokenCount" : "responseTokenCount", 0),
        )
        return Usage(;
            input_tokens=get(d, "promptTokenCount", 0),
            output_tokens=output,
            total_tokens=get(d, "totalTokenCount", nothing),
            cache_read_tokens=get(d, "cachedContentTokenCount", nothing),
            reasoning_tokens=get(d, "thoughtsTokenCount", nothing),
            input_audio_tokens=modality(get(d, "promptTokensDetails", nothing)),
            output_audio_tokens=modality(
                first_nonempty(
                    get(d, "candidatesTokensDetails", nothing),
                    get(d, "responseTokensDetails", nothing),
                ),
            ),
        )
    end
    chat=dialect=="openai-chat"
    input=getobject(d, chat ? "prompt_tokens_details" : "input_tokens_details")
    output=getobject(d, chat ? "completion_tokens_details" : "output_tokens_details")
    return Usage(;
        input_tokens=get(d, chat ? "prompt_tokens" : "input_tokens", nothing),
        output_tokens=get(d, chat ? "completion_tokens" : "output_tokens", nothing),
        total_tokens=get(d, "total_tokens", nothing),
        cache_read_tokens=first_nonempty(get(input, "cached_tokens", nothing), chat ? get(d, "cached_tokens", nothing) : nothing),
        cache_write_tokens=get(input, "cache_write_tokens", nothing),
        reasoning_tokens=get(output, "reasoning_tokens", nothing),
        input_audio_tokens=get(input, "audio_tokens", nothing),
        output_audio_tokens=get(output, "audio_tokens", nothing),
    )
end
function openai_logprobs(raw)
    out=TokenLogprob[]
    for entry in asarray(raw)
        entry isa AbstractDict && haskey(entry, "token") && haskey(entry, "logprob") || continue
        top=TopLogprob[]
        for alt in getarray(entry, "top_logprobs")
            alt isa AbstractDict && haskey(alt, "token") && haskey(alt, "logprob") || continue
            push!(
                top,
                TopLogprob(;
                    token=wire_string(alt["token"]),
                    logprob=alt["logprob"],
                    bytes=get(alt, "bytes", nothing),
                ),
            )
        end
        push!(
            out,
            TokenLogprob(;
                token=wire_string(entry["token"]),
                logprob=entry["logprob"],
                bytes=get(entry, "bytes", nothing),
                top,
            ),
        )
    end
    return Tuple(out)
end
function gemini_logprobs(raw)
    d=asobject(raw)
    tops=getarray(d, "topCandidates")
    out=TokenLogprob[]
    for (i, c) in enumerate(getarray(d, "chosenCandidates"))
        c isa AbstractDict || continue
        entries=i<=length(tops) ? getarray(asobject(tops[i]), "candidates") : []
        top=Tuple(
            TopLogprob(;
                token=wire_string(get(a, "token", "")),
                logprob=get(a, "logProbability", 0.0),
                token_id=get(a, "tokenId", nothing),
            ) for a in entries if a isa AbstractDict
        )
        push!(
            out,
            TokenLogprob(;
                token=wire_string(get(c, "token", "")),
                logprob=get(c, "logProbability", 0.0),
                token_id=get(c, "tokenId", nothing),
                top,
            ),
        )
    end
    return Tuple(out)
end
function unmapped!(out, path, type)
    return push!(out, obj("path"=>path, "type"=>empty_optional(type) ? "<missing>" : string(type)))
end
wire_type(x) =
    if x===nothing
        "NoneType"
    elseif x isa AbstractString
        "str"
    elseif x isa Bool
        "bool"
    elseif x isa Integer
        "int"
    elseif x isa AbstractFloat
        "float"
    elseif x isa AbstractVector
        "list"
    else
        "dict"
    end
function attach_unmapped(d, unmapped)
    isempty(unmapped) && return d
    return merge(d, obj("_lm15_unmapped"=>unmapped))
end
function citation_from(d; source=nothing, anthropic=false)
    url=first_nonempty(get(d, "url", nothing), get(d, "uri", nothing))
    title=if anthropic
        first_nonempty(
            get(d, "title", nothing),
            get(d, "document_title", nothing),
            get(d, "source_title", nothing),
        )
    else
        first_nonempty(
            get(d, "title", nothing), get(d, "filename", nothing), get(d, "file_id", nothing)
        )
    end
    quoted=first_nonempty(
        get(d, "text", nothing),
        get(d, "snippet", nothing),
        get(d, "cited_text", nothing),
        get(d, "quote", nothing),
    )
    if quoted===nothing && source!==nothing
        a=get(d, "start_index", nothing)
        b=get(d, "end_index", nothing)
        if a isa Integer && b isa Integer && 0<=a<b<=length(source)
            quoted=String(collect(source)[(Int(a) + 1):Int(b)])
        end
    end
    url===nothing && title===nothing && quoted===nothing && return nothing
    return CitationPart(;
        url=url===nothing ? nothing : string(url),
        title=title===nothing ? nothing : string(title),
        text=quoted===nothing ? nothing : string(quoted),
    )
end
function unnamed_call(provider, path)
    return throw(
        GenericProviderError("tool call at $path has no name; refusing to guess"; provider)
    )
end
function parse_openai_response(l, r, d)
    rawerror=get(d, "error", nothing)
    if rawerror isa AbstractDict
        code=wire_string(get(rawerror, "code", ""))
        T=get(OPENAI_ERROR_MAP, code, ServerError)
        throw(
            T(
                wire_string(get(rawerror, "message", code));
                provider=l.provider,
                provider_code=isempty(code) ? nothing : code,
            ),
        )
    end
    parts=Part[]
    unmapped=Any[]
    logprobs=TokenLogprob[]
    for (i, item) in enumerate(getarray(d, "output"))
        path="output[$(i-1)]"
        item isa AbstractDict || (unmapped!(unmapped, path, wire_type(item)); continue)
        type=get(item, "type", nothing)
        if type=="message"
            for (j, content) in enumerate(getarray(item, "content"))
                cp="$path.content[$(j-1)]"
                content isa AbstractDict || (unmapped!(unmapped, cp, wire_type(content)); continue)
                tag=get(content, "type", nothing)
                if tag in ("output_text", "text")
                    s=wire_string(get(content, "text", nothing))
                    push!(parts, TextPart(s))
                    append!(logprobs, openai_logprobs(get(content, "logprobs", nothing)))
                    for a in getarray(content, "annotations")
                        a isa AbstractDict || continue
                        citation=citation_from(a; source=s)
                        citation===nothing || push!(parts, citation)
                    end
                elseif tag=="refusal"
                    s=wire_string(
                        first_nonempty(
                            get(content, "refusal", nothing), get(content, "text", nothing)
                        ),
                    )
                    push!(parts, isempty(s) ? TextPart("") : RefusalPart(s))
                elseif tag=="output_image"
                    data=first_nonempty(
                        get(content, "b64_json", nothing), get(content, "image_base64", nothing)
                    )
                    data===nothing || push!(parts, ImagePart(; data=string(data)))
                elseif tag=="output_audio"
                    data=first_nonempty(
                        get(getobject(content, "audio"), "data", nothing),
                        get(content, "b64_json", nothing),
                    )
                    data===nothing || push!(parts, AudioPart(; data=string(data)))
                else
                    unmapped!(unmapped, cp, tag)
                end
            end
        elseif type=="function_call"
            name=string_field(item, "name")
            name===nothing && unnamed_call(l.provider, path)
            id=string(
                first_nonempty(
                    get(item, "call_id", nothing), get(item, "id", nothing), "call_$(length(parts))"
                ),
            )
            push!(
                parts, ToolCallPart(; id, name, input=json_object(get(item, "arguments", nothing)))
            )
        elseif type=="reasoning"
            summary=get(item, "summary", nothing)
            s=if summary isa AbstractVector
                join(
                    (
                        v isa AbstractDict ? wire_string(get(v, "text", nothing)) : string(v) for
                        v in summary
                    ),
                    "\n",
                )
            else
                wire_string(first_nonempty(summary, get(item, "text", nothing)))
            end
            state=obj(
                (
                    k=>item[k] for
                    k in ("id", "encrypted_content") if !empty_optional(get(item, k, nothing))
                )...,
            )
            continuation=if isempty(state)
                ()
            else
                (ContinuationState(; provider="openai", kind="reasoning_item", data=state),)
            end
            isempty(s) && isempty(continuation) || push!(parts, ThinkingPart(s; continuation))
        elseif type in (
            "web_search_call",
            "file_search_call",
            "code_interpreter_call",
            "computer_call",
            "computer_use_call",
        )
            continue
        else
            unmapped!(unmapped, path, type)
        end
    end
    isempty(parts) && push!(parts, TextPart(wire_string(get(d, "output_text", nothing))))
    return Response(;
        id=string_field(d, "id"),
        model=wire_string(first_nonempty(get(d, "model", nothing), r.model)),
        message=assistant(parts),
        finish_reason=response_finish(d; has_tool=any(p->p isa ToolCallPart, parts)),
        usage=usage_from(l.dialect, get(d, "usage", nothing)),
        logprobs=isempty(logprobs) ? nothing : Tuple(logprobs),
        provider_data=attach_unmapped(d, unmapped),
    )
end
function parse_chat_response(l, r, d; choice=nothing)
    rawerror=get(d, "error", nothing)
    if rawerror isa AbstractDict
        code=wire_string(get(rawerror, "code", ""))
        T=get(OPENAI_ERROR_MAP, code, ServerError)
        throw(
            T(
                wire_string(get(rawerror, "message", code));
                provider=l.provider,
                provider_code=isempty(code) ? nothing : code,
            ),
        )
    end
    choices=getarray(d, "choices")
    choice===nothing &&
        length(choices)>1 &&
        unsupported(l.provider, "multiple response choices without an explicit choice index")
    index=choice===nothing ? 0 : integer_value(choice)
    choice===nothing ||
        0<=index<length(choices) ||
        throw(ArgumentError("choice index out of range"))
    chosen=isempty(choices) ? obj() : asobject(choices[index + 1])
    m=getobject(chosen, "message")
    parts=Part[]
    unmapped=Any[]
    path="choices[$index]"
    isempty(choices) ||
        choices[index + 1] isa AbstractDict ||
        unmapped!(unmapped, path, wire_type(choices[index + 1]))
    reason=first_nonempty(get(m, "reasoning_content", nothing), get(m, "reasoning", nothing))
    reason===nothing || push!(parts, ThinkingPart(string(reason)))
    content=get(m, "content", nothing)
    if content isa AbstractString
        isempty(content) || push!(parts, TextPart(content))
    elseif content isa AbstractVector
        for (i, c) in enumerate(content)
            if c isa AbstractDict && get(c, "type", nothing)=="text"
                push!(parts, TextPart(wire_string(get(c, "text", nothing))))
            else
                unmapped!(
                    unmapped,
                    "$path.message.content[$(i-1)]",
                    c isa AbstractDict ? get(c, "type", nothing) : wire_type(c),
                )
            end
        end
    elseif content!==nothing
        unmapped!(unmapped, "$path.message.content", wire_type(content))
    end
    ref=string_field(m, "refusal")
    ref===nothing || push!(parts, RefusalPart(ref))
    for (i, call) in enumerate(getarray(m, "tool_calls"))
        cp="$path.message.tool_calls[$(i-1)]"
        call isa AbstractDict || (unmapped!(unmapped, cp, wire_type(call)); continue)
        tag=get(call, "type", "function")
        tag in (nothing, "function", "") || (unmapped!(unmapped, cp, tag); continue)
        f=getobject(call, "function")
        name=string_field(f, "name")
        name===nothing && unnamed_call(l.provider, cp)
        push!(
            parts,
            ToolCallPart(;
                id=string(first_nonempty(get(call, "id", nothing), "call_$(length(parts))")),
                name,
                input=json_object(get(f, "arguments", nothing)),
            ),
        )
    end
    isempty(parts) && push!(parts, TextPart(""))
    rawfinish=get(chosen, "finish_reason", nothing)
    has_tool=any(p->p isa ToolCallPart, parts)
    !has_tool &&
        !empty_optional(rawfinish) &&
        !haskey(CHAT_FINISH, rawfinish) &&
        unmapped!(unmapped, "$path.finish_reason", rawfinish)
    logprobs=openai_logprobs(get(getobject(chosen, "logprobs"), "content", nothing))
    return Response(;
        id=string_field(d, "id"),
        model=string(first_nonempty(get(d, "model", nothing), r.model)),
        message=assistant(parts),
        finish_reason=finish_reason(l.dialect, rawfinish; has_tool),
        usage=usage_from(l.dialect, get(d, "usage", nothing)),
        logprobs=isempty(logprobs) ? nothing : logprobs,
        provider_data=attach_unmapped(d, unmapped),
    )
end
function parse_anthropic_response(l, r, d)
    parts=Part[]
    unmapped=Any[]
    for (i, b) in enumerate(getarray(d, "content"))
        path="content[$(i-1)]"
        b isa AbstractDict || (unmapped!(unmapped, path, wire_type(b)); continue)
        tag=get(b, "type", nothing)
        if tag=="text"
            push!(parts, TextPart(wire_string(get(b, "text", nothing))))
            for c in getarray(b, "citations")
                c isa AbstractDict || continue
                part=citation_from(c; anthropic=true)
                part===nothing || push!(parts, part)
            end
        elseif tag=="tool_use"
            name=string_field(b, "name")
            name===nothing && unnamed_call(l.provider, path)
            push!(
                parts,
                ToolCallPart(;
                    id=string(first_nonempty(get(b, "id", nothing), "tool_$(length(parts))")),
                    name,
                    input=getobject(b, "input"),
                ),
            )
        elseif tag=="thinking"
            sig=string_field(b, "signature")
            state=if sig===nothing
                ()
            else
                (
                    ContinuationState(;
                        provider="anthropic", kind="thinking_signature", data=obj("signature"=>sig)
                    ),
                )
            end
            push!(
                parts,
                ThinkingPart(
                    wire_string(
                        first_nonempty(get(b, "thinking", nothing), get(b, "text", nothing))
                    );
                    continuation=state,
                ),
            )
        elseif tag=="redacted_thinking"
            state=if get(b, "data", nothing)===nothing
                ()
            else
                (
                    ContinuationState(;
                        provider="anthropic", kind="redacted_thinking", data=obj("data"=>b["data"])
                    ),
                )
            end
            push!(parts, ThinkingPart(""; continuation=state))
        elseif tag in ("server_tool_use", "web_search_tool_result", "code_execution_tool_result")
            continue
        else
            unmapped!(unmapped, path, tag)
        end
    end
    isempty(parts) && push!(parts, TextPart(""))
    return Response(;
        id=string_field(d, "id"),
        model=string(first_nonempty(get(d, "model", nothing), r.model)),
        message=assistant(parts),
        finish_reason=finish_reason(
            l.dialect, get(d, "stop_reason", nothing); has_tool=any(p->p isa ToolCallPart, parts)
        ),
        usage=usage_from(l.dialect, get(d, "usage", nothing)),
        provider_data=attach_unmapped(d, unmapped),
    )
end
function gemini_inband(l, d)
    reason=string_field(getobject(d, "promptFeedback"), "blockReason")
    reason===nothing ||
        reason=="BLOCK_REASON_UNSPECIFIED" ||
        throw(
            InvalidRequestError(
                "Prompt blocked: $reason"; provider=l.provider, provider_code="promptFeedback"
            ),
        )
    candidates=getarray(d, "candidates")
    c=isempty(candidates) ? obj() : asobject(first(candidates))
    finish=get(c, "finishReason", nothing)
    finish in GEMINI_BLOCKED && throw(
        InvalidRequestError(
            wire_string(
                first_nonempty(get(c, "finishMessage", nothing), "Candidate blocked: $finish")
            );
            provider=l.provider,
            provider_code=finish,
        ),
    )
    return c
end
function thought_state(p)
    sig=get(p, "thoughtSignature", nothing)
    return if sig===nothing
        ()
    else
        (
            ContinuationState(;
                provider="gemini", kind="thought_signature", data=obj("value"=>string(sig))
            ),
        )
    end
end
function gemini_citations(candidate, fulltext)
    grounding=getobject(candidate, "groundingMetadata")
    chunks=getarray(grounding, "groundingChunks")
    out=CitationPart[]
    seen=Set()
    for support in getarray(grounding, "groundingSupports")
        support isa AbstractDict || continue
        segment=getobject(support, "segment")
        quoted=string_field(segment, "text")
        a=get(segment, "startIndex", nothing)
        b=get(segment, "endIndex", nothing)
        if quoted===nothing && a isa Integer && b isa Integer && 0<=a<b<=length(fulltext)
            quoted=String(collect(fulltext)[(Int(a) + 1):Int(b)])
        end
        for idx in getarray(support, "groundingChunkIndices")
            idx isa Integer && 0<=idx<length(chunks) || continue
            chunk=asobject(chunks[Int(idx) + 1])
            source=asobject(
                first_nonempty(
                    get(chunk, "web", nothing),
                    get(chunk, "retrievedContext", nothing),
                    get(chunk, "googleSearch", nothing),
                ),
            )
            url=first_nonempty(get(source, "uri", nothing), get(source, "url", nothing))
            title=first_nonempty(get(source, "title", nothing), get(source, "name", nothing))
            key=(url, title, quoted)
            key in seen && continue
            all(isnothing, key) && continue
            push!(seen, key)
            push!(out, CitationPart(; url, title, text=quoted))
        end
    end
    return out
end
function parse_gemini_response(l, r, d)
    candidate=gemini_inband(l, d)
    parts=Part[]
    unmapped=Any[]
    for (i, p) in enumerate(getarray(getobject(candidate, "content"), "parts"))
        path="candidates[0].content.parts[$(i-1)]"
        p isa AbstractDict || (unmapped!(unmapped, path, wire_type(p)); continue)
        if haskey(p, "text")
            T=get(p, "thought", false)===true ? ThinkingPart : TextPart
            push!(parts, T(wire_string(p["text"]); continuation=thought_state(p)))
        elseif get(p, "functionCall", nothing) isa AbstractDict
            fc=p["functionCall"]
            name=string_field(fc, "name")
            name===nothing && unnamed_call(l.provider, path)
            signature=first_nonempty(
                get(p, "thoughtSignature", nothing), get(fc, "thoughtSignature", nothing)
            )
            continuation=if signature===nothing
                ()
            else
                (
                    ContinuationState(;
                        provider="gemini",
                        kind="thought_signature",
                        data=obj("value"=>string(signature)),
                    ),
                )
            end
            push!(
                parts,
                ToolCallPart(;
                    id=string(first_nonempty(get(fc, "id", nothing), "tool_call_$(length(parts))")),
                    name,
                    input=getobject(fc, "args"),
                    continuation,
                ),
            )
        elseif haskey(p, "inlineData") || haskey(p, "fileData")
            inline=haskey(p, "inlineData")
            source=getobject(p, inline ? "inlineData" : "fileData")
            mime=string(
                first_nonempty(get(source, "mimeType", nothing), "application/octet-stream")
            )
            value=string_field(source, inline ? "data" : "fileUri")
            value===nothing && continue
            T=if startswith(mime, "image/")
                ImagePart
            elseif startswith(mime, "audio/")
                AudioPart
            else
                DocumentPart
            end
            push!(
                parts, inline ? T(; media_type=mime, data=value) : T(; media_type=mime, url=value)
            )
        elseif haskey(p, "executableCode") || haskey(p, "codeExecutionResult")
            continue
        else
            unmapped!(unmapped, path, isempty(p) ? "<empty>" : join(sort!(collect(keys(p))), "+"))
        end
    end
    append!(parts, gemini_citations(candidate, join((p.text for p in parts if p isa TextPart))))
    isempty(parts) && push!(parts, TextPart(""))
    logprobs=gemini_logprobs(get(candidate, "logprobsResult", nothing))
    return Response(;
        id=string_field(d, "responseId"),
        model=r.model,
        message=assistant(parts),
        finish_reason=finish_reason(
            l.dialect,
            get(candidate, "finishReason", nothing);
            has_tool=any(p->p isa ToolCallPart, parts),
        ),
        usage=usage_from(l.dialect, get(d, "usageMetadata", nothing)),
        logprobs=isempty(logprobs) ? nothing : logprobs,
        provider_data=attach_unmapped(d, unmapped),
    )
end
function parse_response(l::ProviderLM, r::Request, response::HttpResponse)
    r=wire_request(l, r)
    response.status>=400 && throw(
        attach_error_metadata(
            normalize_error(l, response.status, String(copy(response.body))), response
        ),
    )
    d=reply_json(l, response)
    d isa AbstractDict ||
        throw(GenericProviderError("response body must be a JSON object"; provider=l.provider))
    l.dialect=="typesafe" && return parse_typesafe_response(l, r, d, response)
    parsed=if l.dialect=="openai-responses"
        parse_openai_response(l, r, d)
    elseif l.dialect=="openai-chat"
        parse_chat_response(l, r, d)
    elseif l.dialect=="anthropic"
        parse_anthropic_response(l, r, d)
    else
        parse_gemini_response(l, r, d)
    end
    return fold_judgments(parsed, r)
end
function response_from_openai_chat(body; model=nothing, choice=nothing)
    resolved=first_nonempty(get(body, "model", nothing), model)
    resolved===nothing && throw(ArgumentError("body has no model; pass model"))
    l=OpenAIChatLM(; api_key="parse-only", env=Dict())
    return parse_chat_response(l, Request(string(resolved), (user(""),)), body; choice)
end
