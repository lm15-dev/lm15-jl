# TypeSafe System One (Jev), provider `typesafe` (changes/2026-09-17-judgments.md,
# changes/2026-09-19-jev-state.md). One `POST /v1/systemone` per Request: the one
# user part is Jev's `state`, verbatim; the judgment properties of the json_schema
# become its `questions` (MAP-14 §2); the answers come back as one DataPart with a
# distribution per judgment and method "provider_classification" (§3). Jev generates
# no text: a request without judgments, with tools, or with media is refused before
# the wire.

const TYPESAFE_BASE_URL = "https://api.typesafe.ai"
# Config knobs with no home on the systemone wire: dropped with a record.
const TYPESAFE_DROPPED = (
    :max_tokens, :temperature, :top_p, :top_k, :stop, :seed, :frequency_penalty,
    :presence_penalty, :reasoning, :logprobs, :store, :user_id, :service_tier, :cache,
)
function typesafe_state(l, r::Request)
    for (i, m) in enumerate(r.messages), (j, p) in enumerate(m.parts)
        p isa Union{TextPart,DataPart} || refuse(l.provider, "messages[$(i-1)].parts[$(j-1)]",
            "a $(kind(p)) part has no slot on the systemone wire (MAP-10); Jev reads text or data")
    end
    r.system===nothing || refuse(l.provider, "system",
        "Jev has no system prompt; put context in the state as a named key (user(data(Dict(\"policy\" => ..., \"note\" => ...)))), or the framing in each question's description (changes/2026-09-19-jev-state.md D2)")
    length(r.messages)==1 || refuse(l.provider, "messages",
        "Jev judges one state, got $(length(r.messages)) messages; put a transcript in the state as an array or object (user(data(Dict(\"messages\" => [...])))), where a question can point at a turn with a backtick path")
    m=only(r.messages)
    m.role=="user" || refuse(l.provider, "messages[0].role", "Jev's state is a user message, got role $(repr(m.role))")
    length(m.parts)==1 || refuse(l.provider, "messages[0].parts",
        "Jev's state is one text or data part, got $(length(m.parts)) parts; put several pieces in one data part as named keys")
    p=only(m.parts)
    return p isa TextPart ? p.text : p.value
end
function typesafe_questions(l, r::Request)
    f=r.config.response_format
    f isa AbstractDict && get(f, "type", nothing)=="json_schema" || refuse(l.provider, "config.response_format",
        "Jev answers declared judgments only; give a json_schema response_format whose properties are enums / booleans / ordered levels (MAP-14), e.g. judgments(...)")
    found=request_judgments(r)
    extra=non_judgment_properties(get(f, "schema", nothing), found)
    if isempty(found) || !isempty(extra)
        what=isempty(extra) ? "no property declares a judgment" : "properties $(extra) are free-form"
        refuse(l.provider, "config.response_format", "$what; Jev cannot generate values, only pick among declared keys (MAP-14 §1)")
    end
    questions=JSONObject()
    for (name, j) in found
        instruction=j.instruction
        if instruction===nothing
            adapt!("config.response_format.schema.properties.$name.description", "defaulted",
                "a judgment without a description: the property name goes as the instruction (Jev never sees property names)";
                applied=name)
            instruction=name
        end
        questions[name]=if j.kind=="boolean"
            obj("type"=>"noul", "instructions"=>instruction)
        elseif j.kind=="choice"
            length(j.keys)<=MAX_CHOICE_KEYS || refuse(l.provider, "config.response_format.schema.properties.$name",
                "a Jev choice takes at most $MAX_CHOICE_KEYS keys, got $(length(j.keys))")
            obj("type"=>"choice", "instructions"=>instruction,
                "criteria"=>JSONObject(k=>get(j.descriptions, k, nothing) for k in j.keys))
        else
            length(j.keys)<=MAX_ORDERED_LEVELS || refuse(l.provider, "config.response_format.schema.properties.$name",
                "a Jev score takes at most $MAX_ORDERED_LEVELS levels, got $(length(j.keys))")
            obj("type"=>"score", "instructions"=>instruction,
                "criteria"=>Any[something(get(j.descriptions, k, nothing), k) for k in j.keys])
        end
    end
    return questions
end
function typesafe_payload(l, r::Request; stream=false)
    stream && refuse(l.provider, "stream", "systemone answers in one piece; there is no stream to wrap")
    isempty(r.tools) || refuse(l.provider, "tools", "tools have no slot on the systemone wire")
    c=r.config
    c.tool_choice===nothing || refuse(l.provider, "config.tool_choice", "tool_choice has no slot on the systemone wire")
    for name in TYPESAFE_DROPPED
        value=getfield(c, name)
        (value===nothing || value==()) && continue
        asked=value isa Union{Real,AbstractString,Bool} ? value : (value isa Tuple ? collect(value) : to_dict(value))
        adapt!("config.$name", "dropped", "no such control on the systemone wire (Jev returns decisions, not samples)"; asked)
    end
    questions=typesafe_questions(l, r)
    d=obj("model"=>r.model, "state"=>typesafe_state(l, r), "questions"=>questions)
    for (key, value) in something(c.extensions, obj())
        key=="n" && value isa Real && value>1 &&
            refuse(l.provider, "config.extensions.n", "n > 1 has no canonical multiple-response representation")
        d[key]=value
    end
    return d
end
typesafe_request_id(r::HttpResponse) = header(r, "x-typesafe-request-id")
function parse_typesafe_response(l, r::Request, d, response::HttpResponse)
    found=request_judgments(r)
    rid=typesafe_request_id(response)
    invalid(path, detail)=attach_error_metadata(GenericProviderError(
        "malformed systemone reply at $path: $detail"; provider=l.provider, status=response.status, request_id=rid), response)
    function probability(raw, path)
        raw isa Real && !(raw isa Bool) && isfinite(raw) && 0<=raw<=1 ||
            throw(invalid(path, "expected a finite number in [0, 1]"))
        return Float64(raw)
    end
    answers=get(d, "answers", nothing)
    answers isa AbstractDict || throw(invalid("answers", "expected an object containing every declared judgment"))
    Set(keys(answers))==Set(keys(found)) || throw(invalid("answers", "keys must match the declared judgments exactly"))
    value=JSONObject()
    probabilities=JSONObject()
    for (name, j) in found
        answer=answers[name]
        path="answers.$name"
        answer isa AbstractDict || throw(invalid(path, "expected an answer object"))
        expected=Dict("boolean"=>"noul", "choice"=>"choice", "ordered"=>"score")[j.kind]
        get(answer, "type", nothing)==expected || throw(invalid("$path.type", "expected $(repr(expected))"))
        if j.kind=="boolean"
            p=probability(get(answer, "noul", nothing), "$path.noul")
            value[name]=p>=0.5
            probabilities[name]=JSONObject("true"=>p, "false"=>1.0-p)
            continue
        end
        dist=get(answer, "probabilities", nothing)
        dist isa AbstractDict && Set(keys(dist))==Set(j.keys) ||
            throw(invalid("$path.probabilities", "expected one probability for every declared key, and no other keys"))
        # INV-052: validate measurements individually, never their total.
        probs=JSONObject(k=>probability(dist[k], "$path.probabilities.$k") for k in j.keys)
        probabilities[name]=probs
        if j.kind=="choice"
            pick=get(answer, "choice", nothing)
            pick isa AbstractString && pick in j.keys || throw(invalid("$path.choice", "expected a declared choice key"))
            value[name]=String(pick)
        else
            best=j.keys[argmax([probs[k] for k in j.keys])]
            value[name]=parse(Int, best)
        end
    end
    part=try
        DataPart(; value, probabilities=isempty(probabilities) ? nothing : probabilities,
            method=isempty(probabilities) ? nothing : "provider_classification")
    catch e
        e isa ArgumentError || rethrow()
        throw(invalid("answers", sprint(showerror, e)))
    end
    raw_usage=get(d, "usage", nothing)
    raw_usage===nothing && (raw_usage=obj())
    raw_usage isa AbstractDict || throw(invalid("usage", "expected an object or null"))
    usage=try
        Usage(; input_tokens=get(raw_usage, "input_tokens", nothing), output_tokens=get(raw_usage, "output_tokens", nothing))
    catch e
        e isa ArgumentError || rethrow()
        throw(invalid("usage", sprint(showerror, e)))
    end
    model=get(d, "model", nothing)
    model===nothing || (model isa AbstractString && !isempty(model)) || throw(invalid("model", "expected a non-empty string"))
    return Response(;
        id=rid,
        model=model===nothing ? r.model : String(model),
        message=assistant((part,)),
        finish_reason="stop",
        usage,
        provider_data=obj("typesafe"=>obj("answers"=>answers)),
    )
end
function typesafe_error(l, status, body)
    message=strip(body)
    message=isempty(message) ? "HTTP $status" : String(first(message, 500))
    code=nothing
    payload=try
        JSON.parse(body)
    catch
        nothing
    end
    detail=payload isa AbstractDict ? get(payload, "detail", nothing) : nothing
    if detail isa AbstractDict
        code=nonempty_string(get(detail, "error_type", nothing))
        get(detail, "message", nothing) isa AbstractString && (message=String(detail["message"]))
    elseif detail isa AbstractVector && !isempty(detail)
        first_error=detail[1] isa AbstractDict ? detail[1] : obj()
        loc=join((string(x) for x in get(first_error, "loc", []) if x!="body"), ".")
        msg=string(get(first_error, "msg", "validation error"))
        message=isempty(loc) ? msg : "$loc: $msg"
    end
    kw=(; provider=l.provider, provider_code=code, status=Int(status))
    (status==401 || code=="authentication_error") && return AuthError(message; kw..., env_keys=l.access.env_keys)
    status==429 && return RateLimitError(message; kw...)
    status==400 && occursin("unknown model", lowercase(message)) && return UnsupportedModelError(message; kw...)
    status in (400, 422) && return InvalidRequestError(message; kw...)
    status>=500 && return ServerError(message; kw...)
    return map_http_error(status, message; provider=l.provider, provider_code=code)
end
"""
    TypeSafeLM(; api_key=nothing, kw...)

The TypeSafe System One (Jev) client: judgments over declared keys with probabilities
(`provider_classification`); no text generation, no streaming.
"""
TypeSafeLM(; kw...) = ProviderLM("typesafe"; kw...)
