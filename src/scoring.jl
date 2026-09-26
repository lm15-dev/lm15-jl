# Judgments by candidate-sequence likelihood (MAP-14 §4). A Chat Completions server that
# scores named tokens (compat token_scoring = "logprob_token_ids": vLLM ≥ 0.29,
# receipted 2026-09-17) delivers a distribution over the declared keys of every judgment:
#   1. /tokenize renders each key as the assistant's answer after a prefill, so the
#      server's chat template is honoured; the terminator token is appended, so the
#      paths are prefix-free (two calls per key, one per judgment prefill);
#   2. ONE /v1/completions call carries every trie node as a prompt of token ids,
#      max_tokens 1, logprob_token_ids = the union of child tokens;
#   3. raw log-probs sum along each path; one normalisation over the key set; the raw
#      mass on the key set is `coverage` in provider_data.
# Ordinary (non-judgment) properties of a mixed schema are answered by one extra
# structured-output call and merged; a server that drops logprob_token_ids answers 200
# without the ids, which `required` refuses and `if_available` records.

const JUDGMENT_PREFILL = "Answer:"
scores_named_tokens(l, r) = l.dialect == "openai-chat" && effective_compat(l, r).token_scoring == "logprob_token_ids"
judgments_via_token_scoring(l, r::Request) = scores_named_tokens(l, r) &&
    r.config.probabilities in ("if_available", "required") && !isempty(request_judgments(r))

function judgment_ask(j::Judgment)
    lines = String[]
    for k in j.keys
        label, desc = get(j.titles, k, nothing), get(j.descriptions, k, nothing)
        tail = label !== nothing && desc !== nothing ? ": $label - $desc" :
            (label !== nothing || desc !== nothing) ? ": $(something(label, desc))" : ""
        push!(lines, "- $k$tail")
    end
    return "$(something(j.instruction, j.name))\nOptions:\n" * join(lines, "\n") *
        "\nAnswer with the option only, spelled exactly as listed."
end
# The conversation, the judgment's question as a final user turn, and the answer so far.
function judgment_messages(l, r::Request, j::Judgment, answer)
    messages = collect(Any, openai_payload(l, reconstruct(r; config=Config()); chat=true)["messages"])
    push!(messages, obj("role" => "user", "content" => judgment_ask(j)))
    push!(messages, obj("role" => "assistant", "content" => answer))
    return messages
end
function judgment_tokenize_request(l, model, messages; continue_final)
    root = replace(l.base_url, r"/v1$" => "")
    return emit(l; url=root * "/tokenize", endpoint="tokenize", model,
        payload=obj("model" => model, "messages" => messages, "add_generation_prompt" => false,
            "continue_final_message" => continue_final))
end
judgment_reply_error(l, response, detail) = attach_error_metadata(
    GenericProviderError("malformed judgment reply: $detail"; provider=l.provider, status=response.status), response)
function judgment_tokens(l, response)
    d = reply_json(l, response)
    tokens = d isa AbstractDict ? get(d, "tokens", nothing) : nothing
    tokens isa AbstractVector && !isempty(tokens) && all(t -> t isa Integer && !(t isa Bool) && t >= 0, tokens) ||
        throw(judgment_reply_error(l, response, "tokenize reply carries no non-negative integer token list"))
    return Int[t for t in tokens]
end
"""Each key's token path after the prefill, terminator included."""
function judgment_paths(provider, prefix, keys)
    paths = OrderedDict{String,Vector{Int}}()
    for (key, (open_tokens, closed_tokens)) in keys
        (length(open_tokens) >= length(prefix) && open_tokens[1:length(prefix)] == prefix &&
            length(closed_tokens) >= length(open_tokens) && closed_tokens[1:length(open_tokens)] == open_tokens) ||
            refuse(provider, "config.response_format",
                "key $(repr(key)) does not tokenize as an extension of the prefill in this chat template; candidate-sequence scoring cannot place it (rename the key or use a provider that classifies natively)")
        body = open_tokens[(length(prefix) + 1):end]
        terminator = closed_tokens[(length(open_tokens) + 1):min(end, length(open_tokens) + 1)]
        (isempty(body) || isempty(terminator)) && refuse(provider, "config.response_format",
            "key $(repr(key)) yields no scorable tokens (empty key or no end-of-turn token in the template)")
        paths[key] = vcat(body, terminator)
    end
    return paths
end
function judgment_nodes(paths)
    nodes = OrderedDict{Vector{Int},Set{Int}}()
    for seq in values(paths), i in eachindex(seq)
        push!(get!(nodes, seq[1:(i - 1)], Set{Int}()), seq[i])
    end
    return nodes
end
"""Harmful omissions are refusals, not silent scoring adaptations."""
function validate_judgment_scoring(l, r::Request)
    no(feature, reason) = refuse(l.provider, feature,
        "$reason; use generated JSON with probabilities=\"off\" or a separate scoring request")
    c = r.config
    isempty(r.tools) || no("tools", "candidate scoring cannot execute tools; the program may depend on their results")
    c.tool_choice === nothing || no("config.tool_choice", "candidate scoring cannot preserve tool/action semantics")
    c.cache !== nothing && c.cache.resource !== nothing &&
        no("config.cache.resource", "candidate scoring cannot read a stored cache object; omitting it would lose prompt content")
    ext = something(c.extensions, obj())
    n = get(ext, "n", nothing)
    n isa Real && !(n isa Bool) && n > 1 && no("config.extensions.n", "n > 1 has no canonical multiple-response representation")
    for name in (:store, :user_id, :service_tier)
        getfield(c, name) === nothing || no("config.$name",
            "measurement endpoints have no established mapping for this privacy, safety or billing control")
    end
    if c.cache !== nothing
        c.cache.mode == "off" && no("config.cache.mode", "measurement endpoints cannot guarantee cache writes are disabled")
        c.cache.retention === nothing || no("config.cache.retention", "measurement endpoints cannot preserve cache lifetime and billing intent")
    end
    harmless = ("temperature", "top_p", "top_k", "seed", "frequency_penalty", "presence_penalty")
    for (name, value) in ext
        numeric = (value isa Integer && !(value isa Bool)) || (value isa AbstractFloat && isfinite(value))
        ((name == "n" && numeric && value == 1) || (name in harmless && numeric)) && continue
        no("config.extensions.$name", "unknown measurement extension semantics; dropping it could lose privacy, money or action controls")
    end
end
judgment_mixed(r::Request) = !isempty(non_judgment_properties(r.config.response_format["schema"], request_judgments(r)))
generated_request(r::Request) = reconstruct(r; config=reconstruct(r.config; probabilities="off"))
function judgment_stream_policy(l, r::Request)
    r.config.probabilities == "required" && refuse(l.provider, "config.probabilities",
        "candidate scoring produces a non-streamable DataPart; use complete() for required probabilities or a separate scoring request")
    adapt!("config.probabilities", "dropped",
        "stream() uses generated JSON, not candidate scoring; the answer carries an unmeasured pick only; use complete() for scoring";
        asked=r.config.probabilities)
end
# The offline preflight shared by plan and complete: refusals and records, no I/O.
function judgment_adaptations(l, r::Request; policy=l.adaptations)
    validate_judgment_scoring(l, r)
    _, records = collecting(policy, l.provider) do
        mixed = judgment_mixed(r)
        mixed && adapt!("config.response_format", "client_side",
            "an additional structured-output call answers ordinary properties, which are never scored")
        config = to_dict(r.config)
        for name in ("max_tokens", "temperature", "top_p", "top_k", "stop", "seed", "frequency_penalty",
            "presence_penalty", "reasoning", "logprobs", "cache", "extensions")
            haskey(config, name) && adapt!("config.$name", "dropped",
                "candidate likelihood measures unmodified next-token probabilities with max_tokens=1 per trie node; this generation setting has no measurement slot (any generated JSON call still uses its usual mapping)";
                asked=config[name])
        end
        mixed && openai_payload(l, generated_request(r); chat=true)
        openai_payload(l, reconstruct(r; config=Config()); chat=true)
        nothing
    end
    return records
end
function judgment_scores(l, response, n_prompts)
    d = reply_json(l, response)
    choices = d isa AbstractDict ? get(d, "choices", nothing) : nothing
    (choices isa AbstractVector && length(choices) == n_prompts &&
        all(c -> c isa AbstractDict && get(c, "index", nothing) isa Integer && !(c["index"] isa Bool), choices) &&
        Set(c["index"] for c in choices) == Set(0:(n_prompts - 1))) ||
        throw(judgment_reply_error(l, response, "choices must contain every prompt index exactly once"))
    out = Dict{Int,Float64}[]
    for choice in sort(choices; by=c -> c["index"])
        logprobs = something(get(choice, "logprobs", nothing), obj())
        logprobs isa AbstractDict || throw(judgment_reply_error(l, response, "logprobs must be an object or null"))
        top_list = get(logprobs, "top_logprobs", nothing)
        top = if top_list === nothing || (top_list isa AbstractVector && isempty(top_list))
            obj()
        elseif top_list isa AbstractVector && length(top_list) == 1 && (top_list[1] === nothing || top_list[1] isa AbstractDict)
            something(top_list[1], obj())
        else
            throw(judgment_reply_error(l, response, "expected one top_logprobs object"))
        end
        scores = Dict{Int,Float64}()
        for (token, value) in top
            m = match(r"^token_id:([0-9]+)$", token)
            m === nothing && continue
            # -inf is a zero-likelihood token; NaN, positive log-probs and booleans are not measurements.
            value isa Real && !(value isa Bool) && !isnan(value) && value <= 0 ||
                throw(judgment_reply_error(l, response, "invalid log probability for $token"))
            scores[parse(Int, m.captures[1])] = Float64(value)
        end
        push!(out, scores)
    end
    raw_usage = something(get(d, "usage", nothing), obj())
    raw_usage isa AbstractDict || throw(judgment_reply_error(l, response, "usage must be an object or null"))
    for name in ("prompt_tokens_details", "completion_tokens_details")
        v = get(raw_usage, name, nothing)
        v === nothing || v isa AbstractDict || throw(judgment_reply_error(l, response, "$name must be an object or null"))
    end
    usage = try
        usage_from("openai-chat", raw_usage)
    catch e
        e isa ArgumentError || rethrow()
        throw(judgment_reply_error(l, response, "invalid usage"))
    end
    model = get(d, "model", nothing)
    model === nothing || (model isa AbstractString && !isempty(model)) ||
        throw(judgment_reply_error(l, response, "model must be a non-empty string"))
    return out, usage, model
end
sum_usage_fields(a::Usage, b::Usage) = Usage(; (n => (getfield(a, n) === nothing || getfield(b, n) === nothing ? nothing :
    getfield(a, n) + getfield(b, n)) for n in fieldnames(Usage))...)
function judgment_fold(l, r::Request, per, usage, model, n_nodes, calls)
    value = JSONObject()
    probabilities = JSONObject()
    coverage = JSONObject()
    for (j, paths, table) in per
        raw = OrderedDict(key => sum(table[seq[1:(i - 1)]][seq[i]] for i in eachindex(seq)) for (key, seq) in paths)
        any(isfinite, values(raw)) || throw(GenericProviderError(
            "judgment $(repr(j.name)) has zero likelihood for every declared key; cannot normalize"; provider=l.provider))
        coverage[j.name] = sum(exp, values(raw))
        dist = normalize_logprobs(raw)
        ordered_dist = JSONObject(k => dist[k] for k in j.keys)
        probabilities[j.name] = ordered_dist
        best = j.keys[argmax([dist[k] for k in j.keys])]
        value[j.name] = j.kind == "boolean" ? best == "true" : is_ordered(j) ? parse(Int, best) : best
    end
    part = DataPart(; value, probabilities, method="candidate_sequence_likelihood")
    return Response(; model=something(model, r.model), message=assistant((part,)), finish_reason="stop", usage,
        provider_data=obj("coverage" => coverage, "judgments" => obj("nodes" => n_nodes, "tokenize_calls" => calls,
            "method" => "candidate_sequence_likelihood")))
end
function judgment_generated_value(l, r::Request, response::Response; unmeasured=false)
    try
        response.finish_reason == "stop" || throw(ArgumentError("generated judgment answer did not finish completely"))
        texts = [p for p in response.message.parts if p isa TextPart]
        length(texts) == 1 || throw(ArgumentError("generated judgment answer needs one JSON object"))
        t = only(texts)
        value = strict_json_parse(t.text)
        value isa AbstractDict || throw(ArgumentError("generated judgment answer needs a JSON object"))
        schema = r.config.response_format["schema"]
        measured = request_judgments(r)
        for name in get(schema, "required", ())
            (unmeasured || !haskey(measured, name)) && !haskey(value, name) &&
                throw(ArgumentError("generated answer is missing required property $(repr(name))"))
        end
        part = DataPart(; value, continuation=t.continuation)
        parts = Tuple(p === t ? part : p for p in response.message.parts)
        return reconstruct(response; message=reconstruct(response.message; parts), adaptations=())
    catch e
        e isa ArgumentError || e isa ErrorException || rethrow()
        throw(GenericProviderError("malformed generated judgment JSON: $(e isa ArgumentError ? e.msg : sprint(showerror, e))";
            provider=l.provider))
    end
end
function judgment_generate(l, r::Request; unmeasured=false)
    gen = generated_request(r)
    wire, built = build_request_adapted(l, gen)
    plain = reconstruct(gen; config=reconstruct(gen.config; response_format=nothing))
    if client_side_stop(built)
        # Preserve the ordinary driver's close-at-stop billing semantics.
        return judgment_generated_value(l, gen, materialize_response(stream(l, gen), plain); unmeasured), built
    end
    reply = send_request(l, wire)
    body = reply_json(l, reply)
    try
        body isa AbstractDict && get(body, "error", nothing) isa AbstractDict && parse_response(l, plain, reply)  # in-band provider errors
        choices = body isa AbstractDict ? get(body, "choices", nothing) : nothing
        (choices isa AbstractVector && length(choices) == 1 && choices[1] isa AbstractDict &&
            get(choices[1], "finish_reason", nothing) == "stop") ||
            throw(judgment_reply_error(l, reply, "generated judgment reply needs one complete choice with finish_reason='stop'"))
        return judgment_generated_value(l, gen, parse_response(l, plain, reply); unmeasured), built
    catch e
        e isa ProviderError && rethrow(attach_error_metadata(e.status === nothing ? with_metadata(e; status=reply.status) : e, reply))
        rethrow()
    end
end
function judgment_merge(measured::Response, generated::Response)
    original = first(p for p in generated.message.parts if p isa DataPart)
    scored = first(p for p in measured.message.parts if p isa DataPart)
    part = DataPart(; value=merge(JSONObject(original.value), scored.value), probabilities=scored.probabilities,
        method=scored.method, continuation=original.continuation)
    parts = Tuple(p === original ? part : p for p in generated.message.parts)
    return reconstruct(measured; id=generated.id, finish_reason=generated.finish_reason,
        message=reconstruct(generated.message; parts), usage=sum_usage_fields(measured.usage, generated.usage),
        provider_data=merge(something(measured.provider_data, obj()), obj("scoring_usage" => to_dict(measured.usage),
            "generated_response" => to_dict(generated; include_provider_data=true))))
end
# The server answered 200 without the requested token ids: it dropped logprob_token_ids
# (receipted on vLLM 0.25.1). `required` refuses; `if_available` answers by structured
# output instead and records it.
function judgment_unmeasured(l, r::Request, adaptations, usage)
    r.config.probabilities == "required" && refuse(l.provider, "config.probabilities",
        "config.probabilities='required' but this server ignored logprob_token_ids (vLLM < 0.29?); no distribution can be measured here")
    _, dropped = collecting(l.adaptations, l.provider) do
        adapt!("config.probabilities", "dropped",
            "the server accepted the request and returned no log-probs for the requested token ids (logprob_token_ids ignored); answered by structured output instead";
            asked=r.config.probabilities)
    end
    response, built = judgment_generate(l, r; unmeasured=true)
    response = reconstruct(response; usage=sum_usage_fields(usage, response.usage),
        provider_data=merge(something(response.provider_data, obj()), obj("scoring_usage" => to_dict(usage))))
    records = (adaptations..., dropped..., (a for a in built if !(a in adaptations))...)
    return reconstruct(response; adaptations=visible_adaptations(l, records))
end
function judgment_complete(l, r::Request)
    adaptations = judgment_adaptations(l, r)
    found = request_judgments(r)
    tokenized = Any[]
    calls = 0
    for (_, j) in found
        send_tokens(msgs, cont) = (calls += 1; judgment_tokens(l, send_request(l, judgment_tokenize_request(l, r.model, msgs; continue_final=cont))))
        prefix = send_tokens(judgment_messages(l, r, j, JUDGMENT_PREFILL), true)
        keys = OrderedDict{String,Tuple{Vector{Int},Vector{Int}}}()
        for k in j.keys
            msgs = judgment_messages(l, r, j, "$JUDGMENT_PREFILL $k")
            keys[k] = (send_tokens(msgs, true), send_tokens(msgs, false))
        end
        push!(tokenized, (j, judgment_paths(l.provider, prefix, keys), prefix))
    end
    prompts = Vector{Int}[]
    meta = Tuple{Int,Vector{Int}}[]
    union_ids = Set{Int}()
    nodes_per = Any[]
    for (index, (j, paths, prefix)) in enumerate(tokenized)
        nodes = judgment_nodes(paths)
        push!(nodes_per, nodes)
        for (node_prefix, children) in nodes
            push!(prompts, vcat(prefix, node_prefix))
            push!(meta, (index, node_prefix))
            union!(union_ids, children)
        end
    end
    score_wire = emit(l; url=l.base_url * "/completions", endpoint="completions", model=r.model,
        payload=obj("model" => r.model, "prompt" => prompts, "max_tokens" => 1, "temperature" => 1.0, "logprobs" => 0,
            "return_tokens_as_token_ids" => true, "logprob_token_ids" => sort!(collect(union_ids))))
    reply = send_request(l, score_wire)
    scores, usage, model = judgment_scores(l, reply, length(prompts))
    tables = [Dict{Vector{Int},Dict{Int,Float64}}() for _ in tokenized]
    for ((index, node_prefix), got) in zip(meta, scores)
        children = nodes_per[index][node_prefix]
        all(t -> haskey(got, t), children) || return judgment_unmeasured(l, r, adaptations, usage)
        tables[index][node_prefix] = Dict(t => got[t] for t in children)
    end
    per = [(j, paths, tables[i]) for (i, (j, paths, _)) in enumerate(tokenized)]
    response = try
        judgment_fold(l, r, per, usage, model, length(prompts), calls)
    catch e
        e isa ProviderError && rethrow(attach_error_metadata(with_metadata(e; status=reply.status), reply))
        rethrow()
    end
    judgment_mixed(r) && (response = judgment_merge(response, first(judgment_generate(l, r))))
    return reconstruct(response; adaptations=visible_adaptations(l, adaptations))
end

# Response conveniences for judgment answers.
"""
    data_part(response) -> DataPart or nothing

The response's first DataPart (a judgment answer), if any.
"""
data_part(r::Response) = (i = findfirst(p -> p isa DataPart, r.message.parts); i === nothing ? nothing : r.message.parts[i])
"""
    data(response) -> the answer's JSON value, or nothing

The value of the response's DataPart (a judgment answer).
"""
data(r::Response) = (p = data_part(r); p === nothing ? nothing : p.value)
"""
    probabilities(response) -> Dict or nothing

The measured distribution per judgment, when one was measured; its method is
`data_part(response).method`.
"""
probabilities(r::Response) = (p = data_part(r); p === nothing ? nothing : p.probabilities)
