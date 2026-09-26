# Judgments: declared keys in, a distribution out (MAP-14,
# changes/2026-09-17-judgments.md). A judgment is a top-level property of a
# `json_schema` response_format that declares its answer set: a boolean, a string
# `enum`/`anyOf`-of-`const`, or an ordered integer `enum`/`anyOf`-of-`const`
# `0..n-1`. This file reads the convention off a schema (§1), rewrites judgment
# properties for the two wires that need it (§2), folds a model's JSON text into a
# DataPart (§3), and offers the helpers that emit the convention. No network here.

const MAX_ORDERED_LEVELS = 10  # Jev's Score ceiling
const MAX_CHOICE_KEYS = 255    # Jev's Choice ceiling
const JUDGMENT_METHODS = ("provider_classification", "candidate_sequence_likelihood")
const PROBABILITY_POLICIES = ("off", "if_available", "required")

"""
    Judgment

One declared judgment read off a schema property: `name`, `kind` (`"boolean"`,
`"choice"` or `"ordered"`), its answer `keys` (strings; `"true"`/`"false"`, the choice
keys, or `"0"`…`"n-1"`), the property's `instruction` (its description), and the per-key
`descriptions` and `titles`.
"""
struct Judgment
    name::String
    kind::String
    keys::Vector{String}
    instruction::Maybe{String}
    descriptions::Dict{String,Maybe{String}}
    titles::Dict{String,Maybe{String}}
end
is_ordered(j::Judgment) = j.kind == "ordered"
nonempty_string(v) = v isa AbstractString && !isempty(v) ? String(v) : nothing

function const_branches(prop)
    branches = get(prop, "anyOf", nothing)
    branches isa AbstractVector && !isempty(branches) || return nothing
    all(b -> b isa AbstractDict && haskey(b, "const"), branches) || return nothing
    return branches
end
function judgment_of(name, prop)
    prop isa AbstractDict || return nothing
    instruction = nonempty_string(get(prop, "description", nothing))
    if get(prop, "type", nothing) == "boolean"
        return Judgment(name, "boolean", ["true", "false"], instruction,
            Dict{String,Maybe{String}}("true" => nothing, "false" => nothing), Dict{String,Maybe{String}}())
    end
    enum = get(prop, "enum", nothing)
    branches = const_branches(prop)
    descs = Dict{String,Maybe{String}}()
    titles = Dict{String,Maybe{String}}()
    if enum isa AbstractVector && !isempty(enum) && branches === nothing
        values = enum
    elseif branches !== nothing && enum === nothing
        values = [b["const"] for b in branches]
        for b in branches
            k = string(b["const"])
            descs[k] = nonempty_string(get(b, "description", nothing))
            titles[k] = nonempty_string(get(b, "title", nothing))
        end
    else
        return nothing
    end
    if all(v -> v isa AbstractString && !isempty(v), values)
        get(prop, "type", nothing) in (nothing, "string") || return nothing
        keys = String[v for v in values]
        length(unique(keys)) == length(keys) || return nothing
        return Judgment(name, "choice", keys, instruction,
            Dict{String,Maybe{String}}(k => get(descs, k, nothing) for k in keys),
            Dict{String,Maybe{String}}(k => get(titles, k, nothing) for k in keys))
    end
    if all(v -> v isa Integer && !(v isa Bool), values)
        get(prop, "type", nothing) in (nothing, "integer") || return nothing
        (collect(values) == collect(0:(length(values) - 1)) && length(values) >= 2) || return nothing
        keys = [string(v) for v in values]
        return Judgment(name, "ordered", keys, instruction,
            Dict{String,Maybe{String}}(k => get(descs, k, nothing) for k in keys),
            Dict{String,Maybe{String}}(k => get(titles, k, nothing) for k in keys))
    end
    return nothing
end
"""
    judgments_in_schema(schema) -> OrderedDict{String,Judgment}

The judgments a JSON schema declares, in property order (MAP-14 §1). Any other
property is ordinary structured output and is absent from the result.
"""
function judgments_in_schema(schema)
    out = OrderedDict{String,Judgment}()
    schema isa AbstractDict && get(schema, "type", nothing) in (nothing, "object") || return out
    props = get(schema, "properties", nothing)
    props isa AbstractDict || return out
    for (name, prop) in props
        name isa AbstractString || continue
        j = judgment_of(String(name), prop)
        j === nothing || (out[String(name)] = j)
    end
    return out
end
"""
    request_judgments(request) -> OrderedDict{String,Judgment}

The judgments a request's `json_schema` response_format declares (empty otherwise).
"""
function request_judgments(r::Request)
    f = r.config.response_format
    f isa AbstractDict && get(f, "type", nothing) == "json_schema" || return OrderedDict{String,Judgment}()
    return judgments_in_schema(get(f, "schema", nothing))
end
function non_judgment_properties(schema, found)
    props = schema isa AbstractDict ? get(schema, "properties", nothing) : nothing
    props isa AbstractDict || return String[]
    return [String(n) for n in keys(props) if !haskey(found, n)]
end

# §2 on a wire that measures nothing: `if_available` records `dropped`;
# `required` refuses before the wire (MAP-13 condition b).
function note_unmeasurable_probabilities(r::Request, provider)
    policy = r.config.probabilities
    (policy === nothing || policy == "off" || isempty(request_judgments(r))) && return nothing
    policy == "required" && refuse(provider, "config.probabilities",
        "config.probabilities='required' but this wire cannot measure a distribution over the declared keys (it returns a pick only); use 'if_available' or a provider that can (typesafe, or a vLLM server that honours logprob_token_ids)")
    adapt!("config.probabilities", "dropped",
        "this wire cannot measure a distribution over the declared keys; the answer carries the pick only";
        asked=policy)
    return nothing
end
deepcopy_json(x::AbstractDict) = JSONObject(String(k) => deepcopy_json(v) for (k, v) in x)
deepcopy_json(x::AbstractVector) = Any[deepcopy_json(v) for v in x]
deepcopy_json(x) = x
"""A judgment property carrying both `type` and `anyOf` has its type moved into every
branch: the Messages wire answers 400 otherwise (receipted). Everything else verbatim."""
function anthropic_schema(schema, found)
    isempty(found) && return schema
    out = deepcopy_json(schema)
    for name in keys(found)
        prop = out["properties"][name]
        branches = get(prop, "anyOf", nothing)
        if haskey(prop, "type") && branches isa AbstractVector
            t = pop!(prop, "type")
            for b in branches
                haskey(b, "type") || (b["type"] = t)
            end
        end
    end
    return out
end
"""Judgment properties go to Gemini as `enum` of their keys, with per-key descriptions
folded into the property description (it ignores `anyOf`/`const`; receipted)."""
function gemini_schema(schema, found)
    isempty(found) && return schema
    out = deepcopy_json(schema)
    for (name, j) in found
        prop = out["properties"][name]
        (j.kind == "boolean" || !haskey(prop, "anyOf")) && continue
        delete!(prop, "anyOf")
        prop["type"] = is_ordered(j) ? "integer" : "string"
        prop["enum"] = is_ordered(j) ? Any[parse(Int, k) for k in j.keys] : Any[k for k in j.keys]
        lines = String[]
        for k in j.keys
            label = get(j.titles, k, nothing)
            desc = get(j.descriptions, k, nothing)
            if label === nothing && desc === nothing
                push!(lines, k)
                continue
            end
            push!(lines, "$k = " * (label !== nothing && desc !== nothing ? "$label: $desc" : something(label, desc)))
        end
        if any(k -> get(j.titles, k, nothing) !== nothing || get(j.descriptions, k, nothing) !== nothing, j.keys)
            head = something(nonempty_string(get(prop, "description", nothing)), "")
            prop["description"] = strip(head * (isempty(head) ? "" : " ") * (is_ordered(j) ? "Levels: " : "Options: ") * join(lines, "; "))
        end
    end
    return out
end
"""The model's JSON object as a DataPart (value only), or `nothing` when the text is not
a JSON object (a truncated answer stays a TextPart)."""
function data_part_from_text(s::AbstractString, found)
    isempty(found) && return nothing
    value = try
        JSON.parse(strip(s))
    catch
        return nothing
    end
    value isa AbstractDict || return nothing
    return DataPart(; value)
end
"""Swap the single text part of a judgment answer for its DataPart (MAP-14 §3)."""
function replace_text_with_data(parts, found)
    isempty(found) && return Tuple(parts)
    texts = [p for p in parts if p isa TextPart]
    length(texts) == 1 || return Tuple(parts)
    t = only(texts)
    part = data_part_from_text(t.text, found)
    part === nothing && return Tuple(parts)
    isempty(t.continuation) || (part = DataPart(; value=part.value, continuation=t.continuation))
    return Tuple(p === t ? part : p for p in parts)
end
fold_judgments(r::Response, request::Request) = fold_judgments(r, request_judgments(request))
function fold_judgments(r::Response, found)
    isempty(found) && return r
    parts = replace_text_with_data(r.message.parts, found)
    parts == r.message.parts && return r
    return reconstruct(r; message=reconstruct(r.message; parts))
end
"""One softmax over log-scores: a normalisation over the key set."""
function normalize_logprobs(scores)
    top = maximum(values(scores))
    weights = Dict(k => exp(v - top) for (k, v) in scores)
    total = sum(values(weights))
    return Dict(k => w / total for (k, w) in weights)
end
"""
    expected_level(distribution)

Σ p·i over an ordered judgment's distribution: computed from the numbers, never stored.
"""
expected_level(distribution) = sum(Float64(p) * parse(Int, string(k)) for (k, p) in distribution)

# §4 helpers that emit the convention.
"""
    choice(instruction, options)

A choice judgment property: `options` is a list of keys or a `key => description` map.
"""
function choice(instruction::AbstractString, options)
    items = options isa AbstractDict ? collect(pairs(options)) : [k => nothing for k in options]
    isempty(items) && throw(ArgumentError("choice needs at least one option"))
    all(p -> first(p) isa AbstractString && !isempty(first(p)), items) ||
        throw(ArgumentError("choice option keys must be non-empty strings"))
    length(unique(first.(items))) == length(items) || throw(ArgumentError("choice option keys must be unique"))
    prop = obj("type" => "string", "description" => String(instruction))
    if all(p -> last(p) === nothing, items)
        prop["enum"] = Any[String(first(p)) for p in items]
    else
        prop["anyOf"] = Any[
            last(p) === nothing ? obj("const" => String(first(p))) : obj("const" => String(first(p)), "description" => String(last(p)))
            for p in items]
    end
    return prop
end
"""
    yes_no(instruction)

A yes/no judgment property (keys `true` and `false`).
"""
yes_no(instruction::AbstractString) = obj("type" => "boolean", "description" => String(instruction))
"""
    score(instruction, levels)

An ordered judgment: levels from low to high, as descriptions or `name => description`.
"""
function score(instruction::AbstractString, levels)
    items = levels isa AbstractDict ? collect(pairs(levels)) : [nothing => d for d in levels]
    length(items) >= 2 || throw(ArgumentError("score needs at least two levels"))
    length(items) <= MAX_ORDERED_LEVELS || throw(ArgumentError("score takes at most $MAX_ORDERED_LEVELS levels"))
    branches = Any[]
    for (i, (name, desc)) in enumerate(items)
        b = obj("const" => i - 1)
        name === nothing || isempty(name) || (b["title"] = String(name))
        desc === nothing || isempty(desc) || (b["description"] = String(desc))
        push!(branches, b)
    end
    return obj("type" => "integer", "description" => String(instruction), "anyOf" => branches)
end
"""
    judgments(; name="judgments", strict=true, properties...)
    judgments(properties::Pair...; name="judgments", strict=true)

A `response_format` declaring the given judgment properties, in order.
"""
function judgments(properties::Pair...; name::AbstractString="judgments", strict::Bool=true)
    isempty(properties) && throw(ArgumentError("judgments needs at least one property"))
    props = JSONObject(String(k) => v for (k, v) in properties)
    schema = obj("type" => "object", "properties" => props, "required" => collect(keys(props)),
        "additionalProperties" => false)
    return obj("type" => "json_schema", "name" => String(name), "strict" => strict, "schema" => schema)
end
judgments(; name::AbstractString="judgments", strict::Bool=true, properties...) =
    judgments((string(k) => v for (k, v) in properties)...; name, strict)
