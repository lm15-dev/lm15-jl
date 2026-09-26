# Keep one JSON boundary. Integers larger than machine Int remain exact; canonical
# typed counters are range-checked separately. Ordered objects also preserve JWT
# claim order and user-authored opaque payloads.
module JSON
import ..JSONBackend
import ..OrderedDict
parse(s::AbstractString) = JSONBackend.parse(s; dicttype=OrderedDict{String,Any}, inttype=BigInt)
serialize(x) = JSONBackend.json(x)
end

const JsonObject = AbstractDict{String}
const JSONObject = OrderedDict{String,Any}
obj(pairs::Pair...) = JSONObject(pairs)
function empty_optional(x)
    return x === nothing ||
           (x isa Union{AbstractString,Tuple,AbstractVector,AbstractDict} && isempty(x))
end
clean(d::AbstractDict) = JSONObject(k => v for (k, v) in d if !empty_optional(v))

# INV-055: text with no valid UTF-8 form (a lone surrogate, stray bytes) cannot reach any
# provider; it is refused before the wire as the input error it is, naming where.
function invalid_text(s::AbstractString)
    i = findfirst(c -> !isvalid(c), s)
    i === nothing && return nothing
    c = s[i]
    Base.ismalformed(c) && return "bytes " * join((string(b; base=16, pad=2) for b in codeunits(string(c))), " ")
    return "U+" * uppercase(string(codepoint(c); base=16, pad=4))
end
function check_wire_text(value)
    if value isa AbstractString
        bad = isvalid(value) ? nothing : invalid_text(value)
        bad === nothing || throw(ArgumentError(
            "request contains text that is not valid Unicode ($bad; a lone surrogate or stray bytes), which no provider can receive; repair the text first, e.g. with `isvalid` and `Base.replace` of invalid characters"))
    elseif value isa AbstractDict
        for (k, v) in value
            check_wire_text(k)
            check_wire_text(v)
        end
    elseif value isa AbstractVector
        foreach(check_wire_text, value)
    end
    return value
end
"""The canonical bytes of an outgoing JSON body: strict JSON, valid Unicode (INV-055)."""
wire_json(value) = JSON.serialize(check_wire_text(check_json(value)))
function check_json(value)
    active = IdDict{Any,Nothing}()
    function visit(x)
        if x === nothing || x isa Union{Bool,AbstractString,Integer}
            return nothing
        elseif x isa AbstractFloat
            isfinite(x) || throw(ArgumentError("JSON numbers must be finite"))
        elseif x isa Union{AbstractDict,AbstractVector}
            haskey(active, x) && throw(ArgumentError("JSON containers must not contain cycles"))
            active[x] = nothing
            if x isa AbstractDict
                for (k, v) in x
                    k isa AbstractString || throw(ArgumentError("JSON object keys must be strings"))
                    visit(v)
                end
            else
                foreach(visit, x)
            end
            delete!(active, x)
        else
            throw(ArgumentError("value has no strict JSON representation"))
        end
    end
    visit(value)
    return value
end
