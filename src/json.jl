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
