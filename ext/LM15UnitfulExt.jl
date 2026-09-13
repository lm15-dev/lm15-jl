module LM15UnitfulExt

using LM15
using Unitful: Unitful

# The selected Julia type fixes the unit; incoming text is never evaluated or
# parsed as Julia code. Different units must be converted by the caller explicitly.
function LM15.tool_schema(::Type{Q}) where {T,D,U,Q<:Unitful.Quantity{T,D,U}}
    isconcretetype(Q) || throw(ArgumentError("tool quantities need a concrete value type and unit"))
    return Dict{String,Any}(
        "type" => "object",
        "properties" => Dict(
            "value" => LM15.tool_schema(T),
            "unit" => Dict("type" => "string", "const" => string(Unitful.unit(Q))),
        ),
        "required" => ["value", "unit"],
        "additionalProperties" => false,
    )
end
function LM15.tool_decode(::Type{Q}, value; path="input") where {T,D,U,Q<:Unitful.Quantity{T,D,U}}
    LM15.checked_object(value, (:value, :unit), (:value, :unit), path)
    expected = string(Unitful.unit(Q))
    unit = LM15.tool_decode(String, value["unit"]; path="$path.unit")
    unit == expected ||
        LM15.input_error("$path.unit", "expected unit $expected; convert units explicitly")
    number = LM15.tool_decode(T, value["value"]; path="$path.value")
    return Q(number * Unitful.unit(Q))
end
function LM15.tool_value(value::Unitful.Quantity)
    return Dict(
        "value" => LM15.tool_value(Unitful.ustrip(value)), "unit" => string(Unitful.unit(value))
    )
end

end
