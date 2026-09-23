module LM15UnitfulExt

using LM15
using Unitful: Unitful

# A quantity in a tool result keeps its unit next to its value.
function LM15.tool_value(value::Unitful.Quantity)
    return Dict(
        "value" => LM15.tool_value(Unitful.ustrip(value)), "unit" => string(Unitful.unit(value))
    )
end

end
