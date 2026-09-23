# # Keep physical units through a tool call
#
# **Goal:** compute a speed without confusing metres, seconds or other units.
# **Requires:** LM15 and Unitful. **Effect:** local computation; no account.

using LM15, Unitful

# The schema asks for each quantity as a value with a fixed unit.
quantity_schema(unit) = Dict(
    "type" => "object",
    "properties" => Dict(
        "value" => Dict("type" => "number"),
        "unit" => Dict("type" => "string", "const" => string(unit)),
    ),
    "required" => ["value", "unit"],
    "additionalProperties" => false,
)
speed_tool = FunctionTool(
    name="measured_speed",
    description="Compute speed from metres and seconds",
    parameters=Dict(
        "type" => "object",
        "properties" => Dict("distance" => quantity_schema(u"m"), "duration" => quantity_schema(u"s")),
        "required" => ["distance", "duration"],
    ),
)

# Your code turns the model's JSON into quantities, and refuses what it did not
# offer. Incoming unit strings are compared, never evaluated as Julia code.
function quantity(input, unit)
    input isa AbstractDict && Set(keys(input)) == Set(["value", "unit"]) ||
        throw(ArgumentError("expected {\"value\": ..., \"unit\": \"$(unit)\"}"))
    input["unit"] == string(unit) ||
        throw(ArgumentError("expected unit $(unit); convert units explicitly"))
    value = input["value"]
    value isa Real && !(value isa Bool) && isfinite(value) ||
        throw(ArgumentError("value must be a finite number"))
    return Float64(value) * unit
end
function measured_speed(distance, duration)
    duration > 0u"s" || throw(ArgumentError("duration must be positive"))
    return distance / duration
end

speed_call = tool_call("speed-1", "measured_speed", Dict(
    "distance" => Dict("value" => 12, "unit" => "m"),
    "duration" => Dict("value" => 3, "unit" => "s"),
))
distance = quantity(speed_call.input["distance"], u"m")
duration = quantity(speed_call.input["duration"], u"s")
speed_output = tool_result(speed_call, tool_content(measured_speed(distance, duration)))
only(speed_output.content).text

# Loading Unitful lets tool_content keep a quantity's unit next to its value: the
# result is 4.0 with the unit for metres per second, not a unitless 4.0. The exact
# printed spelling of compound units belongs to Unitful.
#
# ## Wrong units are not guessed
#
# Convert centimetres to metres explicitly in application code, or offer a separate
# operation that takes centimetres. A model-supplied "cm" is refused here.
try
    quantity(Dict("value" => 1200, "unit" => "cm"), u"m")
catch error
    error isa ArgumentError || rethrow()
    showerror(stdout, error)
    println()
end

# Units also survive inside named records, tables and explicit array output.
# Large arrays still need deliberate selection or summarization.
only(array_content([1u"m" 2u"m"])).text

# **Next:** expose a [bounded simulation](simulation.md).
