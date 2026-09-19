# # Keep physical units through a tool call
#
# **Goal:** compute a speed without confusing metres, seconds or other units.
# **Requires:** LM15 and Unitful. **Effect:** local computation; no account.
# This source is a newly written documentation example, not a new test result.

using LM15, Unitful

speed_tool = @tool "Compute speed from metres and seconds" function measured_speed(
    distance::typeof(1.0u"m"), duration::typeof(1.0u"s")
)
    duration > 0u"s" || throw(ArgumentError("duration must be positive"))
    return distance / duration
end

# Loading Unitful activates LM15's quantity conversion. The selected argument type
# fixes the expected unit; incoming unit strings are never evaluated as Julia code.
speed_call = tool_call("speed-1", "measured_speed", Dict(
    "distance" => Dict("value" => 12, "unit" => "m"),
    "duration" => Dict("value" => 3, "unit" => "s"),
))
checked_speed_arguments = tool_arguments(speed_tool, speed_call)
checked_speed_arguments.distance, checked_speed_arguments.duration

speed_output = execute_tool(speed_tool, speed_call)
only(speed_output.content).text

# The result retains value 4.0 and the unit for metres per second. It does not
# silently send a unitless 4.0. The exact printed spelling of compound units belongs
# to Unitful; the tool schema uses the spelling of the declared quantity type.
#
# ## Wrong units are not guessed
#
# Convert centimetres to metres explicitly in application code before creating the
# input object, or expose a separate operation whose declared type is centimetres.
# A model-supplied "cm" is not accepted for the "m" interface.
try
    tool_decode(typeof(1.0u"m"), Dict("value" => 1200, "unit" => "cm"))
catch error
    error isa ToolInputError || rethrow()
    showerror(stdout, error)
    println()
end

# Units also survive inside named records, tables and explicit array output.
# Large arrays still need deliberate selection or summarization.
only(array_content([1u"m" 2u"m"])).text

# **Next:** expose a [bounded simulation](simulation.md).
