# # Run a Julia tool without an account
#
# **Goal:** compute 19 squared and inspect the result that could be sent to a model.
# **Requires:** LM15. **Effect:** local computation only. No key, network or charges.
# This source was written without executing it in the documentation-writing pass.

using LM15

# Define an ordinary Julia function and retain a separate tool binding. BigInt
# arithmetic inside the function avoids overflowing when squaring a valid Int.
square_tool = @tool "Square an integer exactly" function square(n::Int)
    big(n)^2
end

# The original function still works normally. The binding describes its inputs.
square(19)
FunctionTool(square_tool)

# This call is authored here, not returned by a live provider. Its input is JSON
# data; Julia argument conversion happens in tool_arguments.
proposed = tool_call("local-1", "square", Dict("n" => 19))
arguments = tool_arguments(square_tool, proposed)

# Inspect arguments before deciding whether to execute. The approved operation in
# this example is a known, side-effect-free arithmetic function.
output = execute_tool(square_tool, proposed)
only(output.content).text

# The displayed result should be "361". Nothing sends it anywhere yet.
# Invalid input is rejected rather than rounded into a plausible integer.
try
    tool_arguments(square_tool, tool_call("local-2", "square", Dict("n" => 1.5)))
catch error
    error isa ToolInputError || rethrow()
    showerror(stdout, error)
    println()
end

# **Why not infer every Julia method?** A function can have several methods and
# broad input types. @tool captures the typed interface you chose here. Existing
# functions can use the signature-only form, such as `@tool square(n::Int)`.
#
# **Next:** complete a [tool exchange](tools.md), or work with [tables](tables.md).
