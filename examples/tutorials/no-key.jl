# # Answer a tool call without an account
#
# **Goal:** describe a tool, check a proposed call, run the function yourself, and
# build the answer. **Requires:** LM15. **Effect:** local computation only. No key,
# network or charges.

using LM15

# A tool is data: a name, a description the model reads, and a JSON Schema for its
# inputs, written out as in every LM15 language. The function stays yours.
square_tool = FunctionTool(
    name="square",
    description="Square an integer exactly",
    parameters=Dict(
        "type" => "object",
        "properties" => Dict("n" => Dict("type" => "integer")),
        "required" => ["n"],
    ),
)
square(n::Integer) = big(n)^2

# This call is authored here, not returned by a live provider. Its input is JSON
# data; turning it into Julia arguments is your code's job, and so is refusing
# what the schema did not offer.
proposed = tool_call("local-1", "square", Dict("n" => 19))
n = proposed.input["n"]
n isa Integer && !(n isa Bool) || error("n must be an integer")

# The approved operation in this example is a known, side-effect-free arithmetic
# function, so we run it and turn its result into the tool's answer.
result = tool_result(proposed, tool_content(square(n)))
only(result.content).text

# The displayed result should be "361". Nothing sends it anywhere yet.
#
# **Next:** complete a [tool exchange](tools.md), or work with [tables](tables.md).
