# # Check, approve and answer a tool call
#
# **Goal:** follow the complete data flow of a function-tool exchange.
# **Requires:** LM15. **Effect:** local computation with a synthetic provider call.
# The separate live-tools.jl script performs the actual, potentially paid requests;
# it is never executed by the docs generator.

using LM15

square_tool = @tool "Square an integer exactly" square(n::Int) = big(n)^2
req = Request("docs-model", user("What is 19 squared?"); tools=square_tool)

# The request contains a FunctionTool description, not the Julia callable.
typeof(only(req.tools))

# A provider could propose this call. Here we construct it explicitly so the
# tutorial works without an account. Do not pretend this is a live model response.
proposed = tool_call("call-1", "square", Dict("n" => 19))
first_answer = Response(model="docs-model", message=assistant(proposed),
    finish_reason="tool_call")

calls = tool_calls(first_answer)
length(calls) == 1 || error("This tutorial allows exactly one call")
arguments = tool_arguments(square_tool, only(calls))

# This arithmetic operation is the one we deliberately approve. For an operation
# that writes files, sends data or spends money, ask the user or an application
# policy for approval before this explicit execution step.
output = execute_tool(square_tool, only(calls))
only(output.content).text

follow_up = Request(req; messages=(
    req.messages...,
    first_answer.message,
    tool_message(output),
))
length(follow_up.messages)

# The next live step is `complete(client, follow_up)`. It sends the result back to
# the provider for a final answer. No request is submitted merely by constructing
# follow_up. The [live example](../downloads/live-tools.jl) includes both requests,
# an explicit confirmation prompt and a two-request stopping rule.
#
# ## Failures are not permission to retry
#
# Incorrect names, unknown arguments, missing required arguments, overflow and
# lossy conversions fail before the bound function is invoked. An error *inside*
# the function or while encoding its output may occur after a side effect. Do not
# automatically retry it or report fabricated success.
#
# **Next:** [conversion rules](../guides/conversions.md),
# [extending codecs](../guides/extensions.md), and [data privacy](../guides/privacy.md).
