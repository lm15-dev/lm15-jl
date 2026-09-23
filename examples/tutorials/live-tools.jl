# Live example: two provider requests at most, with explicit tool approval.
# Never included or executed by the documentation build. This source is unverified.
# Set LM15_RUN_LIVE_EXAMPLES=yes and OPENAI_API_KEY only when you intend to send
# data and potentially incur charges. LM15_MODEL chooses an account-supported model.
using LM15

get(ENV, "LM15_RUN_LIVE_EXAMPLES", "no") == "yes" ||
    error("Live examples are disabled; explicitly authorize them before running")

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
client = OpenAILM(api_key=ENV["OPENAI_API_KEY"])
model = get(ENV, "LM15_MODEL", "gpt-4.1-mini")
req = Request(model, user("Use square to compute 19 squared, then explain the result.");
    tools=square_tool, config=Config(max_tokens=120))
first_answer = complete(client, req)
calls = tool_calls(first_answer)

if isempty(calls)
    println(something(text(first_answer), "No text answer; inspect the response parts."))
else
    length(calls) == 1 || error("Refusing more than one function call")
    call = only(calls)
    call.name == "square" || error("Unknown tool: ", call.name)
    n = get(call.input, "n", nothing)
    n isa Integer && !(n isa Bool) || error("n must be an integer")
    println("Requested square argument: ", n)
    print("Run this known local arithmetic function? Type yes: ")
    readline() == "yes" || error("Tool execution declined")
    output = tool_result(call, tool_content(square(n)))
    follow_up = Request(req; messages=(req.messages..., first_answer.message, tool_message(output)))
    final_answer = complete(client, follow_up)
    isempty(tool_calls(final_answer)) || error("Request limit reached; no further tools executed")
    println("Finish reason: ", final_answer.finish_reason)
    println(something(text(final_answer), "Non-text response; inspect the response parts."))
end
