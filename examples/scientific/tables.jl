# # Summarize a table without uploading it all
#
# **Goal:** return a useful statistic or selected rows while keeping the full table
# local. **Requires:** LM15, DataFrames, Tables and Statistics.
# **Effect:** local computation only. The calls are explicitly authored examples,
# not requests captured from a model. This refactored source has not been run in
# the documentation-writing pass.

using LM15, DataFrames, Tables, Statistics

const measurements = DataFrame(trial=[1, 2, 3], height=[1.5, 2.0, 2.5])

# Offer a specific operation, not a general expression evaluator. Only the height
# column is allowed; a model-provided string never becomes Julia source code.
summary_tool = @tool "Summarize one measured column" function summarize_column(column::String)
    column == "height" || throw(ArgumentError("choose the height column"))
    values = measurements[!, column]
    return (count=length(values), mean=mean(values), minimum=minimum(values), maximum=maximum(values))
end

summary_call = tool_call("table-1", "summarize_column", Dict("column" => "height"))
summary_output = execute_tool(summary_tool, summary_call)
only(summary_output.content).text

# The JSON text contains count 3 and mean 2.0. That is the result that could be sent
# back in a tool message—not the whole source table.
#
# ## Select rows explicitly
#
# A table is not automatically encoded by tool_content. Choose table_content as
# the output conversion and put a domain limit on the operation itself.
rows_tool = @tool "Read at most three measurement rows" function measurement_rows(count::Int)
    0 <= count <= nrow(measurements) || throw(ArgumentError("count must be between zero and three"))
    return view(measurements, 1:count, :)
end
rows_tool = tool(rows_tool; output=table_content)
rows_output = execute_tool(rows_tool, tool_call("table-2", "measurement_rows", Dict("count" => 2)))
only(rows_output.content).text

# table_content also has a default max_rows=100. Exceeding it is an error, not an
# instruction to silently truncate. Choose a subset or summary before encoding.
#
# ## Missing is a decision
#
# Julia's missing does not automatically become JSON null. Here the application
# explicitly chooses that representation for a tiny separate table.
with_missing_example = DataFrame(x=[1, missing])
missing_policy(x) = ismissing(x) ? nothing : tool_value(x)
only(table_content(with_missing_example; cell=missing_policy)).text

# Returning a summary can still reveal sensitive information. Approval and data
# policy belong to the application. Keep identifiers and unrelated columns out of
# the offered operation. See [privacy](../guides/privacy.md).
#
# **Next:** preserve [units](units.md), or read the [conversion rules](../guides/conversions.md).
