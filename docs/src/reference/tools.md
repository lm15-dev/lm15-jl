# Function tools

```@meta
CurrentModule = LM15
```

A tool is data: a name, a description and a JSON Schema, written out. LM15 carries
calls and answers; checking and running a call is your code. Receiving a call does
not approve it. See the [tutorial](../tutorials/tools.md) and
[conversion guide](../guides/conversions.md).

## Specifications and calls

```@docs
Tool
FunctionTool
BuiltinTool
ToolChoice
ToolCallPart
ToolResultPart
ToolCallInfo
tool_call
tool_calls
tool_result
tool_message
```

## Result presentation and extensions

The Unitful extension supplies quantity result values when Unitful is loaded. Tables
supplies table_content when Tables is loaded. Neither package is a mandatory LM15
runtime dependency. See [extension agreements](../guides/extensions.md).

```@docs
tool_value
tool_content
array_content
table_content
```
