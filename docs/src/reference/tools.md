# Function tools

```@meta
CurrentModule = LM15
```

Describe, check and execute are separate operations. Receiving a call does not
approve it. See the [tutorial](../tutorials/tools.md) and
[conversion guide](../guides/conversions.md).

## Specifications and calls

```@docs
Tool
FunctionTool
BuiltinTool
ToolChoice
ToolBinding
ToolCallPart
ToolResultPart
ToolCallInfo
@tool
tool
tool_call
tool_calls
tool_result
tool_message
```

## Explicit execution

```@docs
tool_arguments
execute_tool
ToolInputError
```

## Type and presentation extensions

The Unitful extension supplies quantity methods when Unitful is loaded. Tables
supplies table_content when Tables is loaded. Neither package is a mandatory LM15
runtime dependency. See [extension agreements](../guides/extensions.md).

```@docs
tool_schema
tool_decode
tool_value
tool_content
array_content
table_content
```
