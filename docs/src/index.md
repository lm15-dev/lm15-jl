# LM15.jl

**Call a model. Keep control of the conversation and your Julia code.**

LM15 sends requests directly to model providers and returns the same Julia request,
response and stream types across supported protocols. There is no LM15 relay.
You decide which data to send, which functions may run, and when to stop.

!!! warning "Documentation draft"
    This manual, its new source help and example files were written without running
    tests or a documentation build. They are not a new verification result.
    See [Status and evidence](project/status.md) for the earlier, version-specific checks.

## Start with a useful task

- **Have a provider key?** [Make your first request](start/first-request.md).
- **No key yet?** [Answer a tool call locally](tutorials/no-key.md). No account or paid call.
- **Working with data?** Start with [tables](tutorials/tables.md),
  [physical units](tutorials/units.md), or [a simulation](tutorials/simulation.md).
- **Looking up a function?** Use the [reference](reference/index.md) or type
  `?complete`, `?Request`, or `?FunctionTool` in Julia.

## A small example

This example is local. It answers a tool call with a Julia function; it does not
ask an online model to choose the call.

```jldoctest
julia> using LM15

julia> square(n::Integer) = big(n)^2;

julia> call = tool_call("example-1", "square", Dict("n" => 19));

julia> output = tool_result(call, tool_content(square(call.input["n"])));

julia> only(output.content).text
"361"
```

A real model can return that call description in a `Response`. Merely receiving or
inspecting it runs nothing. Your application checks it, decides whether to allow
it, and runs the function itself. Learn the
[complete exchange](tutorials/tools.md).

## Find the right kind of page

**Tutorials** walk through a complete result. **Guides** answer a specific question.
**Reference** gives exact call forms and behavior. **Project** pages describe
support, verification, contributing and release work.

The library targets Julia 1.10 and later. Earlier recorded verification used
Julia 1.12.5 on Linux; target platforms are not the same as tested platforms.
Browser execution is not supported. Read [installation](start/installation.md)
and [the support boundaries](project/status.md) before choosing a deployment.
