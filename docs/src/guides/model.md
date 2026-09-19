# How LM15 works

There are three distinct kinds of work: creating data, talking to a provider, and
executing your application code. Keeping them separate makes costs and permissions
visible.

```text
Your messages + settings
          │
          ▼
        Request ── complete / stream ──► provider
          ▲                                │
          │                                ▼
   append tool results ◄── your code ◄── Response with a proposed call
                           checks and
                           approval
```

A normal textual response needs no tool step. A tool call only describes an
operation. The model never receives a Julia function object, and LM15 does not
interpret model output as Julia source code.

## Requests and responses

`Request` owns the conversation data for one submission. Keep history explicitly by
copying earlier messages and appending the new user message or tool result.
`Request(old; changes...)` and `Config(old; changes...)` validate shallow copies.
Tuples and immutable structs do not make their nested dictionaries immutable.

A `Response` contains an assistant `Message`, a finish reason, optional usage and
optional provider data. Its message can contain text, images, audio, citations,
reasoning or tool calls. `text` is a convenience for text-representable responses,
not a lossy conversion of every possible answer into a string.

## A stream is another way to receive a response

An event stream carries start metadata, typed deltas and final status. A
`ResponseStream` assembles those events while you consume them. An empty network
read or an exhausted iterator is not a substitute for an end event. Early closing
can leave a partial answer; it must not be labeled complete.

## Tools have two representations

`FunctionTool` is a provider-facing name, description and input schema. `ToolBinding`
is the Julia-side callable, selected argument types and output conversion. A request
accepts a binding as convenience, but serializes only its specification.

`tool_arguments` checks a complete call. `execute_tool` checks again and invokes
exactly one binding. Application policy decides whether to execute, how many calls
to allow, which data to reveal and whether to send another request.

## Local and remote are not interchangeable

- Building messages, reading a response and inspecting a registry are local.
- Provider calls, resource operations and live-session sends/receives perform I/O.
- Building a wire request can read a media path or resolve credentials. A credential
  callback may itself contact an identity service or use a configured CLI.
- A completed local wait does not cancel a remote job. A timeout can leave work and
  billing active.

See [operation ownership](lifecycle.md) and [the API reference](../reference/index.md).
