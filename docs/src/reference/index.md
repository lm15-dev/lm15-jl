# Function and type reference

Use this section to look up a particular operation. For a complete task, start with
the tutorials. Every exported name has help, and the manual build fails if one is missing from these pages.

- [Requests and content](content.md): messages, settings, response access and JSON.
- [Function tools](tools.md): specifications, bindings, input/output conversion.
- [Providers and routing](providers.md): direct clients, models, compatibility.
- [Authentication](auth.md): credential values, source selection, storage and login.
- [Streams and live sessions](streams.md): events, assembly and connection ownership.
- [Resources and jobs](resources.md): files, caches, media generation and polling.
- [Errors](errors.md): exception families and recovery meaning.
- [Advanced interfaces](advanced.md): transports, wire mapping and migration helpers.

Most names also work in Julia help mode, for example `?complete` and `?FunctionTool`.
The documented defaults are source-level API facts, not guarantees that every
provider/model accepts them. Operations can fail after a provider has begun work;
read effect and ownership notes before retrying.

## All documented names

```@index
```

## The module

```@docs
LM15
```
