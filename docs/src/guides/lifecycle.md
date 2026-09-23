# Know who owns the work

LM15 uses explicit operations rather than hiding network access in read-only
properties. That does not mean every function without `!` is local: `file_delete`,
for example, changes a remote resource. Julia's `!` marks mutation of arguments,
not a universal distinction between free and billable work.

| Operation | Local effect | Remote effect |
|---|---|---|
| Construct `Request`, `Config`, message or tool specification | Validate data | None for built-in construction |
| `text`, `tool_calls`, `done(job)`, `snapshot(view)` | Read existing state | None |
| `resolve(router, model)` | Read rules/catalog | None |
| Resolve/create a provider client | Can inspect profiles/login files | Acquisition mechanisms may be used later |
| `build_request` and other wire builders | Resolve credentials; may read media paths | Credential callbacks may do I/O; no model submission |
| `complete` | Buffer/parse response | Model submission, potentially billable |
| Consume `stream` | Assemble events incrementally | Active provider connection |
| Running your tool function | Your code | Whatever your function explicitly does |
| Files, caches, batches and generation | Materialize metadata/media | Provider resource operations, possible charges |
| `wait!` | Update a job snapshot | Repeated status polling |
| `cancel!(batch_job)` | Update snapshot | Request cancellation; not a billing guarantee |
| `live`, `send!`, `recv` | Manage socket/turn state | Live provider session |

Custom callbacks and type extensions are application code. LM15 cannot promise
they are pure merely because they are used during validation or construction.

## Streams

Prefer `stream(client, req) do rs ... end`. It closes the stream even when your block
throws. It returns your block's value. A manual stream needs `try/finally` and
`close`; `break` alone is insufficient.

A ResponseStream keeps accumulated content for its final answer. Its bounded
network channel is not a promise of constant memory regardless of answer length.
Use one consumer for an individual stream; do not race `events`, `text_chunks` and
`response` from unrelated tasks.

## Live turns and sessions

The `live` block owns the socket. A `TurnView` only owns its consumption state;
closing the view does not close the session. `result(view)` yields at a tool call,
while ordinary iteration can continue across it. See [live sessions](live.md).

## Background jobs

`done(job)` reads the cached snapshot, not the server. `refresh!` polls once.
`wait!` polls until terminal; terminal can mean failed, cancelled or expired rather
than successful. `results` and `result` do not wait automatically.

A wait deadline bounds the polling loop, not a request already in flight. Configure
HTTP timeouts as well. Retrying a submission after uncertainty can create a second
job. Keep the job ID and inspect it instead.
