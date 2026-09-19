# Submit and monitor background jobs

Batch and video operations can outlive your Julia process. Save their IDs and provider
identity so you can inspect the same work later. The snippets below are application
patterns for remote, potentially billable operations; they are not docs-build examples.

## Batch requests

```julia
using LM15
client = OpenAILM(api_key=ENV["OPENAI_API_KEY"])
model = get(ENV, "LM15_MODEL", "gpt-4.1-mini")
requests = [Request(model, user(prompt); config=Config(max_tokens=80))
    for prompt in ("Explain a tuple.", "Explain an iterator.")]
job = batch(client, requests)
println("Keep this job ID: ", job.info.id)
try
    wait!(job; poll_every=30, timeout=3600)
catch error
    error isa WaitTimeout || rethrow()
    println("Still not terminal; saved snapshot: ", error.snapshot.status)
end
```

`done(job)` reads the current snapshot. `refresh!` polls once. `wait!` polls until a
terminal state, which can be failed, cancelled or expired as well as completed.
A wait timeout does not cancel provider work.

After deciding the job is ready, `results(job)` fetches `BatchEntry` values. Use each
entry's zero-based `index` to match its request, and branch on `outcome`. A successful
entry has `response`; an errored entry has `error`. Do not assume every submitted
request succeeded because the batch itself reached a terminal state.

For the `job` above, fetch output only after deciding its status is appropriate:

```julia
if job.info.status == "completed"
    for entry in results(job)
        if entry.outcome == "succeeded"
            println("Entry ", entry.index, ": ",
                something(text(entry.response), "non-text response"))
        elseif entry.outcome == "errored"
            println("Entry ", entry.index, " failed: ", entry.error.code)
        else
            println("Entry ", entry.index, ": ", entry.outcome)
        end
    end
end
```

Canonical entry indices start at zero. Validate the range before using
`requests[entry.index + 1]` to access a Julia vector.

`batch_job(client, id)` attaches to an existing job. `batches(client)` lists handles
where supported. `cancel!(job)` requests cancellation and updates its snapshot;
it does not guarantee immediate termination or reversal of charges.

## Video jobs

`video_generate(client, VideoGenerationRequest(...))` submits and returns a VideoJob.
Use `wait!`, inspect `job.info.status`, then `result(job)` to fetch a VideoPart.
`video_job(client, id)` attaches to existing work. Listing is provider-dependent;
there is no universal video `cancel!` method.

## Avoid duplicate work

A submission may involve an upload and then a job request. A failure between steps
can leave an uploaded file or a remotely running job. LM15 does not automatically
retry the whole sequence. Prefer checking known IDs over blindly resubmitting.

Polling deadlines do not interrupt an HTTP call already in progress. Configure the
client's transport timeout too. An application-wide budget must account for all
submissions and polling, not only a single `wait!` invocation.

See [generation](generation.md), [ownership](lifecycle.md), and
[job reference](../reference/resources.md).
