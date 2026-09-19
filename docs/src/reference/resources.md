# Resources, generation and jobs

```@meta
CurrentModule = LM15
```

These are remote operations unless a specific accessor/constructor says otherwise.
Some high-level submissions make several HTTP requests. Inspect IDs and status
before retrying uncertain work. See [files and caching](../guides/files-caches.md)
and [jobs](../guides/jobs.md).

## Files

```@docs
FileUploadRequest
FileInfo
FilePage
file_upload
file_get
file_list
file_delete
file_download
file_wait_ready
```

## Reusable context

```@docs
CacheInfo
CachePage
CachedPrefix
cache
cache_config
request
cache_create
cache_get
cache_list
cache_delete
cache_update
```

## Batches

```@docs
BatchRequest
BatchJobInfo
BatchEntry
BatchJob
batch
batch_job
batches
batch_submit
batch_status
batch_results
batch_cancel
batch_list
```

## Image, speech and video generation

```@docs
ImageGenerationRequest
ImageGenerationResponse
SpeechGenerationRequest
SpeechGenerationResponse
VideoGenerationRequest
VideoJobInfo
VideoJob
image_generate
speech_generate
video_generate
video_job
video_jobs
video_submit
video_status
video_result
video_list
```

## Job lifecycle and results

A terminal job need not be successful. A wait deadline does not cancel the remote
job. `result(::TurnView)` is documented here with the same generic function as the
video result operation, but its stopping behavior is different.

```@docs
done
refresh!
wait!
cancel!
result
results
WaitTimeout
```
