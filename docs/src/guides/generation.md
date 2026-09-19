# Generate images, speech and video

Generation calls require provider/model support and can incur charges. Model IDs,
voices, sizes and formats are provider choices, not universal LM15 constants.
These snippets are illustrative live operations and are not executed by the docs.

## Images

```julia
using LM15
client = OpenAILM(api_key=ENV["OPENAI_API_KEY"])
generated = image_generate(client,
    ImageGenerationRequest(model=ENV["LM15_IMAGE_MODEL"], prompt="A simple botanical sketch."))
for part in generated.images
    println(part) # Summary, not a dump of base64 content.
end
```

Inspect whether each ImagePart has inline data, a URL or a file reference.
`bytes(part)` does not automatically fetch URLs. Write local output files only to a
path chosen by the application. Media may be large and is buffered in memory.

Input images for editing require an explicitly supported mapping. A type accepting
an images field does not imply every backend can perform that edit.

## Speech

```julia
speech = speech_generate(client, SpeechGenerationRequest(
    model=ENV["LM15_SPEECH_MODEL"], prompt="Hello from Julia.",
    voice=ENV["LM15_VOICE"], format="mp3"))
```

The result contains an AudioPart. The voice/model/format must be supported by the
provider. These functions do not configure a microphone or audio playback device.

## Video

```julia
job = video_generate(client, VideoGenerationRequest(
    model=ENV["LM15_VIDEO_MODEL"], prompt="Clouds crossing a mountain ridge."))
wait!(job; timeout=3600)
job.info.status == "completed" || error("Video did not complete successfully")
part = result(job)
```

This is asynchronous work, not a quick property lookup. Some providers reject
explicit durations, input images or video listing. `result` does not wait on its
own. Keep the job ID if the wait times out and inspect it later rather than creating
a duplicate. See [jobs](jobs.md).

The canonical request types describe shared shapes. Actual provider mappings can
be narrower; explicit refusal is preferable to silently dropping requested inputs.
See [status](../project/status.md) and [resource reference](../reference/resources.md).
