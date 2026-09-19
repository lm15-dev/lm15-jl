# # Read a response as it arrives
#
# **Goal:** print text fragments, then read the assembled answer.
# **Requires:** LM15. **Effect:** local synthetic events; no network.
# An online provider produces analogous events, but not these predetermined words.

using LM15

req = Request("docs-model", user("Say hello."))
source = StreamEvent[
    StreamStartEvent(model="docs-model"),
    StreamDeltaEvent(delta=TextDelta(text="Hello ")),
    StreamDeltaEvent(delta=TextDelta(text="from Julia.")),
    StreamEndEvent(finish_reason="stop", usage=Usage(input_tokens=4, output_tokens=4)),
]

answer = ResponseStream(source, req) do rs
    for fragment in text_chunks(rs)
        print(fragment)
    end
    println()
    response(rs)
end
text(answer)

# The block returns the final Response and owns the stream's cleanup. text_chunks
# filters what you see, but other events still contribute to response assembly.
# There is one consumption state: do not race this iterator with another task
# independently draining the same stream.
#
# A source without a terminal event is not a completed answer.
try
    ResponseStream(source[1:end-1], req) do rs
        response(rs)
    end
catch error
    error isa StreamAssemblyError || rethrow()
    println("Incomplete stream: ", error.code)
end

# ## With a provider
#
# In a live application use `stream(client, req) do rs ... end` around the same
# block body. That form contacts a provider when consumed and may incur charges.
# It returns the block's value and closes on errors as well as ordinary completion.
#
# **Common mistake:** break alone does not close a manually owned network source.
# Use the scoped form, or a try/finally that calls close. Closing locally does not
# guarantee that provider billing stops. See [ownership](../guides/lifecycle.md).
#
# **Next:** [handle proposed function calls](tools.md).
