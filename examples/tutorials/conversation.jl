# # Keep a conversation explicit
#
# **Goal:** carry an earlier response into a follow-up Request.
# **Requires:** LM15. **Effect:** an in-memory HTTP fixture; no network or credentials.
# Replies below are deliberately fabricated, not evidence of model behavior.

using LM15

# A callable transport can return a buffered HttpResponse. These two fixture
# responses make the workflow deterministic without contacting an online service.
replies = [
    """{"model":"docs-model","choices":[{"message":{"role":"assistant","content":"8"},"finish_reason":"stop"}]}""",
    """{"model":"docs-model","choices":[{"message":{"role":"assistant","content":"64"},"finish_reason":"stop"}]}""",
]
position = Ref(0)
fixture = wire -> begin
    position[] += 1
    HttpResponse(status=200, body=Vector{UInt8}(codeunits(replies[position[]])))
end
client = OpenAIChatLM(api_key="docs-only-not-a-real-key", env=Dict{String,String}(),
    transport=fixture)

first_request = Request("docs-model", user("What is 3 plus 5?"))
first_answer = complete(client, first_request)
text(first_answer)

# Retain the whole assistant message. Rebuilding it from text could discard
# citations, media or continuation state in a real response.
follow_up = Request(first_request; messages=(
    first_request.messages...,
    first_answer.message,
    user("Square that number."),
))
final_answer = complete(client, follow_up)
text(final_answer)

# The fixture's final text is "64". A real model's wording is not deterministic.
# Notice that first_request still has one message; copies do not grow an invisible
# conversation in the client.
length(first_request.messages), length(follow_up.messages)

# To run against an actual Chat provider, replace the fixture-backed client with
# an explicitly configured real client and use its model ID. That change sends
# data and may incur charges. See [your first request](../start/first-request.md).
#
# **Common mistake:** sending only the last user message loses context. The client
# does not retain history on your behalf. For long conversations, deliberately
# select or summarize history rather than relying on hidden truncation.
#
# **Next:** [stream the response](streaming.md).
