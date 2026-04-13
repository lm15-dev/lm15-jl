# Conversation — stateless message accumulator.

mutable struct Conversation
    system::Union{String,Nothing}
    messages::Vector{Message}
end

Conversation(; system=nothing) = Conversation(system, Message[])

user!(c::Conversation, text::String) = push!(c.messages, UserMessage(text))
assistant!(c::Conversation, resp::LMResponse) = push!(c.messages, resp.message)
prefill!(c::Conversation, text::String) = push!(c.messages, AssistantMessage(text))
clear!(c::Conversation) = empty!(c.messages)
