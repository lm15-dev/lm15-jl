# Live streaming example. Not executed during documentation generation.
# Explicitly authorize provider access and charges before running this script.
using LM15

get(ENV, "LM15_RUN_LIVE_EXAMPLES", "no") == "yes" ||
    error("Live examples are disabled; explicitly authorize them before running")
client = OpenAILM(api_key=ENV["OPENAI_API_KEY"])
req = Request(get(ENV, "LM15_MODEL", "gpt-4.1-mini"), user("Explain a Julia iterator briefly.");
    config=Config(max_tokens=120))
answer = stream(client, req) do rs
    for fragment in text_chunks(rs)
        print(fragment)
    end
    println()
    response(rs)
end
println("Finish reason: ", answer.finish_reason)
