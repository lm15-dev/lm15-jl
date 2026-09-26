# Ask for a judgment, with probabilities

A judgment is a question whose answer is one of the keys you declare: yes or no, one of
several named options, or a level on an ordered scale. You ask it with an ordinary JSON
schema response format; the answer comes back as a [`DataPart`](@ref) holding the answer
object, and, where the provider can measure it, a probability for every key.

```julia
using LM15
format = judgments(
    "style" => choice("Dominant style?", ["fruit" => "Fruit-forward", "oak" => "Oak-driven", "mineral" => nothing]),
    "quality" => score("How good is this wine?", ["poor", "fair", "great"]),
    "ageing" => yes_no("Will it improve with age?"),
)
req = Request("typesafe:jev-latest", user("Deep ruby, blackcurrant and cedar, firm tannins.");
    config=Config(; response_format=format, probabilities="required"))
```

Options are given in order as a vector (a plain `Dict` has no order). `score` levels go
from low to high; their keys are `"0"`, `"1"`, …

After `answer = complete(router, req)`:

- `data(answer)` is the answer object, e.g. `Dict("style" => "oak", "quality" => 2, "ageing" => true)`;
- `probabilities(answer)` is `Dict("style" => Dict("fruit" => 0.1, …), …)`, or `nothing`;
- `data_part(answer).method` says how the numbers were measured:
  `"provider_classification"` (TypeSafe) or `"candidate_sequence_likelihood"` (a vLLM
  server that scores named tokens);
- `expected_level(probabilities(answer)["quality"])` is Σ p·level for an ordered judgment.

## Which providers measure a distribution

| Where | What you get |
|---|---|
| `typesafe` | the provider's own classification: always a distribution |
| a vLLM ≥ 0.29 server (`compat="vllm"`) via `complete` | LM15 scores every key as a token path and normalises once over the keys |
| every other provider | the model's pick only |

`config.probabilities` decides what happens where nothing can be measured: `"off"` (the
default) spends nothing extra; `"if_available"` returns the pick and records a `dropped`
adaptation on the response; `"required"` refuses before anything is sent. LM15 never
invents a distribution from a single pick or from a number the model wrote.

On vLLM the scoring path sends two `/tokenize` calls per key, one per judgment, and one
batched `/v1/completions` call; other properties of the same schema are answered by one
extra structured-output call. `stream` never scores: it answers with generated JSON and
records that.
