# Real scientific workflows, without a provider account

These examples use the installed DataFrames, Tables, Unitful and SciML packages—not
stand-ins. They do local computation and build provider requests without sending
them. They need no API keys and cannot incur inference charges.

From this directory, install once (this step downloads packages):

```sh
julia --project=. -e 'using Pkg; Pkg.develop(path="../.."); Pkg.instantiate()'
```

Then run:

```sh
julia --startup-file=no --project=. runtests.jl
```

The executable examples and assertions are together in [`runtests.jl`](runtests.jl):

- **DataFrames:** choose a column, compute its summary, and return selected table
  rows. Excess rows cause an error instead of hidden truncation. Converting
  `missing` to null requires an explicit choice.
- **Unitful:** accept distances in metres and durations in seconds, compute speed,
  and return both its value and unit. Wrong units are refused, never evaluated as
  Julia expressions. Quantities also work inside tables and explicit array output.
- **SciML:** solve an actual exponential-decay differential equation using Tsit5,
  check the numerical result, and return a small summary. The solver object stays
  local. The operation has explicit input limits and an iteration budget.
- **Provider boundary:** build the tool declaration and its result replay for
  OpenAI Responses, OpenAI Chat, Anthropic and Gemini. No requests are sent.

This environment has its own dependencies. Installing LM15 alone does not install
DataFrames, Unitful or a differential-equation solver. The examples are tested on
Julia 1.12; the package's wider native-platform claims remain separate.

See [the function-tool guide](../../docs/function-tools.md) for the API and its
intentional limits.
