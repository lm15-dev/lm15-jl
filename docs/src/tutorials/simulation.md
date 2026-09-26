# Explain a simulation without sending the solver object

Solve exponential decay with SciML and Tsit5, check that the solver succeeded, and
return a small record containing the initial value, final value and final time.
Compare the result with the analytic solution using a numerical tolerance.

[Read the complete annotated Julia lesson](../../../examples/scientific/simulation.jl).
The operation has domain bounds and an iteration budget. It runs locally; no model
is asked to perform the numerical calculation, and no provider request is made.
The documentation build runs it with the scientific assertions.

Keep solver caches and internal arrays local. Sending the summary in a later tool
message is a separate, explicit decision that can expose data and incur charges.
The scientific docs stage renders the source; the final site does not execute it
again.

Next: [extension interfaces](../guides/extensions.md) and
[notebook use](../guides/notebooks.md).
