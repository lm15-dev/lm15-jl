# # Explain a simulation without sending the solver object
#
# **Goal:** solve a small exponential-decay problem locally and return a concise
# summary. **Requires:** LM15, SciMLBase and OrdinaryDiffEqTsit5.
# **Effect:** local numerical work. No provider account or request is involved.
# This documentation source has not been executed in the writing pass.

using LM15, SciMLBase, OrdinaryDiffEqTsit5

# The model gets this bounded interface—not every method and option behind solve.
# For production, also enforce resource/time budgets outside the function if work
# can be expensive. A type declaration is not a CPU or memory sandbox.
decay_tool = @tool "Predict exponential decay over a bounded interval" function predict_decay(
    initial::Float64, rate::Float64; duration::Float64=1.0
)
    initial >= 0 && 0 <= rate <= 100 && 0 < duration <= 10 ||
        throw(ArgumentError("initial must be nonnegative, rate 0–100 and duration (0,10]"))
    problem = ODEProblem((u, p, t) -> -p*u, initial, (0.0, duration), rate)
    solution = solve(problem, Tsit5(); abstol=1e-10, reltol=1e-10,
        save_everystep=false, maxiters=10_000)
    SciMLBase.successful_retcode(solution) || error("solver did not finish successfully")
    return (initial=initial, final=last(solution.u), time=last(solution.t))
end

decay_call = tool_call("decay-1", "predict_decay", Dict("initial" => 2, "rate" => 0.5))
decay_output = execute_tool(decay_tool, decay_call)
only(decay_output.content).text

# The final value is approximately 1.2130613 at time 1.0. The analytic solution is
# initial*exp(-rate*time), so a scientific check can compare with a tolerance rather
# than relying on an exact decimal string from a particular solver version.
analytic_final = 2exp(-0.5)
local_summary = predict_decay(2.0, 0.5)
isapprox(local_summary.final, analytic_final; atol=1e-9)

# The solver's full solution, caches and internal arrays stay local. To send this
# summary later, append tool_message(decay_output) after the matching assistant
# call message and explicitly submit the next Request. That later submission can
# incur charges; this example does not perform it.
#
# **Common mistake:** serializing an entire solver object can expose irrelevant
# state and huge arrays. Choose the scientific quantities the user needs, retain
# units where appropriate, and verify the numerical result before asking a model
# to explain it.
#
# **Next:** [custom result types](../guides/extensions.md) and
# [precision and missing values](../guides/conversions.md).
