# Executable scientific workflows. Real packages and local computation; no keys,
# provider calls, network access or billing are needed after installation.
using Test, Statistics
using LM15, DataFrames, Tables, Unitful, SciMLBase, OrdinaryDiffEqTsit5

call(t, input) = tool_call("science-1", t.name, Dict{String,Any}(input))
readout(result) = LM15.JSON.parse(only(result.content).text)

# The reader-facing sources own the operations; assertions below remain here.
# This extraction and the new example demonstrations have not been executed yet.
include("tables.jl")

@testset "DataFrames and Tables workflows" begin
    @test Base.get_extension(LM15, :LM15TablesExt) !== nothing
    summary = readout(execute_tool(summary_tool, call(summary_tool, Dict("column"=>"height"))))
    @test summary["count"] == 3
    @test summary["mean"] == 2.0
    rows = readout(execute_tool(rows_tool, call(rows_tool, Dict("count"=>2))))
    @test rows["columns"] == ["trial", "height"]
    @test rows["rows"] == [[1, 1.5], [2, 2.0]]
    @test isempty(readout(execute_tool(rows_tool, call(rows_tool, Dict("count"=>0))))["rows"])
    @test_throws ArgumentError table_content(measurements; max_rows=2)
    @test_throws ArgumentError table_content(measurements; max_rows=true)
    @test_throws ArgumentError table_content(measurements; max_rows=-1)
    @test_throws ArgumentError table_content(12)
    @test_throws ArgumentError tool_content(measurements) # No accidental entire-table upload.
    @test LM15.JSON.parse(only(table_content(Tables.rowtable(measurements))).text)["rows"] ==
        [[1, 1.5], [2, 2.0], [3, 2.5]]
    with_missing = DataFrame(x=[1, missing])
    @test_throws ArgumentError table_content(with_missing)
    null_policy(x) = ismissing(x) ? nothing : tool_value(x)
    explicit = LM15.JSON.parse(only(table_content(with_missing; cell=null_policy)).text)
    @test explicit["rows"] == [[1], [nothing]]
end

include("units.jl")
@testset "Unitful: values retain their units" begin
    @test Base.get_extension(LM15, :LM15UnitfulExt) !== nothing
    t = speed_tool
    schema = FunctionTool(t).parameters["properties"]
    @test schema["distance"]["properties"]["unit"]["const"] == "m"
    input = Dict(
        "distance"=>Dict("value"=>12, "unit"=>"m"), "duration"=>Dict("value"=>3, "unit"=>"s")
    )
    args = tool_arguments(t, call(t, input))
    @test args.distance === 12.0u"m"
    @test args.duration === 3.0u"s"
    output = readout(execute_tool(t, call(t, input)))
    @test output["value"] == 4.0
    @test output["unit"] == string(u"m/s")
    @test tool_decode(typeof(1.0u"m/s"), output) === 4.0u"m/s"
    for bad in (
        Dict("value"=>1200, "unit"=>"cm"),
        Dict("value"=>12),
        Dict("value"=>12, "unit"=>"m", "extra"=>1),
        Dict("value"=>12, "unit"=>"run(`false`)"),
        12,
    )
        @test_throws ToolInputError execute_tool(t, call(t, merge(input, Dict("distance"=>bad))))
    end
    @test_throws ToolInputError tool_decode(typeof(1u"m"), Dict("value"=>1.5, "unit"=>"m"))
    values = LM15.JSON.parse(only(table_content(DataFrame(distance=[1.0u"m", 2.0u"m"]))).text)
    @test values["rows"][1][1] == Dict("value"=>1.0, "unit"=>"m")
    matrix = LM15.JSON.parse(only(array_content([1u"m" 2u"m"])).text)
    @test matrix["shape"] == [1, 2]
    @test all(v -> v["unit"] == "m", matrix["values"])
end

include("simulation.jl")
@testset "SciML: real differential-equation solve through a tool" begin
    result = execute_tool(decay_tool, call(decay_tool, Dict("initial"=>2, "rate"=>0.5)))
    values = readout(result)
    @test values["final"] ≈ 2exp(-0.5) atol=1e-9
    @test values["time"] == 1.0
    @test_throws ArgumentError execute_tool(
        decay_tool, call(decay_tool, Dict("initial"=>2, "rate"=>0.5, "duration"=>100))
    )
    @test_throws ToolInputError execute_tool(
        decay_tool, call(decay_tool, Dict("initial"=>2, "rate"=>"arbitrary Julia code"))
    )
    # The same tool specification goes to each provider; computation stays local.
    for constructor in (OpenAILM, OpenAIChatLM, AnthropicLM, GeminiLM)
        lm = constructor(api_key="offline-synthetic", env=Dict{String,String}())
        req = Request("model", user("Predict decay"); tools=decay_tool)
        wire = build_request(lm, req)
        @test occursin("predict_decay", String(copy(wire.body)))
        @test only(req.tools) isa FunctionTool
        resumed = Request(
            req;
            messages=[
                req.messages...,
                assistant(call(decay_tool, Dict("initial"=>2, "rate"=>0.5))),
                tool_message(result),
            ],
        )
        @test !isempty(build_request(lm, resumed).body)
    end
end

# Detect new ambiguities involving our methods, not unrelated upstream pairs.
@testset "extension dispatch" begin
    extensions = (
        Base.get_extension(LM15, :LM15TablesExt), Base.get_extension(LM15, :LM15UnitfulExt)
    )
    ambiguities = Test.detect_ambiguities(LM15, extensions...; recursive=true)
    @test isempty(ambiguities)
    reverse_order = raw"""
        using Unitful, Tables
        using LM15
        @assert Base.get_extension(LM15, :LM15TablesExt) !== nothing
        @assert Base.get_extension(LM15, :LM15UnitfulExt) !== nothing
        @assert tool_decode(typeof(1.0u"m"), Dict("value"=>2,"unit"=>"m")) === 2.0u"m"
        @assert LM15.JSON.parse(only(table_content([(x=1,)])).text) == Dict("columns"=>["x"],"rows"=>[[1]])
    """
    @test success(
        pipeline(
            `$(Base.julia_cmd()) --startup-file=no --project=$(@__DIR__) -e $reverse_order`;
            stdout=stdout,
            stderr=stderr,
        ),
    )
end
