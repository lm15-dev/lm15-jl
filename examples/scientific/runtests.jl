# Executable scientific workflows. Real packages and local computation; no keys,
# provider calls, network access or billing are needed after installation.
using Test, Statistics
using LM15, DataFrames, Tables, Unitful, SciMLBase, OrdinaryDiffEqTsit5

call(name, input) = tool_call("science-1", name, Dict{String,Any}(input))
readout(result) = LM15.JSON.parse(only(result.content).text)

# The reader-facing sources own the operations; assertions below remain here.
include("tables.jl")

@testset "DataFrames and Tables workflows" begin
    @test Base.get_extension(LM15, :LM15TablesExt) !== nothing
    @test summary_tool.parameters["properties"]["column"]["enum"] == ["height"]
    summary = readout(summary_output)
    @test summary["count"] == 3
    @test summary["mean"] == 2.0
    @test_throws ArgumentError summarize_column("trial")
    rows = readout(rows_output)
    @test rows["columns"] == ["trial", "height"]
    @test rows["rows"] == [[1, 1.5], [2, 2.0]]
    @test isempty(LM15.JSON.parse(only(table_content(measurement_rows(0))).text)["rows"])
    @test_throws ArgumentError measurement_rows(4)
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
    schema = speed_tool.parameters["properties"]
    @test schema["distance"]["properties"]["unit"]["const"] == "m"
    @test distance === 12.0u"m"
    @test duration === 3.0u"s"
    output = readout(speed_output)
    @test output["value"] == 4.0
    @test output["unit"] == string(u"m/s")
    for bad in (
        Dict("value"=>1200, "unit"=>"cm"),
        Dict("value"=>12),
        Dict("value"=>12, "unit"=>"m", "extra"=>1),
        Dict("value"=>12, "unit"=>"run(`false`)"),
        Dict("value"=>true, "unit"=>"m"),
        12,
    )
        @test_throws ArgumentError quantity(bad, u"m")
    end
    values = LM15.JSON.parse(only(table_content(DataFrame(distance=[1.0u"m", 2.0u"m"]))).text)
    @test values["rows"][1][1] == Dict("value"=>1.0, "unit"=>"m")
    matrix = LM15.JSON.parse(only(array_content([1u"m" 2u"m"])).text)
    @test matrix["shape"] == [1, 2]
    @test all(v -> v["unit"] == "m", matrix["values"])
end

include("simulation.jl")
@testset "SciML: real differential-equation solve through a tool" begin
    values = readout(decay_output)
    @test values["final"] ≈ 2exp(-0.5) atol=1e-9
    @test values["time"] == 1.0
    @test_throws ArgumentError predict_decay(2.0, 0.5; duration=100.0)
    @test_throws ArgumentError number(Dict("rate"=>"arbitrary Julia code"), "rate")
    # The same tool specification goes to each provider; computation stays local.
    for constructor in (OpenAILM, OpenAIChatLM, AnthropicLM, GeminiLM)
        lm = constructor(api_key="offline-synthetic", env=Dict{String,String}())
        req = Request("model", user("Predict decay"); tools=decay_tool)
        wire = build_request(lm, req)
        @test occursin("predict_decay", String(copy(wire.body)))
        @test only(req.tools) === decay_tool
        resumed = Request(
            req;
            messages=[req.messages..., assistant(decay_call), tool_message(decay_output)],
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
        @assert tool_value(2.0u"m") == Dict("value"=>2.0, "unit"=>"m")
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
