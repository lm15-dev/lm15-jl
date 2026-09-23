module ToolTests
using Test, LM15

# A tool is data, written out as in every LM15 language. LM15.jl does not derive
# tools from Julia methods or run them (removed before 1.0, 2026-09-23); what it
# offers is turning a function's result into tool-result content.

readout(result) = LM15.JSON.parse(only(result.content).text)
square(n::Int) = n^2
const SQUARE = FunctionTool(
    name="square",
    description="Square a number",
    parameters=Dict(
        "type" => "object",
        "properties" => Dict("n" => Dict("type" => "integer")),
        "required" => ["n"],
    ),
)

@testset "a request holds tools, never a function" begin
    @test only(Request("m", user("hi"); tools=SQUARE).tools) === SQUARE
    @test only(Request("m", user("hi"); tools=[SQUARE]).tools) === SQUARE
    @test only(LiveConfig(model="m", tools=[SQUARE]).tools) === SQUARE
    for tools in (square, [square], (square,))
        err = try
            Request("m", user("hi"); tools)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("not the function `square`", err.msg)
        @test occursin("describe it as FunctionTool(name=..., description=..., parameters=", err.msg)
    end
    @test_throws ArgumentError LiveConfig(model="m", tools=[square])
    @test_throws ArgumentError Request("m", user("hi"); tools=["square"])
    for name in (:tool, :ToolBinding, :tool_arguments, :execute_tool, :tool_schema, :tool_decode, :ToolInputError)
        @test !isdefined(LM15, name)
    end
end

@testset "your code runs the call and answers it" begin
    call = from_json(ToolCallPart, """{"type":"tool_call","name":"square","id":"call-1","input":{"n":3}}""")
    n = Int(call.input["n"])  # JSON integers arrive as BigInt; converting is your code's choice
    result = tool_result(call, tool_content((answer=square(n),)))
    @test result.id == "call-1" && result.name == "square"
    @test readout(result) == Dict("answer" => 9)
end

struct OffsetProbe <: AbstractVector{Int}
    values::Vector{Int}
end
Base.size(x::OffsetProbe) = size(x.values)
Base.axes(x::OffsetProbe) = (0:(length(x.values) - 1),)
Base.getindex(x::OffsetProbe, i::Int) = x.values[i + 1]

@testset "output conversion preserves meaning or refuses" begin
    @test_throws ArgumentError tool_content(OffsetProbe([1, 2]))
    @test_throws ArgumentError array_content(OffsetProbe([1, 2]))
    @test readout(tool_result("1", tool_content((a=1, b=[2, 3])))) == Dict("a"=>1, "b"=>[2, 3])
    @test tool_value(1+2im) == Dict("real"=>1, "imag"=>2)
    @test tool_value((1, 2)) == [1, 2]
    @test only(tool_content(())).text == "[]"
    @test LM15.JSON.parse(only(tool_content(0.1f0)).text) == Float64(0.1f0)
    @test only(tool_content("hello")).text == "hello"
    @test only(tool_content(image(data=UInt8[1, 2, 3]))) isa ImagePart
    @test only(tool_content(TextPart[TextPart("a")])).text == "a"
    @test LM15.JSON.parse(only(tool_content(big(2)^100)).text) == big(2)^100
    for x in (missing, NaN, Inf, BigFloat("1.2"), [1 2; 3 4], Dict(:x=>1))
        @test_throws ArgumentError tool_content(x)
    end
    cyclic = Any[]
    push!(cyclic, cyclic)
    @test_throws ArgumentError tool_content(cyclic)
    d = Dict{String,Any}()
    d["cycle"] = d
    @test_throws ArgumentError tool_content(d)
    matrix = LM15.JSON.parse(only(array_content([1 2; 3 4])).text)
    @test matrix["shape"] == [2, 2]
    @test matrix["order"] == "column-major"
    @test matrix["values"] == [1, 3, 2, 4]
end

end
