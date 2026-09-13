module ToolTests
using Test, LM15

call(t, input) = tool_call("call-1", t.name, Dict{String,Any}(input))
readout(result) = LM15.JSON.parse(only(result.content).text)

const executions = Ref(0)
const defaults = Ref(0)
square_tool = @tool "Square a number" function square(n::Int; scale::Float64=(defaults[] += 1; 1.0))
    executions[] += 1
    return (answer=n^2 * scale, input_type=string(typeof(n)))
end

@testset "function-derived tools stay separate from execution" begin
    @test square(2) == (answer=4.0, input_type="Int64")
    executions[] = 0
    defaults[] = 0
    t = square_tool
    spec = FunctionTool(t)
    @test spec.name == "square"
    @test spec.description == "Square a number"
    @test spec.parameters["required"] == ["n"]
    @test spec.parameters["properties"]["n"]["type"] == "integer"
    @test spec.parameters["additionalProperties"] === false
    req = Request("m", user("hi"); tools=t)
    @test only(req.tools) isa FunctionTool
    @test only(LiveConfig(model="m", tools=[t]).tools) isa FunctionTool
    @test to_json(req) == to_json(Request("m", user("hi"); tools=spec))
    @test from_json(Request, to_json(req)).tools[1].parameters == spec.parameters
    @test executions[] == defaults[] == 0
    c = from_json(
        ToolCallPart, """{"type":"tool_call","name":"square","id":"call-1","input":{"n":3}}"""
    )
    @test tool_arguments(t, c) === (n=3,)
    @test executions[] == defaults[] == 0
    result = execute_tool(t, c)
    @test result.id == c.id && result.name == c.name
    @test readout(result)["answer"] == 9.0
    @test executions[] == defaults[] == 1
    @test readout(execute_tool(t, call(t, Dict("n"=>3, "scale"=>2))))["answer"] == 18.0
    @test defaults[] == 1
    @test readout(
        execute_tool(tool(t; name="renamed"), tool_call("2", "renamed", Dict("n"=>2)))
    )["answer"] == 4
    @test !occursin("executions", sprint(show, t))
end

@testset "invalid calls cannot execute functions" begin
    before = executions[]
    for input in (
        Dict(),
        Dict("n"=>true),
        Dict("n"=>1.5),
        Dict("n"=>nothing),
        Dict("n"=>"SECRET"),
        Dict("n"=>big(2)^80),
        Dict("n"=>1, "extra"=>2),
    )
        @test_throws ToolInputError execute_tool(square_tool, call(square_tool, input))
    end
    @test_throws ToolInputError execute_tool(square_tool, tool_call("1", "other", Dict("n"=>1)))
    err = try
        tool_arguments(square_tool, call(square_tool, Dict("n"=>"SECRET")))
    catch e
        e
    end
    @test occursin("input.n", sprint(showerror, err))
    @test !occursin("SECRET", sprint(showerror, err))
    @test executions[] == before
end

const nullable = @tool function nullable_input(x::Union{Nothing,Int}; y::Int=7)
    (x=x, y=y)
end
const positional = @tool positional_input(x::Int, y::Int=2, z::Int=3) = x+y+z
const no_args = @tool no_arguments() = "ok"
@testset "required, nullable, positional and keyword defaults" begin
    @test FunctionTool(nullable).parameters["required"] == ["x"]
    @test readout(execute_tool(nullable, call(nullable, Dict("x"=>nothing)))) ==
        Dict("x"=>nothing, "y"=>7)
    @test_throws ToolInputError execute_tool(nullable, call(nullable, Dict()))
    @test_throws ToolInputError execute_tool(nullable, call(nullable, Dict("x"=>1, "y"=>nothing)))
    @test readout(execute_tool(positional, call(positional, Dict("x"=>1)))) == 6
    @test readout(execute_tool(positional, call(positional, Dict("x"=>1, "y"=>5)))) == 9
    @test_throws ToolInputError execute_tool(positional, call(positional, Dict("x"=>1, "z"=>5)))
    @test only(execute_tool(no_args, call(no_args, Dict())).content).text == "ok"
    @test isempty(FunctionTool(no_args).parameters["required"])
end

choose(x::AbstractVector{Float64}) = "abstract method"
choose(x::Vector{Float64}) = "concrete method"
selected = @tool choose(x::AbstractVector{Float64})
struct Multiplier
    factor::Int
end
(m::Multiplier)(x::Int) = m.factor*x
@testset "chosen method, closures and callable objects" begin
    @test only(execute_tool(selected, call(selected, Dict("x"=>[1, 2]))).content).text ==
        "abstract method"
    closure = tool(x -> x+1, NamedTuple{(:x,),Tuple{Int}}; name="increment")
    @test readout(execute_tool(closure, call(closure, Dict("x"=>2)))) == 3
    functor = tool(Multiplier(3), NamedTuple{(:x,),Tuple{Int}}; name="multiply")
    @test readout(execute_tool(functor, call(functor, Dict("x"=>2)))) == 6
    existing = @tool square(n::Int)
    @test readout(execute_tool(existing, call(existing, Dict("n"=>2))))["answer"] == 4
end

@enum Mode begin
    fast
    slow
end
@testset "type-directed codecs" begin
    for (T, raw, expected) in (
        (Int, big(12), 12),
        (UInt8, 255, 0xff),
        (Float64, 2, 2.0),
        (Float32, 0.5, 0.5f0),
        (String, "abc", "abc"),
        (Bool, true, true),
        (Nothing, nothing, nothing),
        (Vector{Float64}, [1, 2], [1.0, 2.0]),
        (Vector{Int}, Any[], Int[]),
        (Dict{String,Int}, Dict("x"=>big(3)), Dict("x"=>3)),
        (NamedTuple{(:x, :y),Tuple{Int,String}}, Dict("x"=>1, "y"=>"a"), (x=1, y="a")),
        (Mode, "fast", fast),
        (ComplexF64, Dict("real"=>1, "imag"=>2), 1.0+2.0im),
    )
        @test !isempty(tool_schema(T))
        decoded = tool_decode(T, raw)
        @test decoded isa T
        @test decoded == expected
    end
    for (T, raw) in (
        (Int, true),
        (UInt8, -1),
        (Int, 0.5),
        (Bool, 1),
        (Float64, big(2)^60+1),
        (Float32, 0.1),
        (Float32, 1e300),
        (Float64, Inf),
        (Float64, true),
        (String, 12),
        (Mode, "unknown"),
        (Vector{Int}, [1 2]),
        (Union{Nothing,Int}, "x"),
    )
        @test_throws ToolInputError tool_decode(T, raw)
    end
    for T in (
        Any,
        Real,
        AbstractString,
        Matrix{Float64},
        Union{Int,String},
        Union{Missing,Int},
        BigFloat,
        Tuple{Int,Int},
        Dict{Symbol,Int},
    )
        @test_throws ArgumentError tool_schema(T)
    end
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
    @test tool_decode(Float32, LM15.JSON.parse(only(tool_content(0.1f0)).text)) === 0.1f0
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

const throws_tool = @tool throws_error(n::Int) = throw(DomainError(n))
const side_effects = Ref(0)
const bad_output = @tool function bad_result(n::Int)
    side_effects[] += 1
    return missing
end
@testset "execution has no retries, implicit error conversion or loops" begin
    @test_throws DomainError execute_tool(throws_tool, call(throws_tool, Dict("n"=>1)))
    @test_throws ArgumentError execute_tool(bad_output, call(bad_output, Dict("n"=>1)))
    @test side_effects[] == 1
    explicit = tool(bad_output; output=x -> x === missing ? "unreported" : tool_content(x))
    @test only(execute_tool(explicit, call(explicit, Dict("n"=>1))).content).text == "unreported"
end

struct Label
    value::String
end
struct BrokenDecoder end
LM15.tool_schema(::Type{BrokenDecoder}) = tool_schema(Int)
LM15.tool_decode(::Type{BrokenDecoder}, raw; path="input") = 1
broken_tool = @tool broken_nested(values::Vector{BrokenDecoder}) = error("must never execute")
return_typed = @tool function typed_return(n::Int)::Int
    n + 1
end
LM15.tool_schema(::Type{Label}) = tool_schema(String)
LM15.tool_decode(::Type{Label}, value; path="input") = Label(tool_decode(String, value; path))
LM15.tool_value(value::Label) = value.value
label_tool = @tool label_length(label::Label) = length(label.value)

@testset "application-owned types extend the same dispatch interface" begin
    @test readout(execute_tool(label_tool, call(label_tool, Dict("label"=>"hello")))) == 5
    @test tool_arguments(label_tool, call(label_tool, Dict("label"=>"hello"))).label isa Label
    @test_throws ToolInputError execute_tool(broken_tool, call(broken_tool, Dict("values"=>[1])))
    @test readout(execute_tool(return_typed, call(return_typed, Dict("n"=>1)))) == 2
    @test_throws ToolInputError execute_tool(label_tool, call(label_tool, Dict("label"=>12)))
    @test only(tool_content(Label("hello"))).text == "\"hello\""
    spec = FunctionTool(label_tool)
    spec.parameters["properties"]["label"]["type"] = "number"
    @test FunctionTool(label_tool).parameters["properties"]["label"]["type"] == "string"
    @test_throws ToolInputError execute_tool(label_tool, call(label_tool, Dict("label"=>12)))
end

@testset "unsupported definitions fail early" begin
    @test_throws ArgumentError tool(square_tool; output=3)
    @test isempty(Test.detect_ambiguities(LM15; recursive=true))
    for expression in (
        :(@tool f(x)),
        :(@tool f(x::Int...)),
        :(@tool f(x::T) where {T}),
        :(@tool f(x::Int=2)),
        :(@tool f((x, y)::Tuple)),
    )
        @test_throws Exception macroexpand(@__MODULE__, expression)
    end
    @test_throws ArgumentError tool(identity, NamedTuple{(:x,),Tuple{Any}})
    @test_throws ArgumentError tool(identity, NamedTuple{(:x,),Tuple{Int}}; optional=(:x,))
    @test_throws ArgumentError tool(identity, NamedTuple{(:x,),Tuple{Int}}; keywords=(:typo,))
end

end
