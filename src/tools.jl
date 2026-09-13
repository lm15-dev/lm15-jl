# Julia-only conveniences. Requests still contain ordinary canonical FunctionTools;
# executable functions and conversion policies never enter serialized requests.

"""An invalid tool argument. Messages identify the field, not its potentially private value."""
struct ToolInputError <: Exception
    path::String
    message::String
end
Base.showerror(io::IO, e::ToolInputError) = print(io, "Tool input ", e.path, ": ", e.message)
input_error(path, message) = throw(ToolInputError(path, message))

"""
    tool_schema(T)

Describe the JSON accepted by `tool_decode(T, value)`. Extend both functions for
an application-owned type. Unknown types are refused, never reflected into an
invented representation. Nullable values and optional arguments are distinct.
"""
function tool_schema(T::Type)
    if T isa Union
        members = Base.uniontypes(T)
        length(members) == 2 && Nothing in members ||
            throw(ArgumentError("tool unions must be Union{Nothing,T}; use a typed wrapper"))
        return obj("anyOf" => [tool_schema(t) for t in members])
    end
    return throw(
        ArgumentError(
            "no tool representation for $T; define tool_schema/tool_decode or use a typed wrapper"
        ),
    )
end
tool_schema(::Type{Nothing}) = obj("type" => "null")
tool_schema(::Type{Bool}) = obj("type" => "boolean")
tool_schema(::Type{String}) = obj("type" => "string")
function tool_schema(::Type{T}) where {T<:Integer}
    isconcretetype(T) || throw(ArgumentError("use a concrete integer type"))
    T === BigInt && return obj("type" => "integer")
    return obj("type" => "integer", "minimum" => typemin(T), "maximum" => typemax(T))
end
function tool_schema(::Type{T}) where {T<:AbstractFloat}
    T in (Float16, Float32, Float64) ||
        throw(ArgumentError("use Float64 or an explicit decimal-string representation for $T"))
    return obj("type" => "number")
end
function tool_schema(::Type{T}) where {T<:AbstractVector}
    T in (Vector{eltype(T)}, AbstractVector{eltype(T)}) ||
        throw(ArgumentError("use Vector{T} or an explicit codec for $T"))
    return obj("type" => "array", "items" => tool_schema(eltype(T)))
end
function tool_schema(::Type{T}) where {T<:NamedTuple}
    isconcretetype(T) || throw(ArgumentError("tool named tuples need concrete field types"))
    names = fieldnames(T)
    return obj(
        "type" => "object",
        "properties" => obj((string(n) => tool_schema(fieldtype(T, n)) for n in names)...),
        "required" => string.(collect(names)),
        "additionalProperties" => false,
    )
end
function tool_schema(::Type{Dict{String,T}}) where {T}
    return obj("type" => "object", "additionalProperties" => tool_schema(T))
end
function tool_schema(::Type{T}) where {T<:Enum}
    return obj("type" => "string", "enum" => string.(collect(instances(T))))
end
function tool_schema(::Type{Complex{T}}) where {T}
    return tool_schema(NamedTuple{(:real, :imag),Tuple{T,T}})
end

"""
    tool_decode(T, value; path="input")

Check and convert JSON to the declared Julia type, without truncation, overflow,
or precision-losing conversion. No function is executed. A custom decoder must
return a value of T and enforce the same rules advertised by its schema.
"""
function tool_decode(T::Type, value; path="input")
    if T isa Union
        members = Base.uniontypes(T)
        length(members) == 2 && Nothing in members || input_error(path, "unsupported union")
        value === nothing && return nothing
        return decode_argument(only(t for t in members if t !== Nothing), value, path)
    end
    return input_error(path, "no decoder for $T; use a typed wrapper")
end
function tool_decode(::Type{Nothing}, value; path="input")
    value === nothing || input_error(path, "expected null")
    return nothing
end
function tool_decode(::Type{Bool}, value; path="input")
    value isa Bool || input_error(path, "expected a boolean")
    return value
end
function tool_decode(::Type{String}, value; path="input")
    value isa AbstractString || input_error(path, "expected a string")
    return String(value)
end
function tool_decode(::Type{T}, value; path="input") where {T<:Integer}
    value isa Real && !(value isa Bool) && isfinite(value) && isinteger(value) ||
        input_error(path, "expected an integer")
    try
        return T(value)
    catch error
        error isa Union{InexactError,OverflowError} || rethrow()
        input_error(path, "integer is outside $T's range")
    end
end
function tool_decode(::Type{T}, value; path="input") where {T<:AbstractFloat}
    value isa Union{Integer,AbstractFloat} && !(value isa Bool) && isfinite(value) ||
        input_error(path, "expected a finite number")
    converted = T(value)
    isfinite(converted) && converted == value ||
        input_error(path, "conversion to $T would lose precision or overflow")
    return converted
end
function tool_decode(::Type{T}, value; path="input") where {T<:AbstractVector}
    value isa AbstractVector || input_error(path, "expected an array")
    result = [decode_argument(eltype(T), x, "$path[$i]") for (i, x) in enumerate(value)]
    # Empty comprehensions and custom decoders must also respect the declared type.
    return convert(Vector{eltype(T)}, result)
end
function checked_object(value, names, required, path)
    value isa AbstractDict || input_error(path, "expected an object")
    all(k -> k isa AbstractString && k in string.(names), keys(value)) ||
        input_error(path, "unknown argument; expected $(join(string.(names), ", "))")
    for n in required
        haskey(value, string(n)) || input_error("$path.$n", "required argument is absent")
    end
    return value
end
function tool_decode(::Type{T}, value; path="input") where {T<:NamedTuple}
    names = fieldnames(T)
    checked_object(value, names, names, path)
    return T(Tuple(decode_argument(fieldtype(T, n), value[string(n)], "$path.$n") for n in names))
end
function tool_decode(::Type{Dict{String,T}}, value; path="input") where {T}
    value isa AbstractDict && all(k -> k isa AbstractString, keys(value)) ||
        input_error(path, "expected an object with string keys")
    return Dict{String,T}(k => decode_argument(T, v, "$path.$k") for (k, v) in value)
end
function tool_decode(::Type{T}, value; path="input") where {T<:Enum}
    value isa AbstractString || input_error(path, "expected an enum name")
    for item in instances(T)
        string(item) == value && return item
    end
    return input_error(path, "unknown enum name")
end
function tool_decode(::Type{Complex{T}}, value; path="input") where {T}
    pair = tool_decode(NamedTuple{(:real, :imag),Tuple{T,T}}, value; path)
    return Complex{T}(pair.real, pair.imag)
end

"""
    tool_value(value)

Convert an output to explicit JSON-shaped data. Supports finite ordinary numbers,
strings, null, vectors, tuples, string-keyed dictionaries, named tuples and complex numbers
(`real`/`imag`). `missing`, arbitrary structs, matrices and high-precision floats
need an explicit representation. Extend this function for your own result types.
"""
function tool_value(x)
    return throw(
        ArgumentError(
            "no tool output representation for $(typeof(x)); supply output=... or define tool_value for your type",
        ),
    )
end
tool_value(::Nothing) = nothing
tool_value(x::Bool) = x
tool_value(x::Integer) = x
tool_value(x::AbstractString) = String(x)
function tool_value(x::AbstractFloat)
    x isa Union{Float16,Float32,Float64} && isfinite(x) || throw(
        ArgumentError(
            "tool numbers must be finite; encode high-precision numbers explicitly as strings"
        ),
    )
    # Promote exactly before JSON encoding: Float32's short display spelling
    # need not denote the same value when the receiver parses it as Float64.
    return Float64(x)
end
tool_value(x::Enum) = string(x)
tool_value(x::Complex) = obj("real" => tool_value(real(x)), "imag" => tool_value(imag(x)))
function tool_value(x::Union{AbstractDict,AbstractVector,Tuple,NamedTuple})
    return output_value(x, IdDict{Any,Nothing}())
end
output_value(x, active) = tool_value(x)
function output_value(x::Union{AbstractDict,AbstractVector,Tuple,NamedTuple}, active)
    haskey(active, x) && throw(ArgumentError("tool outputs cannot contain cycles"))
    active[x] = nothing
    try
        if x isa Union{AbstractVector,Tuple}
            x isa AbstractVector &&
                Base.has_offset_axes(x) &&
                throw(ArgumentError("encode offset axes explicitly"))
            return [output_value(v, active) for v in x]
        end
        x isa AbstractDict &&
            !all(k -> k isa AbstractString, keys(x)) &&
            throw(ArgumentError("tool output dictionary keys must be strings"))
        return obj((string(k) => output_value(v, active) for (k, v) in pairs(x))...)
    finally
        delete!(active, x)
    end
end

"""Convert a function result to presentational Parts; data becomes JSON inside a TextPart."""
function tool_content(value)
    data = tool_value(value)
    check_json(data)
    return (TextPart(JSON.serialize(data)),)
end
tool_content(value::AbstractString) = (TextPart(value),)
tool_content(value::Part) = (value,)
tool_content(::Tuple{}) = (TextPart("[]"),)
function tool_content(value::Union{Tuple{Vararg{Part}},AbstractVector{<:Part}})
    return Tuple(value)
end

"""
    array_content(array)

Explicit array output: shape, column-major ordering and values. Nothing is
flattened without recording its dimensions. Offset axes require an application
representation instead. This materializes the array; summarize large arrays first.
"""
function array_content(array::AbstractArray)
    Base.has_offset_axes(array) && throw(ArgumentError("encode offset axes explicitly"))
    values = Any[]
    for x in array
        push!(values, tool_value(x))
    end
    return tool_content((shape=collect(size(array)), order="column-major", values=values))
end

"""
    table_content(table; max_rows=100, cell=tool_value)

Available when Tables.jl is loaded. Encode named columns and rows without silently
truncating. Too many rows are an error. Choose columns/rows before calling; use an
explicit cell converter when missing values or custom types need a policy.
"""
function table_content end

struct ToolBinding{F,A,O}
    f::F
    arguments::Type{A}
    name::String
    description::Union{Nothing,String}
    keywords::Tuple{Vararg{Symbol}}
    optional::Tuple{Vararg{Symbol}}
    output::O
end
function Base.show(io::IO, t::ToolBinding)
    return print(io, "ToolBinding(", repr(t.name), "; explicit execution)")
end

"""
    tool(f, NamedTuple{names, Tuple{types...}}; name, description=nothing,
         keywords=(), optional=(), output=tool_content)

Bind an explicitly selected typed interface. Works with functions, closures and
callable structs. `optional` names use the function's own defaults, evaluated only
on execution. Missing positional arguments must form a suffix. Prefer keywords
for independently optional arguments. Use `@tool` to obtain names/types from syntax.
"""
function tool(
    f,
    ::Type{A};
    name=string(f isa Function ? nameof(f) : nameof(typeof(f))),
    description=nothing,
    keywords=(),
    optional=(),
    output=tool_content,
) where {A<:NamedTuple}
    isempty(methods(output)) && throw(ArgumentError("output must be callable"))
    names = fieldnames(A)
    keywords, optional = Tuple(keywords), Tuple(optional)
    all(n -> n isa Symbol && n in names, (keywords..., optional...)) ||
        throw(ArgumentError("keywords/optional must name declared arguments"))
    length(unique(keywords)) == length(keywords) && length(unique(optional)) == length(optional) ||
        throw(ArgumentError("duplicate keyword/optional argument"))
    positional = filter(n -> !(n in keywords), names)
    seen_optional = false
    for n in positional
        seen_optional &&
            !(n in optional) &&
            throw(ArgumentError("optional positional arguments must be trailing"))
        seen_optional |= n in optional
    end
    signature = Tuple{(fieldtype(A, n) for n in positional)...}
    hasmethod(f, signature, keywords) ||
        throw(ArgumentError("function has no method accepting this interface"))
    for count in 0:length(positional)
        all(n -> n in optional, positional[(count + 1):end]) || continue
        hasmethod(f, Tuple{(fieldtype(A, n) for n in positional[1:count])...}, keywords) ||
            throw(ArgumentError("function has no method for the declared positional defaults"))
    end
    t = ToolBinding(
        f,
        A,
        String(name),
        description === nothing ? nothing : String(description),
        keywords,
        optional,
        output,
    )
    FunctionTool(t) # Fail on unsupported types at binding time, never on the first network request.
    return t
end
function tool(t::ToolBinding; name=t.name, description=t.description, output=t.output)
    return tool(
        t.f, t.arguments; name, description, keywords=t.keywords, optional=t.optional, output
    )
end
function FunctionTool(t::ToolBinding)
    names = fieldnames(t.arguments)
    return FunctionTool(;
        name=t.name,
        description=t.description,
        parameters=obj(
            "type" => "object",
            "properties" =>
                obj((string(n) => tool_schema(fieldtype(t.arguments, n)) for n in names)...),
            "required" => [string(n) for n in names if !(n in t.optional)],
            "additionalProperties" => false,
        ),
    )
end
tool_spec(t::ToolBinding) = FunctionTool(t)
tool_spec(t) = t
normalize_tools(t::Union{Tool,ToolBinding}) = (tool_spec(t),)
normalize_tools(ts) = Tuple(tool_spec(t) for t in ts)

function decode_argument(T, raw, path)
    value = tool_decode(T, raw; path)
    value isa T || input_error(path, "custom decoder did not return $T")
    return value
end

"""Validate/decode one call, without executing it. Useful before asking for approval."""
function tool_arguments(t::ToolBinding, call::ToolCallPart)
    call.name == t.name || input_error("call", "name does not match the selected tool")
    names = fieldnames(t.arguments)
    checked_object(call.input, names, filter(n -> !(n in t.optional), names), "input")
    positional = filter(n -> !(n in t.keywords), names)
    absent = false
    for n in positional
        present = haskey(call.input, string(n))
        absent &&
            present &&
            input_error(
                "input.$n",
                "a preceding positional argument is absent; use keyword arguments for independent defaults",
            )
        absent |= !present
    end
    present = filter(n -> haskey(call.input, string(n)), names)
    values = Tuple(
        decode_argument(fieldtype(t.arguments, n), call.input[string(n)], "input.$n") for
        n in present
    )
    return NamedTuple{present}(values)
end

"""
    execute_tool(binding, call)

Explicitly execute exactly one selected tool and return a canonical ToolResultPart.
Arguments are checked before invoking the function. Function errors propagate;
there are no retries, automatic loops, or fabricated successful results. Output
conversion can fail *after* side effects: do not blindly retry an execution.
"""
function execute_tool(t::ToolBinding, call::ToolCallPart)
    args = tool_arguments(t, call)
    positional = filter(n -> !(n in t.keywords), keys(args))
    keywords = filter(n -> n in t.keywords, keys(args))
    signature = Tuple{(fieldtype(t.arguments, n) for n in positional)...}
    kwargs = NamedTuple{keywords}(Tuple(args[n] for n in keywords))
    value = invoke(t.f, signature, (args[n] for n in positional)...; kwargs...)
    return tool_result(call, t.output(value))
end

"""
    binding = @tool function summarize(values::Vector{Float64}; digits::Int=2)
        ...
    end
    binding = @tool existing_function(x::Int)
    binding = @tool "Description for the model" existing_function(x::Int)

Capture an explicit typed interface, not an arbitrary method table. The definition
form defines an ordinary Julia function and returns a separate binding. The
signature-only form calls nothing and defines nothing. Defaults are supported in
function definitions; existing optional methods can use `tool(...; optional=...)`.
Varargs, destructuring, `where`, untyped arguments and ambiguous unions are refused.
"""
macro tool(args...)
    length(args) in (1, 2) ||
        error("@tool expects an optional description and a typed function/signature")
    description = length(args) == 2 ? args[1] : nothing
    ex = args[end]
    definition = ex isa Expr && ex.head in (:function, :(=))
    sig = definition ? ex.args[1] : ex
    # A declared return type affects Julia execution, not the input schema.
    sig isa Expr && sig.head == :(::) && (sig = sig.args[1])
    sig isa Expr && sig.head == :call ||
        error("@tool requires a named typed signature; wrap where/vararg methods explicitly")
    fn = sig.args[1]
    fn isa Symbol ||
        (fn isa Expr && fn.head == :.) ||
        error("@tool requires a function name; use tool(f, NamedTupleType) for callables")
    names, types, optional, keywords = Symbol[], Any[], Symbol[], Symbol[]
    function parameter(arg, keyword)
        default = arg isa Expr && arg.head in (:kw, :(=))
        default &&
            !definition &&
            error(
                "defaults belong to the function definition; use tool(...; optional=...) for existing methods",
            )
        decl = default ? arg.args[1] : arg
        decl isa Expr && decl.head == :(::) && length(decl.args) == 2 && decl.args[1] isa Symbol ||
            error(
                "every @tool argument needs a name and type; varargs/destructuring are not supported",
            )
        n = decl.args[1]
        n in names && error("duplicate @tool argument")
        push!(names, n)
        push!(types, decl.args[2])
        default && push!(optional, n)
        return keyword && push!(keywords, n)
    end
    # Keep positional names first, then keyword names, irrespective of AST order.
    for arg in sig.args[2:end]
        arg isa Expr && arg.head == :parameters && continue
        parameter(arg, false)
    end
    for arg in sig.args[2:end]
        arg isa Expr && arg.head == :parameters || continue
        foreach(a -> parameter(a, true), arg.args)
    end
    A = :(NamedTuple{$(QuoteNode(Tuple(names))),Tuple{$(types...)}})
    binding = :($(GlobalRef(@__MODULE__, :tool))(
        $fn,
        $A;
        name=($(string(fn isa Symbol ? fn : fn.args[end].value))),
        description=($description),
        optional=($(QuoteNode(Tuple(optional)))),
        keywords=($(QuoteNode(Tuple(keywords)))),
    ))
    return esc(definition ? quote
        $ex
        $binding
    end : binding)
end
