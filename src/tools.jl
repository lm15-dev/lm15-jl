# Tools are data: a FunctionTool (name, description, JSON Schema) or a
# BuiltinTool, written out as in every LM15 language. LM15.jl does not derive
# tools from Julia methods or run them; `@tool` and its bindings were removed
# before 1.0 (2026-09-23). What remains here turns a function's *result* into
# tool-result content, for the answer you send back.

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
            "no tool output representation for $(typeof(x)); define tool_value for your type",
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

# A request holds only tools, never a function: a function gets the fix named.
function tool_spec(t)
    t isa Tool && return t
    what = t isa Function ? "the function `$(nameof(t))`" : "a $(typeof(t))"
    throw(
        ArgumentError(
            "tools must be FunctionTool or BuiltinTool values, not $what: describe it as " *
            "FunctionTool(name=..., description=..., parameters=Dict(...JSON Schema...)) " *
            "and keep the function to run when the model calls it",
        ),
    )
end
normalize_tools(t::Tool) = (t,)
normalize_tools(t::Function) = (tool_spec(t),)
normalize_tools(ts) = Tuple(tool_spec(t) for t in ts)
