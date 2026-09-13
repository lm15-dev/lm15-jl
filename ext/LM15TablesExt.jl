module LM15TablesExt

using LM15
using Tables: Tables

function LM15.table_content(table; max_rows=100, cell=LM15.tool_value)
    max_rows isa Integer && !(max_rows isa Bool) && max_rows >= 0 ||
        throw(ArgumentError("max_rows must be a nonnegative integer"))
    Tables.istable(typeof(table)) || throw(ArgumentError("expected a Tables.jl table"))
    rows = Tables.rows(table)
    schema = Tables.schema(rows)
    names = schema === nothing ? nothing : Tuple(schema.names)
    values = Vector{Any}[]
    for row in rows
        length(values) < max_rows || throw(
            ArgumentError("table exceeds max_rows=$max_rows; select or summarize rows explicitly"),
        )
        rownames = Tuple(Tables.columnnames(row))
        names === nothing && (names = rownames)
        names == rownames || throw(ArgumentError("table rows have inconsistent columns"))
        push!(values, Any[cell(Tables.getcolumn(row, n)) for n in names])
    end
    names === nothing && throw(ArgumentError("empty table has no declared column schema"))
    length(unique(names)) == length(names) ||
        throw(ArgumentError("table column names must be unique"))
    return LM15.tool_content((columns=string.(collect(names)), rows=values))
end

end
