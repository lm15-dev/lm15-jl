# Minimal JSON parser/serializer — zero dependencies.
# Handles: null, bool, number, string, array, object.

module JSON

"""Parse a JSON string into Julia types (Dict, Vector, String, Number, Bool, Nothing)."""
function parse(s::AbstractString)
    pos = Ref(1)
    val = parse_value(s, pos)
    val
end

"""Serialize a Julia value to a JSON string."""
function serialize(val)::String
    io = IOBuffer()
    write_value(io, val)
    String(take!(io))
end

# ── Parser ──────────────────────────────────────────────────────────

function skip_ws(s, pos)
    while pos[] <= length(s) && s[pos[]] in (' ', '\t', '\n', '\r')
        pos[] += 1
    end
end

function parse_value(s, pos)
    skip_ws(s, pos)
    pos[] > length(s) && error("unexpected end of JSON")
    c = s[pos[]]
    if c == '"'
        parse_string(s, pos)
    elseif c == '{'
        parse_object(s, pos)
    elseif c == '['
        parse_array(s, pos)
    elseif c == 't'
        expect(s, pos, "true"); true
    elseif c == 'f'
        expect(s, pos, "false"); false
    elseif c == 'n'
        expect(s, pos, "null"); nothing
    elseif c == '-' || isdigit(c)
        parse_number(s, pos)
    else
        error("unexpected character '$(c)' at position $(pos[])")
    end
end

function parse_string(s, pos)
    pos[] += 1  # skip opening "
    io = IOBuffer()
    while pos[] <= length(s)
        c = s[pos[]]
        if c == '"'
            pos[] += 1
            return String(take!(io))
        elseif c == '\\'
            pos[] += 1
            pos[] > length(s) && error("unexpected end in string escape")
            esc = s[pos[]]
            if esc == '"'; write(io, '"')
            elseif esc == '\\'; write(io, '\\')
            elseif esc == '/'; write(io, '/')
            elseif esc == 'n'; write(io, '\n')
            elseif esc == 'r'; write(io, '\r')
            elseif esc == 't'; write(io, '\t')
            elseif esc == 'b'; write(io, '\b')
            elseif esc == 'f'; write(io, '\f')
            elseif esc == 'u'
                hex = s[pos[]+1:pos[]+4]
                write(io, Char(Base.parse(UInt16, hex; base=16)))
                pos[] += 4
            else
                write(io, esc)
            end
        else
            write(io, c)
        end
        pos[] += 1
    end
    error("unterminated string")
end

function parse_number(s, pos)
    start = pos[]
    if s[pos[]] == '-'; pos[] += 1; end
    while pos[] <= length(s) && isdigit(s[pos[]]); pos[] += 1; end
    is_float = false
    if pos[] <= length(s) && s[pos[]] == '.'
        is_float = true; pos[] += 1
        while pos[] <= length(s) && isdigit(s[pos[]]); pos[] += 1; end
    end
    if pos[] <= length(s) && s[pos[]] in ('e', 'E')
        is_float = true; pos[] += 1
        if pos[] <= length(s) && s[pos[]] in ('+', '-'); pos[] += 1; end
        while pos[] <= length(s) && isdigit(s[pos[]]); pos[] += 1; end
    end
    numstr = s[start:pos[]-1]
    is_float ? Base.parse(Float64, numstr) : Base.parse(Int64, numstr)
end

function parse_object(s, pos)
    pos[] += 1  # skip {
    d = Dict{String,Any}()
    skip_ws(s, pos)
    if pos[] <= length(s) && s[pos[]] == '}'
        pos[] += 1; return d
    end
    while true
        skip_ws(s, pos)
        key = parse_string(s, pos)
        skip_ws(s, pos)
        s[pos[]] == ':' || error("expected ':' at $(pos[])")
        pos[] += 1
        val = parse_value(s, pos)
        d[key] = val
        skip_ws(s, pos)
        if s[pos[]] == '}'
            pos[] += 1; return d
        elseif s[pos[]] == ','
            pos[] += 1
        else
            error("expected ',' or '}' at $(pos[])")
        end
    end
end

function parse_array(s, pos)
    pos[] += 1  # skip [
    arr = Any[]
    skip_ws(s, pos)
    if pos[] <= length(s) && s[pos[]] == ']'
        pos[] += 1; return arr
    end
    while true
        push!(arr, parse_value(s, pos))
        skip_ws(s, pos)
        if s[pos[]] == ']'
            pos[] += 1; return arr
        elseif s[pos[]] == ','
            pos[] += 1
        else
            error("expected ',' or ']' at $(pos[])")
        end
    end
end

function expect(s, pos, word)
    for c in word
        (pos[] > length(s) || s[pos[]] != c) && error("expected '$word' at $(pos[])")
        pos[] += 1
    end
end

# ── Serializer ──────────────────────────────────────────────────────

function write_value(io, val::Nothing)
    write(io, "null")
end

function write_value(io, val::Bool)
    write(io, val ? "true" : "false")
end

function write_value(io, val::Integer)
    write(io, string(val))
end

function write_value(io, val::AbstractFloat)
    if isinteger(val) && isfinite(val)
        write(io, string(Int64(val)))
    else
        write(io, string(val))
    end
end

function write_value(io, val::AbstractString)
    write(io, '"')
    for c in val
        if c == '"'; write(io, "\\\"")
        elseif c == '\\'; write(io, "\\\\")
        elseif c == '\n'; write(io, "\\n")
        elseif c == '\r'; write(io, "\\r")
        elseif c == '\t'; write(io, "\\t")
        elseif c < ' '; write(io, "\\u", lpad(string(UInt16(c); base=16), 4, '0'))
        else write(io, c)
        end
    end
    write(io, '"')
end

function write_value(io, val::AbstractVector)
    write(io, '[')
    for (i, v) in enumerate(val)
        i > 1 && write(io, ',')
        write_value(io, v)
    end
    write(io, ']')
end

function write_value(io, val::AbstractDict)
    write(io, '{')
    first = true
    for (k, v) in val
        first || write(io, ',')
        first = false
        write_value(io, string(k))
        write(io, ':')
        write_value(io, v)
    end
    write(io, '}')
end

function write_value(io, val::Symbol)
    write_value(io, string(val))
end

function write_value(io, val)
    write_value(io, string(val))
end

end # module JSON
