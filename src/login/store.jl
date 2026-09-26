# The managed credential store (spec/auth-managed.md AUTH-25;
# lm15-contract auth/managed/store-layout.md, the layout every SDK shares). One JSON
# document per scope: provider entries (secret, provider-private shape) and `_lm15`,
# non-secret bookkeeping (identity generations, credential revisions, renewal markers,
# logout suppression markers, display metadata). Every commit is one serialized
# read-modify-write (`mutate`), so material and slot metadata change together.
# An unreadable or unrecognised document is a typed AuthOperationError, never an empty
# store and never overwritten.

const META_KEY = "_lm15"
const STORE_VERSION = 1

storage_error(message; reason="storage_unavailable", stage="persistence") = auth_operation_error(
    message; reason, stage, commit_state="not_committed", recovery="repair_storage", operation="store")

# AUTH-25: strict JSON. JSON.jl collapses duplicate members; a duplicate is an
# ambiguity a post-parse check cannot see, so the text is scanned first. Non-finite
# numbers (NaN, Infinity) are not JSON and are refused too.
function strict_json_parse(text::AbstractString)
    s = String(text)
    n = ncodeunits(s)
    i = 1
    skip() = (while i <= n && codeunit(s, i) in (0x20, 0x09, 0x0a, 0x0d); i += 1; end)
    function string_token()
        start = i
        i += 1
        while i <= n
            c = codeunit(s, i)
            if c == UInt8('\\')
                i += 2
            elseif c == UInt8('"')
                i += 1
                return JSON.parse(s[start:prevind(s, i)])
            else
                i += 1
            end
        end
        throw(ArgumentError("unterminated string"))
    end
    function value()
        skip()
        i > n && throw(ArgumentError("unexpected end of JSON"))
        c = codeunit(s, i)
        if c == UInt8('{')
            i += 1
            seen = Set{String}()
            skip()
            i <= n && codeunit(s, i) == UInt8('}') && (i += 1; return)
            while true
                skip()
                i <= n && codeunit(s, i) == UInt8('"') || throw(ArgumentError("expected a member name"))
                key = string_token()
                key in seen && throw(ArgumentError("duplicate member"))
                push!(seen, key)
                skip()
                i <= n && codeunit(s, i) == UInt8(':') || throw(ArgumentError("expected ':'"))
                i += 1
                value()
                skip()
                i <= n || throw(ArgumentError("unexpected end of JSON"))
                codeunit(s, i) == UInt8('}') && (i += 1; return)
                codeunit(s, i) == UInt8(',') || throw(ArgumentError("expected ',' or '}'"))
                i += 1
            end
        elseif c == UInt8('[')
            i += 1
            skip()
            i <= n && codeunit(s, i) == UInt8(']') && (i += 1; return)
            while true
                value()
                skip()
                i <= n || throw(ArgumentError("unexpected end of JSON"))
                codeunit(s, i) == UInt8(']') && (i += 1; return)
                codeunit(s, i) == UInt8(',') || throw(ArgumentError("expected ',' or ']'"))
                i += 1
            end
        elseif c == UInt8('"')
            string_token()
        else
            start = i
            while i <= n && !(codeunit(s, i) in (UInt8(','), UInt8('}'), UInt8(']'), 0x20, 0x09, 0x0a, 0x0d))
                i += 1
            end
            token = s[start:prevind(s, i)]
            token in ("true", "false", "null") && return
            occursin(r"^-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?$", token) ||
                throw(ArgumentError("invalid or non-finite number"))
        end
        return nothing
    end
    value()
    skip()
    i > n || throw(ArgumentError("trailing data after JSON"))
    parsed = JSON.parse(s)
    check_json(parsed)
    return parsed
end

function validate_document(data, where)
    data isa AbstractDict || throw(storage_error("Credential store at $where is not a JSON object; not touching it."))
    meta = get(data, META_KEY, nothing)
    if meta !== nothing
        meta isa AbstractDict ||
            throw(storage_error("Credential store at $where has a malformed $(repr(META_KEY)) block; not touching it."))
        version = get(meta, "version", nothing)
        version == STORE_VERSION && !(version isa Bool) || throw(storage_error(
            "Credential store at $where is managed-store version $(repr(version)); this lm15 reads version $STORE_VERSION. Upgrade LM15 or point LM15_CREDENTIALS_PATH at another file.";
            reason="unsupported_store_version"))
        slots = get(meta, "slots", obj())
        slots isa AbstractDict && all(v -> v isa AbstractDict, values(slots)) ||
            throw(storage_error("Credential store at $where has malformed slot metadata; not touching it."))
    end
    for (key, value) in data
        key == META_KEY || value isa AbstractDict ||
            throw(storage_error("Credential store at $where: entry $(repr(key)) is not an object; not touching it."))
    end
    return data
end

"""
    Store

The credential-store contract (AUTH-25). `MemoryStore()` and `FileStore(path)` implement
it; an application may subtype it and implement `read_document`, `transaction` and
`reserve!`.
"""
abstract type Store end
"""
    MemoryStore()

A process-lifetime managed store (`Auth(MemoryStore())`, `memory_auth()`).
"""
mutable struct MemoryStore <: Store
    data::JSONObject
    lock::ReentrantLock
end
MemoryStore() = MemoryStore(JSONObject(), ReentrantLock())
Base.show(io::IO, ::MemoryStore) = print(io, "MemoryStore(memory)")
description(::MemoryStore) = "memory"
read_document(s::MemoryStore) = lock(() -> deepcopy_json(s.data), s.lock)
struct StoreTransaction{S}
    store::S
end
function transaction(f, s::MemoryStore)
    lock(s.lock) do
        f(StoreTransaction(s))
    end
end
read_document(t::StoreTransaction{MemoryStore}) = deepcopy_json(t.store.data)
function write_document!(t::StoreTransaction{MemoryStore}, document)
    t.store.data = validate_document(deepcopy_json(document), "memory")
    return nothing
end
reserve!(::MemoryStore) = nothing

"""
    default_store_path(env=ENV)

`\$LM15_CREDENTIALS_PATH`, else `\$XDG_CONFIG_HOME/lm15/credentials.json`, else
`~/.config/lm15/credentials.json` (AUTH-8). Read at call time.
"""
default_store_path(env=ENV) = default_credentials_path(env)
"""
    FileStore(path=default_store_path(); lock_timeout_s=30)

The private file store: mode 0600, atomic writes, the AUTH-4 cross-process lock on the
canonical path. The absolute path is anchored at construction; nothing is read or created
until an operation needs it.
"""
struct FileStore <: Store
    path::String
    lock_timeout_s::Float64
end
function FileStore(path=nothing; lock_timeout_s=30.0)
    chosen = expanduser(path === nothing ? default_store_path() : String(path))
    return FileStore(isabspath(chosen) ? chosen : abspath(chosen), Float64(lock_timeout_s))
end
Base.show(io::IO, s::FileStore) = print(io, "FileStore(", s.path, ")")
description(s::FileStore) = s.path
function load_document(s::FileStore)
    isfile(s.path) || (ispath(s.path) && throw(storage_error("Could not read credential store at $(s.path)")); return JSONObject())
    text = try
        read(s.path, String)
    catch
        throw(storage_error("Could not read credential store at $(s.path)"))
    end
    data = try
        isvalid(text) || throw(ArgumentError("not UTF-8"))
        strict_json_parse(text)
    catch e
        e isa ArgumentError || e isa ErrorException || rethrow()
        throw(storage_error("Credential store at $(s.path) is not valid JSON; not touching it."))
    end
    return validate_document(data, s.path)
end
read_document(s::FileStore) = load_document(s)
function transaction(f, s::FileStore)
    hold_file_lock(s.path; timeout=s.lock_timeout_s) do
        f(StoreTransaction(s))
    end
end
read_document(t::StoreTransaction{FileStore}) = load_document(t.store)
function write_document!(t::StoreTransaction{FileStore}, document)
    validate_document(document, t.store.path)
    try
        write_private_json_atomic(t.store.path, document)
    catch e
        e isa Union{LM15Error,InterruptException} && rethrow()
        throw(storage_error("Could not write credential store at $(t.store.path)"))
    end
    return nothing
end
"""Prove the store can be written before any external authorization starts (AUTH-17)."""
function reserve!(s::FileStore)
    try
        mkpath(dirname(s.path))
    catch
        throw(storage_error("Cannot create $(dirname(s.path)) for the credential store"; stage="reservation"))
    end
    if isfile(s.path)
        writable = try
            open(io -> true, s.path, "a")
        catch
            false
        end
        writable || throw(storage_error("Credential store at $(s.path) is not writable."; stage="reservation"))
    end
    hold_file_lock(() -> load_document(s), s.path; timeout=s.lock_timeout_s)
    return nothing
end
"""
    mutate(fn, store)

Serialized read-modify-write: `fn` gets a private copy of the document and returns the
new one, or `nothing` to leave the store untouched. Returns the post-write document.
"""
function mutate(fn, s::Store)
    transaction(s) do txn
        current = read_document(txn)
        replacement = fn(deepcopy_json(current))
        replacement === nothing && return current
        write_document!(txn, replacement)
        return deepcopy_json(replacement)
    end
end
