struct WireRequest
    method::String
    url::String
    headers::Vector{Pair{String,String}}
    body::Vector{UInt8}
end
function Base.show(io::IO, r::WireRequest)
    return print(
        io,
        "WireRequest(",
        r.method,
        ", <URL and headers withheld>, ",
        length(r.body),
        " body bytes)",
    )
end
Base.@kwdef struct HttpResponse
    status::Int = 200
    headers::Vector{Pair{String,String}} = Pair{String,String}[]
    body::Vector{UInt8} = UInt8[]
end
function header(r::HttpResponse, name)
    return (
        i=findfirst(p->lowercase(first(p))==lowercase(name), r.headers);
        i===nothing ? nothing : last(r.headers[i])
    )
end
function percent_encode(s; safe="")
    io=IOBuffer()
    for byte in codeunits(String(s))
        c=Char(byte)
        if ('A'<=c<='Z') || ('a'<=c<='z') || ('0'<=c<='9') || c in "-._~" || c in safe
            write(io, byte)
        else
            print(io, '%', uppercase(string(byte; base=16, pad=2)))
        end
    end
    return String(take!(io))
end
function form_encode(d)
    return join(
        (
            replace(percent_encode(string(k)), "%20"=>"+")*"="*replace(
                percent_encode(string(v)), "%20"=>"+"
            ) for (k, v) in d
        ),
        "&",
    )
end
function with_query(url, params)
    isempty(params) && return url
    return url*(occursin('?', url) ? "&" : "?")*form_encode(params)
end
path_id(id; resource_name=false) = percent_encode(id; safe=resource_name ? "/" : "")
asobject(x) = x isa AbstractDict ? x : obj()
asarray(x) = x isa AbstractVector ? x : Any[]
getobject(d, k) = asobject(get(d, k, nothing))
getarray(d, k) = asarray(get(d, k, nothing))
wire_string(x) = x === nothing ? "" : string(x)
first_nonempty(xs...) = (i=findfirst(x->!empty_optional(x), xs); i===nothing ? nothing : xs[i])
function json_object(value)
    value isa AbstractDict && return value
    value isa AbstractString && !isempty(value) || return obj()
    try
        parsed=JSON.parse(value)
        parsed isa AbstractDict ? parsed : obj("value"=>parsed)
    catch
        obj("partial_json"=>value)
    end
end
function parts_to_text(parts; provider="", where="a text field")
    out=String[]
    for p in parts
        p isa MediaPart && unsupported(provider, "$(kind(p)) in $where")
        if p isa TextPart
            push!(out, p.text)
        elseif p isa ThinkingPart && !isempty(p.text)
            push!(out, p.text)
        elseif p isa CitationPart
            push!(out, join(filter(v->!empty_optional(v), (p.title, p.url, p.text)), " — "))
        elseif p isa RefusalPart
            push!(out, p.text)
        elseif p isa DataPart
            push!(out, data_part_text(p))
        else
            unsupported(provider, "$(kind(p)) in $where")
        end
    end
    return join(out, "\n")
end
"""A data part on a wire that takes only text: its value as compact canonical JSON,
nothing added (changes/2026-09-19-jev-state.md D3)."""
data_part_text(p::DataPart) = JSON.serialize(p.value)
function system_text(s; provider="")
    return if s isa AbstractString
        String(s)
    else
        parts_to_text(s; provider, where="system instructions")
    end
end
media_base64(p::MediaPart) = p.data === nothing ? base64encode(bytes(p)) : p.data
media_uri(p::MediaPart) = "data:$(p.media_type);base64,$(media_base64(p))"
function check_result_media(provider, p, policy)
    allowed=if policy=="native"
        ("image", "document")
    elseif policy=="images"
        ("image",)
    else
        ()
    end
    for content in p.content
        content isa MediaPart &&
            !(kind(content) in allowed) &&
            unsupported(provider, "$(kind(content)) in tool results (policy=$policy)")
    end
end
error_text(p, s) = p.is_error ? "[error] "*s : s
function mark_error!(blocks, typekey, textkey)
    index=findfirst(b->get(b, "type", nothing)==typekey, blocks)
    if index===nothing
        pushfirst!(blocks, obj("type"=>typekey, textkey=>"[error]"))
    else
        (blocks[index][textkey]="[error] "*blocks[index][textkey])
    end
    return blocks
end
const EFFORT_BUDGETS=Dict(
    "minimal"=>1024, "low"=>2048, "medium"=>8192, "high"=>16384, "xhigh"=>24576, "max"=>32768
)
function has_cache_options(model)
    m=match(r"^gpt-(\d+)\.(\d+)", lowercase(model))
    return m !== nothing && (parse(Int, m[1]), parse(Int, m[2])) >= (5, 6)
end
function anthropic_adaptive(model)
    return any(
        s->occursin(s, lowercase(model)),
        (
            "sonnet-5",
            "opus-5",
            "sonnet-4-6",
            "opus-4-6",
            "opus-4-7",
            "opus-4-8",
            "fable",
            "mythos",
            "haiku-5",
        ),
    )
end
gemini_level(model) = startswith(replace(lowercase(model), r"^models/"=>""), "gemini-3")
function normalized_time(value)
    value === nothing && return nothing
    try
        seconds=value isa Real && !(value isa Bool) ? Float64(value) : expiry_seconds(value)
        Dates.format(unix2datetime(floor(seconds)), dateformat"yyyy-mm-ddTHH:MM:SS")*"Z"
    catch
        nothing
    end
end

struct SigV4Signature
    canonical_request::String
    string_to_sign::String
    authorization::String
    headers::Dict{String,String}
end
Base.show(io::IO, ::SigV4Signature) = print(io, "SigV4Signature(<redacted>)")
function sigv4_sign(
    method, url, headers, payload, credential::AwsCredentials, region, service; now=time()
)
    # Signing accepts AWS's raw request-target examples, including spaces in
    # paths. Escape them before strict URI parsing; canonicalization below
    # decodes and re-encodes each segment exactly once.
    uri=HTTP.URI(replace(url, " "=>"%20"))
    d=Dict(lowercase(k)=>string(v) for (k, v) in headers if lowercase(k)!="authorization")
    d["host"]=uri.host*(isempty(uri.port) ? "" : ":"*uri.port)
    date=Dates.format(unix2datetime(now), dateformat"yyyymmdd")
    amz=date*"T"*Dates.format(unix2datetime(now), dateformat"HHMMSS")*"Z"
    d["x-amz-date"]=amz
    delete!(d, "x-amz-security-token")
    credential.session_token===nothing || (d["x-amz-security-token"]=credential.session_token)
    segments=String[]
    for segment in split(uri.path, '/')
        if segment==".."
            (isempty(segments) || pop!(segments))
        elseif segment in ("", ".")
            nothing
        else
            push!(segments, String(segment))
        end
    end
    path="/"*join((percent_encode(HTTP.URIs.unescapeuri(s)) for s in segments), "/")
    endswith(uri.path, "/") && !endswith(path, "/") && (path*="/")
    querypairs=Tuple{String,String}[]
    for pair in split(uri.query, '&'; keepempty=false)
        bits=split(pair, '='; limit=2)
        push!(
            querypairs,
            (
                percent_encode(HTTP.URIs.unescapeuri(replace(bits[1], "+"=>" "))),
                percent_encode(
                    HTTP.URIs.unescapeuri(replace(length(bits)==2 ? bits[2] : "", "+"=>" "))
                ),
            ),
        )
    end
    query=join((k*"="*v for (k, v) in sort!(querypairs)), "&")
    for (key, value) in d
        d[key]=join(split(value), " ")
    end
    names=sort!(collect(keys(d)))
    signed=join(names, ";")
    canonical=join(
        (
            uppercase(method),
            path,
            query,
            join((k*":"*d[k]*"\n" for k in names)),
            signed,
            bytes2hex(sha256(payload)),
        ),
        "\n",
    )
    scope="$date/$region/$service/aws4_request"
    signing=join(("AWS4-HMAC-SHA256", amz, scope, bytes2hex(sha256(canonical))), "\n")
    key=Vector{UInt8}(codeunits("AWS4"*credential.secret_access_key))
    for piece in (date, region, service, "aws4_request")
        key=hmac_sha256(key, piece)
    end
    authorization="AWS4-HMAC-SHA256 Credential=$(credential.access_key_id)/$scope, SignedHeaders=$signed, Signature=$(bytes2hex(hmac_sha256(key,signing)))"
    d["authorization"]=authorization
    return SigV4Signature(canonical, signing, authorization, d)
end
