# Shared helpers for provider adapters.

function parts_to_text(parts)
    join([p.text for p in parts if p.type == "text" && p.text !== nothing], "\n")
end

function part_to_openai_input(p::Part)
    if p.type == "text"
        return Dict{String,Any}("type" => "input_text", "text" => something(p.text, ""))
    elseif p.type == "image" && p.source !== nothing
        if p.source.type == "url"
            return Dict{String,Any}("type" => "input_image", "image_url" => p.source.url)
        elseif p.source.type == "base64"
            return Dict{String,Any}("type" => "input_image", "image_url" => "data:$(p.source.media_type);base64,$(p.source.data)")
        end
    elseif p.type == "audio" && p.source !== nothing && p.source.type == "base64"
        fmt = split(something(p.source.media_type, "audio/wav"), "/")[end]
        return Dict{String,Any}("type" => "input_audio", "audio" => p.source.data, "format" => fmt)
    elseif p.type == "document" && p.source !== nothing
        if p.source.type == "url"
            return Dict{String,Any}("type" => "input_file", "file_url" => p.source.url)
        elseif p.source.type == "base64"
            return Dict{String,Any}("type" => "input_file", "file_data" => "data:$(p.source.media_type);base64,$(p.source.data)")
        end
    elseif p.type == "tool_result"
        txt = p.content !== nothing ? parts_to_text(p.content) : ""
        return Dict{String,Any}("type" => "input_text", "text" => txt)
    end
    Dict{String,Any}("type" => "input_text", "text" => something(p.text, ""))
end

function message_to_openai_input(m::Message)
    Dict{String,Any}("role" => m.role, "content" => [part_to_openai_input(p) for p in m.parts])
end

function get_nested(d::Dict, keys...; default=nothing)
    val = d
    for k in keys
        if val isa Dict && haskey(val, k)
            val = val[k]
        else
            return default
        end
    end
    val
end

safe_string(v) = v === nothing ? "" : string(v)
safe_int(v) = v === nothing ? 0 : (v isa Number ? Int(v) : 0)
