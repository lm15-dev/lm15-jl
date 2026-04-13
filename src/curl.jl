# Dump lm15 requests as curl commands or structured HTTP for comparison.

const _AUTH_HEADERS = Set(["authorization", "x-api-key", "x-goog-api-key"])

function _build_lm_request(model::String, prompt; messages=nothing, system=nothing, tools=Tool[],
                           reasoning=nothing, prefill=nothing, output=nothing,
                           prompt_caching=false, temperature=nothing, max_tokens=nothing,
                           top_p=nothing, stop=nothing, provider_config=nothing)
    final_messages = if messages !== nothing
        Message[m for m in messages]
    elseif prompt !== nothing
        [UserMessage(prompt)]
    else
        throw(ProviderError("either prompt or messages is required"))
    end

    prefill !== nothing && push!(final_messages, AssistantMessage(prefill))

    provider_cfg = Dict{String,Any}()
    prompt_caching && (provider_cfg["prompt_caching"] = true)
    output !== nothing && (provider_cfg["output"] = output)

    # Merge provider_config passthrough
    if provider_config !== nothing
        for (k, v) in provider_config
            provider_cfg[k] = v
        end
    end

    reasoning_cfg = if reasoning === true
        Dict{String,Any}("enabled" => true)
    elseif reasoning isa Dict
        merge(Dict{String,Any}("enabled" => true), reasoning)
    else
        nothing
    end

    cfg = Config(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
        reasoning=reasoning_cfg,
        provider=isempty(provider_cfg) ? nothing : provider_cfg,
    )

    LMRequest(model=model, messages=final_messages, system=system, tools=tools, config=cfg)
end

"""
    build_http_request(model, prompt=nothing; kwargs...) -> HttpRequest

Build the provider-level HTTP request without sending it.
"""
function build_http_request(model::String, prompt=nothing; stream=false, provider=nothing,
                            api_key=nothing, env=nothing, kwargs...)
    request = _build_lm_request(model, prompt; kwargs...)
    provider_name = provider !== nothing ? provider : resolve_provider(model)
    client = build_default(api_key=api_key, env=env)
    adapter = resolve_adapter(client, model, provider_name)
    build_request(adapter, request, stream)
end

"""Convert an HttpRequest to a JSON-serializable Dict with redacted auth headers."""
function http_request_to_dict(req::HttpRequest)
    body = nothing
    if req.body !== nothing
        body = try
            JSON.parse(String(req.body))
        catch
            "<binary>"
        end
    end

    headers = Dict{String,String}()
    for (k, v) in req.headers
        headers[k] = lowercase(k) in _AUTH_HEADERS ? "REDACTED" : v
    end

    Dict{String,Any}(
        "method" => req.method,
        "url" => req.url,
        "headers" => headers,
        "params" => isempty(req.params) ? nothing : Dict(req.params),
        "body" => body,
    )
end

_shell_quote(s::String) = "'" * replace(s, "'" => "'\\''") * "'"

"""Convert an HttpRequest to a curl command string."""
function http_request_to_curl(req::HttpRequest; redact_auth::Bool=true)
    parts = String["curl"]

    req.method != "GET" && push!(parts, "-X $(req.method)")

    url = req.url
    if !isempty(req.params)
        qs = join(["$k=$v" for (k, v) in req.params], "&")
        url *= "?" * qs
    end
    push!(parts, _shell_quote(url))

    for (k, v) in req.headers
        value = redact_auth && lowercase(k) in _AUTH_HEADERS ? "REDACTED" : v
        push!(parts, "-H $(_shell_quote("$k: $value"))")
    end

    if req.body !== nothing
        body = try
            JSON.serialize(JSON.parse(String(req.body)))
        catch
            nothing
        end
        if body === nothing
            push!(parts, "--data-binary @-")
        else
            push!(parts, "-d $(_shell_quote(body))")
        end
    end

    join(parts, " \\\n  ")
end

"""Build a curl command for the given call parameters."""
function dump_curl(model::String, prompt=nothing; redact_auth::Bool=true, kwargs...)
    req = build_http_request(model, prompt; kwargs...)
    http_request_to_curl(req; redact_auth=redact_auth)
end

"""Build the structured HTTP request dump for comparison."""
function dump_http(model::String, prompt=nothing; kwargs...)
    req = build_http_request(model, prompt; kwargs...)
    http_request_to_dict(req)
end
