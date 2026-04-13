# Middleware for complete operations.

function with_retries(max_retries::Int, sleep_base::Float64=0.2)
    (req, next) -> begin
        last_err = nothing
        for i in 0:max_retries
            try
                return next(req)
            catch e
                (i == max_retries || !is_transient(e)) && rethrow()
                last_err = e
                sleep(sleep_base * (1 << i))
            end
        end
        throw(last_err)
    end
end

function with_cache(cache::Dict{String,LMResponse})
    (req, next) -> begin
        key = string(req.model, "|", req.messages)
        haskey(cache, key) && return cache[key]
        resp = next(req)
        cache[key] = resp
        resp
    end
end

struct MiddlewareHistoryEntry
    model::String
    messages::Int
    finish_reason::String
    usage::Usage
end

function with_history(history::Vector{MiddlewareHistoryEntry})
    (req, next) -> begin
        resp = next(req)
        push!(history, MiddlewareHistoryEntry(req.model, length(req.messages), resp.finish_reason, resp.usage))
        resp
    end
end
