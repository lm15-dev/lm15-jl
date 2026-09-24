# MAP-13 — adapt freely, never invisibly; refuse only when a guess could hurt.
#
# When a wire cannot take a setting as asked, the adapter does the obvious
# thing and records it: the record rides the Response and the first stream
# event, and `plan` previews it with no network. Translations (stop becoming
# stop_sequences, an effort word becoming a budget) are the adapter's ordinary
# job and are never recorded. The policy is set on the client and the router:
#   "note" (default): adapt and record.
#   "silent": adapt the same way; the response carries no record.
#   "refuse": every deviation (dropped, clamped, substituted, client_side) is an
#             UnsupportedFeatureError before the wire, naming the config path.
# satisfied and defaulted change nothing the caller asked for and are recorded
# under every policy but "silent". Nothing here prints.

const ADAPTATION_POLICIES = ("note", "silent", "refuse")
const ADAPTATION_ACTIONS = ("dropped", "clamped", "substituted", "client_side", "satisfied", "defaulted")
const DEVIATIONS = ("dropped", "clamped", "substituted", "client_side")
const ACTION_VERBS = Dict(
    "dropped"=>"would be dropped",
    "clamped"=>"would be clamped",
    "substituted"=>"would be substituted",
    "client_side"=>"would be applied client-side",
)

function check_policy(policy)
    policy in ADAPTATION_POLICIES ||
        throw(ArgumentError("adaptations must be one of \"note\", \"silent\", \"refuse\""))
    return String(policy)
end

mutable struct AdaptationScope
    policy::String
    provider::Union{Nothing,String}
    records::Vector{Adaptation}
end

# One scope per request build, in task-local storage (one task builds one request at a time).
current_scope() = get(task_local_storage(), :lm15_adaptation_scope, nothing)
function collecting(build, policy, provider)
    scope=AdaptationScope(check_policy(policy), provider, Adaptation[])
    value=task_local_storage(build, :lm15_adaptation_scope, scope)
    return value, Tuple(scope.records)
end

function adapt!(field, action, reason; asked=nothing, applied=nothing, provider=nothing)
    scope=current_scope()
    policy=scope===nothing ? "note" : scope.policy
    who=something(provider, scope===nothing ? nothing : scope.provider, Some(nothing))
    if policy=="refuse" && action in DEVIATIONS
        head=who===nothing ? "" : "$(who): "
        throw(UnsupportedFeatureError(
            "$(head)$(field) $(ACTION_VERBS[action]): $(reason) (adaptations=\"refuse\")";
            provider=who, feature=String(field),
        ))
    end
    scope===nothing || push!(scope.records, Adaptation(; field, action, reason, asked, applied))
    return nothing
end

visible_adaptations(l, records) = l.adaptations=="silent" ? () : records
client_side_stop(records) = any(a->a.field=="config.stop" && a.action=="client_side", records)

# The effort ladder, and the nearest declared level to an asked one. A tie goes
# to the lower level: the cheaper guess.
const EFFORT_LADDER = ("minimal", "low", "medium", "high", "xhigh", "max")
function nearest_effort(asked, available)
    levels=[l for l in available if l in EFFORT_LADDER]
    isempty(levels) && throw(ArgumentError("no comparable effort levels"))
    asked in levels && return String(asked)
    want=something(findfirst(==(asked), EFFORT_LADDER), 1)
    index(l)=findfirst(==(l), EFFORT_LADDER)
    return String(first(sort(levels; by=l->(abs(index(l)-want), index(l)))))
end
