# Bounded live-turn collection, against the contract's consumer vectors
# (lm15-contract consumer/live-collection-limits.json, 2026-09-15).
mutable struct ScriptedSession
    events::Vector{Any}
    reads::Int
end
function LM15.recv(s::ScriptedSession)
    s.reads += 1
    return s.events[s.reads]
end

@testset "live turn collection limits (consumer vectors)" begin
    path = joinpath(CONTRACT, "consumer", "live-collection-limits.json")
    if !isfile(path)
        @info "lm15-contract not found next to the package; the consumer vectors were not run" path
    else
        vectors = JSON.parse(read(path, String))
        @test vectors["defaults"]["max_bytes"] == LM15.DEFAULT_TURN_MAX_BYTES
        @test vectors["defaults"]["max_events"] == LM15.DEFAULT_TURN_MAX_EVENTS
        for case in vectors["cases"]
            events = [from_dict(LiveServerEvent, e) for e in case["events"]]
            session = ScriptedSession(events, 0)
            limits = case["limits"]
            view = LM15.turn_view(session; max_bytes=limits["max_bytes"], max_events=limits["max_events"])
            failure = nothing
            try
                for _ in view
                end
            catch e
                e isa CollectionLimitError || rethrow()
                failure = e
            end
            expect = case["expect"]
            @testset "$(case["id"])" begin
                @test length(view.events) == expect["accepted"]
                @test session.reads == expect["reads"]
                @test view.retained_bytes == expect["retained_bytes"]
                if expect["limit"] === nothing
                    @test failure === nothing
                else
                    @test failure isa CollectionLimitError
                    @test failure.limit == expect["limit"]
                    @test failure.code == "collection_limit"
                    @test !retryable(failure)
                    @test failure.retained_events == expect["accepted"]
                    if expect["rejected_index"] === nothing
                        @test failure.rejected_event === nothing
                    else
                        @test failure.rejected_event == events[expect["rejected_index"] + 1]
                    end
                    # Sealed: no further read, the same failure, an incomplete snapshot.
                    reads = session.reads
                    @test_throws CollectionLimitError iterate(view)
                    @test session.reads == reads
                    @test LM15.snapshot(view).ended_by == "incomplete"
                    @test failure.partial.ended_by == "incomplete"
                end
            end
        end
    end
    @test_throws ArgumentError LM15.turn_view(ScriptedSession([], 0); max_bytes=0)
    @test_throws ArgumentError LM15.turn_view(ScriptedSession([], 0); max_events=true)
end
