struct ScriptedEvents
    items::Vector{Any}
    closes::Base.RefValue{Int}
    close_error::Union{Nothing,Exception}
end
ScriptedEvents(items; close_error=nothing) = ScriptedEvents(Any[items...], Ref(0), close_error)
Base.IteratorSize(::Type{ScriptedEvents}) = Base.SizeUnknown()
function Base.iterate(source::ScriptedEvents, index=1)
    index > length(source.items) && return nothing
    value = source.items[index]
    value isa Exception && throw(value)
    return value, index + 1
end
function Base.close(source::ScriptedEvents)
    source.closes[] += 1
    source.close_error === nothing || throw(source.close_error)
    return nothing
end

@testset "stream source closes once, including early and failed exits" begin
    req = Request("m", user("hi"))
    fragment = StreamDeltaEvent(delta=TextDelta(text="hello"))
    source = ScriptedEvents([
        StreamStartEvent(model="m"), fragment, StreamEndEvent(finish_reason="stop")
    ])
    rs = ResponseStream(source, req)
    @test join(text_chunks(rs)) == "hello"
    @test text(response(rs)) == "hello"
    close(rs)
    close(rs)
    @test source.closes[] == 1

    source = ScriptedEvents([fragment])
    rs = ResponseStream(source, req)
    @test_throws StreamAssemblyError response(rs)
    @test source.closes[] == 1

    source = ScriptedEvents([fragment, StreamEndEvent()])
    rs = ResponseStream(source, req)
    iterate(rs)
    close(rs)
    @test_throws StreamAssemblyError response(rs)
    @test source.closes[] == 1
end

@testset "cleanup never overwrites the answer or original error" begin
    req = Request("m", user("hi"))
    fragment = StreamDeltaEvent(delta=TextDelta(text="hello"))
    cleanup = ErrorException("cleanup failed")
    source = ScriptedEvents([fragment, StreamEndEvent()]; close_error=cleanup)
    rs = ResponseStream(source, req)
    answer = @test_logs (:warn, r"Stream cleanup failed") response(rs)
    @test text(answer) == "hello"
    @test rs.cleanup_errors == [cleanup]
    @test source.closes[] == 1

    primary = TransportError("read failed")
    source = ScriptedEvents([fragment, primary]; close_error=cleanup)
    rs = ResponseStream(source, req)
    @test_logs (:warn, r"Stream cleanup failed") begin
        try
            response(rs)
            @test false
        catch error
            @test error === primary
        end
    end
    @test primary.cleanup_errors == [cleanup]
    @test source.closes[] == 1

    source = ScriptedEvents([fragment, StreamEndEvent(), fragment])
    rs = ResponseStream(source, req)
    @test_throws StreamAssemblyError response(rs)
    @test text(rs.failure.partial) == "hello"
    @test source.closes[] == 1
end

@testset "scoped ResponseStream closes on an application exception" begin
    req = Request("m", user("hi"))
    source = ScriptedEvents([StreamEndEvent()])
    failure = ErrorException("application stopped")
    try
        ResponseStream(source, req) do rs
            throw(failure)
        end
        @test false
    catch error
        @test error === failure
    end
    @test source.closes[] == 1
end

struct MemoryTransport <: AbstractTransport
    bytes::Vector{UInt8}
    opens::Base.RefValue{Int}
    closes::Base.RefValue{Int}
end
function LM15.open_response(f, transport::MemoryTransport, request::WireRequest)
    transport.opens[] += 1
    io = IOBuffer(transport.bytes; read=true, write=false)
    try
        f(HttpResponse(; status=200), io)
    finally
        close(io)
        transport.closes[] += 1
    end
end

@testset "custom streaming transports and scoped client calls" begin
    raw = """
        data: {"choices":[{"delta":{"content":"hello"}}]}

        data: {"choices":[{"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}

        data: [DONE]

        """
    transport = MemoryTransport(Vector{UInt8}(codeunits(raw)), Ref(0), Ref(0))
    client = OpenAIChatLM(api_key="explicit", transport=transport)
    req = Request("m", user("hi"))
    source = stream(client, req)
    @test transport.opens[] == 0
    answer = ResponseStream(response, source, req)
    @test text(answer) == "hello"
    @test transport.opens[] == transport.closes[] == 1
end

include("sse_io.jl")

@testset "SSE accepts Unix, Windows and bare carriage-return lines" begin
    for newline in ("\n", "\r\n", "\r")
        frames = parse_sse(IOBuffer(join(["data: hello", "", ""], newline)))
        @test length(frames) == 1
        @test only(frames).data == "hello"
    end
end
