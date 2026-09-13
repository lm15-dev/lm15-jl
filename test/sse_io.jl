# Simulate an HTTP chunk header arriving before its payload. read(io, n) may
# temporarily return no bytes without reaching EOF; the next packet is still valid.
mutable struct PacketBoundaryIO <: IO
    body::IOBuffer
    header_pending::Bool
end
Base.eof(io::PacketBoundaryIO) = eof(io.body)
Base.bytesavailable(io::PacketBoundaryIO) = io.header_pending ? 0 : min(1, bytesavailable(io.body))
function Base.read(io::PacketBoundaryIO, count::Integer)
    if io.header_pending
        io.header_pending = false
        return UInt8[]
    end
    return read(io.body, min(1, count))
end

@testset "SSE handles empty packet boundaries and split UTF-8" begin
    io = PacketBoundaryIO(IOBuffer("data: hé🙂\r\n\r\n"), true)
    frames = parse_sse(io)
    @test length(frames) == 1
    @test only(frames).data == "hé🙂"
    @test eof(io)
end
