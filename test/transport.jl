using HTTP: HTTP
using Logging: Logging

function with_local_server(f, handler; streaming=false)
    server = if streaming
        HTTP.listen!(handler, "127.0.0.1", 0; verbose=false, listenany=true)
    else
        HTTP.serve!(handler, "127.0.0.1", 0; verbose=false, listenany=true)
    end
    try
        f("http://127.0.0.1:$(HTTP.port(server))")
    finally
        close(server)
    end
end

@testset "real HTTP transport, headers, error metadata and no retries" begin
    received = Any[]
    mode = Ref(:complete)
    handler = function (request)
        push!(
            received,
            (
                request.target,
                HTTP.header(request, "authorization"),
                JSON.parse(String(copy(request.body))),
            ),
        )
        if mode[] === :error
            return HTTP.Response(
                429,
                [
                    "Content-Type"=>"application/json",
                    "Retry-After"=>"3",
                    "x-request-id"=>"local-429",
                ],
                JSON.serialize(
                    Dict("error"=>Dict("code"=>"rate_limit_exceeded", "message"=>"slow down"))
                ),
            )
        elseif mode[] === :redirect
            return HTTP.Response(307, ["Location"=>"/must-not-follow"])
        elseif mode[] === :stream
            return HTTP.Response(
                200,
                ["Content-Type"=>"text/event-stream"],
                "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n" *
                "data: {\"choices\":[{\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1}}\n\n" *
                "data: [DONE]\n\n",
            )
        end
        HTTP.Response(
            200,
            ["Content-Type"=>"application/json"],
            JSON.serialize(
                Dict(
                    "id"=>"local",
                    "model"=>"local-model",
                    "choices"=>[Dict("message"=>Dict("content"=>"hello"), "finish_reason"=>"stop")],
                    "usage"=>Dict("prompt_tokens"=>2, "completion_tokens"=>1),
                ),
            ),
        )
    end
    with_local_server(handler) do base
        client = OpenAIChatLM(api_key=SENTINEL, base_url=base * "/v1")
        req = Request("local-model", user("hello"))
        logger = Test.TestLogger(min_level=Logging.Debug)
        answer = Logging.with_logger(logger) do
            complete(client, req)
        end
        @test text(answer) == "hello"
        @test answer.usage.total_tokens == 3
        @test received[1][1] == "/v1/chat/completions"
        @test received[1][2] == "Bearer " * SENTINEL
        @test received[1][3]["model"] == "local-model"
        @test !occursin(SENTINEL, sprint(show, logger.logs))

        mode[] = :stream
        answer = stream(client, req) do rs
            @test join(text_chunks(rs)) == "hello"
            response(rs)
        end
        @test answer.usage.total_tokens == 3
        @test received[2][3]["stream"] === true

        mode[] = :error
        before = length(received)
        err = try
            complete(client, req)
            nothing
        catch error
            error
        end
        @test err isa RateLimitError
        @test err.request_id == "local-429"
        @test err.retry_after === 3.0
        @test length(received) == before + 1

        # Redirects are not followed, even to a second path on the same host.
        mode[] = :redirect
        before = length(received)
        wire = build_request(client, req)
        status = open_response(HTTPTransport(), wire) do head, body
            read(body)
            head.status
        end
        @test status == 307
        @test length(received) == before + 1

        failure = ArgumentError("application callback stopped")
        mode[] = :complete
        thrown = try
            open_response(HTTPTransport(), wire) do head, body
                read(body)
                throw(failure)
            end
            nothing
        catch error
            error
        end
        @test thrown === failure
    end
    @test_throws ArgumentError HTTPTransport(read_timeout=0.5)
    @test_throws ArgumentError HTTPTransport(connect_timeout=true)

    # This cloud door uses a query key. A refused local connection must not
    # expose the underlying ConnectError URL through Julia's exception stack.
    access = LM15.provider_definition("vertex-express").access
    client = GeminiLM(api_key=SENTINEL, access=access, base_url="http://127.0.0.1:1/v1")
    stack = try
        complete(client, Request("local-model", user("hi")))
        nothing
    catch error
        @test error isa TransportError
        Base.current_exceptions()
    end
    @test stack !== nothing
    @test !occursin(SENTINEL, sprint(showerror, stack))
end

@testset "live WebSocket connection and do-block return value" begin
    seen = Any[]
    handler = function (stream)
        HTTP.WebSockets.upgrade(stream) do ws
            for _ in 1:3
                push!(seen, JSON.parse(HTTP.WebSockets.receive(ws)))
            end
            HTTP.WebSockets.send(
                ws,
                JSON.serialize(Dict("type"=>"response.output_text.delta", "delta"=>"hello live")),
            )
            HTTP.WebSockets.send(
                ws,
                JSON.serialize(
                    Dict(
                        "type"=>"response.done",
                        "response"=>Dict(
                            "status"=>"completed",
                            "output"=>[],
                            "usage"=>Dict("input_tokens"=>2, "output_tokens"=>1),
                        ),
                    ),
                ),
            )
        end
    end
    with_local_server(handler; streaming=true) do base
        client = OpenAILM(api_key=SENTINEL, base_url=base * "/v1")
        answer = live(client, LiveConfig(model="local-realtime")) do session
            send_text!(session, "hello")
            result(turn(session))
        end
        @test answer isa Turn
        @test answer.text == "hello live"
        @test answer.usage.total_tokens == 3
        @test ok(answer)
        @test seen[1]["type"] == "session.update"
        @test seen[2]["type"] == "conversation.item.create"
        @test seen[3]["type"] == "response.create"
    end
end
