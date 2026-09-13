@testset "private atomic storage preserves other fields and failed mutations" begin
    mktempdir() do dir
        withenv("LM15_LOCK_DIR"=>joinpath(dir, "locks")) do
            path = joinpath(dir, "credentials.json")
            store = CredentialFileStore(path)
            mutate!(store, "one") do _
                Dict("access"=>SENTINEL, "empty"=>Dict(), "null"=>nothing)
            end
            mutate!(store, "two") do _
                Dict("access"=>"other")
            end
            original = read(path, String)
            @test_throws ErrorException mutate!(store, "one") do current
                current["access"] = "must-not-persist"
                error("abort mutation")
            end
            @test read(path, String) == original
            data = JSON.parse(original)
            @test data["one"]["empty"] == Dict()
            @test data["one"]["null"] === nothing
            @test data["two"]["access"] == "other"
            if !Sys.iswindows()
                @test filemode(path) & 0o777 == 0o600
            end
            delete_credential!(store, "one")
            @test !haskey(JSON.parse(read(path, String)), "one")
            @test !any(name->endswith(name, ".tmp"), readdir(dir))
        end
    end
end

@testset "refresh under a real file lock; reuse the fresh token" begin
    mktempdir() do dir
        withenv("LM15_LOCK_DIR"=>joinpath(dir, "locks")) do
            path = joinpath(dir, "claude.json")
            original = Dict(
                "foreign"=>Dict("keep"=>nothing),
                "claudeAiOauth"=>Dict(
                    "accessToken"=>"expired",
                    "refreshToken"=>"old-refresh",
                    "expiresAt"=>1,
                    "keep"=>17,
                ),
            )
            write(path, JSON.serialize(original))
            refreshed = Ref(0)
            refresh = function (provider, previous)
                refreshed[] += 1
                @test provider == "claude-code"
                @test previous.refresh_token == "old-refresh"
                LocalOAuthCredential(
                    "fresh", "rotated", round(Int64, time()*1000)+3_600_000, nothing
                )
            end
            one = LM15.get_local_credential("claude-code", path; refresh_fn=refresh)
            two = LM15.get_local_credential("claude-code", path; refresh_fn=refresh)
            @test one.access_token == two.access_token == "fresh"
            @test refreshed[] == 1
            saved = JSON.parse(read(path, String))
            @test saved["claudeAiOauth"]["refreshToken"] == "rotated"
            @test saved["claudeAiOauth"]["keep"] == 17
            @test saved["foreign"]["keep"] === nothing

            write(path, JSON.serialize(original))
            before = read(path, String)
            @test_throws AuthError LM15.get_local_credential(
                "claude-code", path; refresh_fn=(args...)->error("refresh failed with "*SENTINEL)
            )
            @test read(path, String) == before
            delete!(original["claudeAiOauth"], "refreshToken")
            write(path, JSON.serialize(original))
            @test_throws AuthError LM15.get_local_credential(
                "claude-code", path; refresh_fn=refresh
            )
            @test refreshed[] == 1
        end
    end
end

@testset "credential locks contend across Julia processes and release cleanly" begin
    mktempdir() do dir
        withenv("LM15_LOCK_DIR"=>joinpath(dir, "locks")) do
            path = joinpath(dir, "credentials.json")
            code = """
                using LM15
                try
                    LM15.hold_file_lock(ARGS[1]; timeout=0.05) do
                        exit(2)
                    end
                catch error
                    error isa LM15.LockTimeoutError || rethrow()
                    error.code == "lock_timeout" || exit(3)
                    LM15.retryable(error) || exit(4)
                    println("lock_timeout")
                end
                """
            command = `$(Base.julia_cmd()) --startup-file=no --history-file=no --project=$(dirname(Base.active_project())) -e $code $path`
            output = LM15.hold_file_lock(path) do
                read(command, String)
            end
            @test strip(output) == "lock_timeout"
            acquired = LM15.hold_file_lock(path; timeout=0.2) do
                true
            end
            @test acquired
        end
    end
end
