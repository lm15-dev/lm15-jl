@testset "direct-client keys are explicit but profile and login paths are honored" begin
    withenv("OPENAI_API_KEY"=>SENTINEL) do
        @test_throws NotConfiguredError OpenAILM()
        @test_throws NotConfiguredError OpenAIChatLM(compat="ollama")
    end

    mktempdir() do dir
        credentials = joinpath(dir, "aws-credentials")
        config = joinpath(dir, "aws-config")
        write(
            credentials,
            """
[default]
aws_access_key_id = wrong-default-account
aws_secret_access_key = default-secret
[work]
aws_access_key_id = selected-work-account
aws_secret_access_key = work-secret
""",
        )
        write(config, "")
        withenv(
            "AWS_PROFILE"=>"work",
            "AWS_REGION"=>"us-west-2",
            "AWS_SHARED_CREDENTIALS_FILE"=>credentials,
            "AWS_CONFIG_FILE"=>config,
            "AWS_ACCESS_KEY_ID"=>nothing,
            "AWS_SECRET_ACCESS_KEY"=>nothing,
            "AWS_BEARER_TOKEN_BEDROCK"=>nothing,
            "AWS_WEB_IDENTITY_TOKEN_FILE"=>nothing,
            "AWS_ROLE_ARN"=>nothing,
            "AWS_EC2_METADATA_DISABLED"=>"true",
        ) do
            client = ProviderLM("bedrock-chat")
            credential = resolve_credential(client.credential)
            @test credential isa AwsCredentials
            @test credential.access_key_id == "selected-work-account"
            @test client.settings["region"] == "us-west-2"
        end

        path = joinpath(dir, "custom-login.json")
        write(
            path,
            JSON.serialize(
                Dict(
                    "xai"=>Dict(
                        "type"=>"oauth",
                        "access"=>"subscription-token",
                        "refresh"=>"refresh-token",
                        "expires"=>round(Int64, time()*1000)+3_600_000,
                    ),
                ),
            ),
        )
        withenv("LM15_CREDENTIALS_PATH"=>path, "XAI_API_KEY"=>SENTINEL) do
            client = XaiLM()
            @test client.credentials_source === :stored
            @test resolve_credential(client.credential).value == "subscription-token"
        end
    end
end
