const GCP_SCOPE="https://www.googleapis.com/auth/cloud-platform"
const JWT_BEARER="urn:ietf:params:oauth:grant-type:jwt-bearer"
const CLIENT_ASSERTION="urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
Base.@kwdef struct ChainContext
    env::AbstractDict = ENV
    home::String = get(env, "HOME", homedir())
    files::Union{Nothing,AbstractDict} = nothing
    settings::AbstractDict = Dict{String,String}()
    clock::Function = time
    http::Any = nothing
    command::Any = nothing
end
Base.show(io::IO, ::ChainContext) = print(io, "ChainContext(<credential sources withheld>)")
function cloud_path(ctx, path)
    return if startswith(path, "~/") || startswith(path, "~\\")
        joinpath(ctx.home, path[3:end])
    else
        String(path)
    end
end
function cloud_read(ctx, path)
    path=cloud_path(ctx, path)
    if ctx.files!==nothing
        for (key, value) in ctx.files
            normpath(cloud_path(ctx, key))==normpath(path) && return String(value)
        end
        return nothing
    end
    try
        read(path, String)
    catch
        nothing
    end
end
function cloud_json(ctx, path)
    raw=cloud_read(ctx, path)
    raw===nothing && return nothing
    try
        d=JSON.parse(raw)
        d isa AbstractDict || throw(ArgumentError("not object"))
        d
    catch
        throw(NotConfiguredError("malformed cloud credential file"))
    end
end
function cloud_http(ctx, method, url, headers=Dict{String,String}(), body=nothing; timeout=30)
    if ctx.http!==nothing
        return ctx.http(method, url, headers, body, timeout)
    end
    try
        r=http_request(
            method,
            url,
            collect(headers),
            body===nothing ? UInt8[] : body;
            status_exception=false,
            retry=false,
            redirect=false,
            readtimeout=timeout,
            connect_timeout=timeout,
        )
        return Int(r.status), Dict(lowercase(k)=>v for (k, v) in r.headers), r.body
    catch e
        e isa InterruptException && rethrow()
        throw(AuthError("cloud credential connection failed"))
    end
end
function cloud_exchange(ctx, url, payload; form=false, headers=Dict{String,String}())
    content=form ? form_encode(payload) : JSON.serialize(payload)
    headers=merge(
        Dict("content-type"=>form ? "application/x-www-form-urlencoded" : "application/json"),
        headers,
    )
    status, _, raw=cloud_http(ctx, "POST", url, headers, content)
    200<=status<300 || throw(AuthError("cloud token exchange rejected (HTTP $status)"))
    d=try
        JSON.parse(String(raw))
    catch
        throw(AuthError("cloud token exchange returned malformed JSON"))
    end
    d isa AbstractDict || throw(AuthError("cloud token exchange returned a non-object"))
    return d
end
function oauth_credential(d, ctx)
    value=string_field(d, "access_token")
    value===nothing && throw(AuthError("token response contains no access token"))
    expiry=nothing
    if get(d, "expires_on", nothing)!==nothing
        t=tryparse(Float64, string(d["expires_on"]))
        t===nothing || !isfinite(t) || (expiry=normalized_time(t))
    end
    if expiry===nothing && get(d, "expires_in", nothing)!==nothing
        t=tryparse(Float64, string(d["expires_in"]))
        t===nothing || !isfinite(t) || t<0 || (expiry=normalized_time(ctx.clock()+t))
    end
    return BearerToken(value; expires_at=expiry)
end
function aws_credential(d)
    key=first_nonempty(string_field(d, "AccessKeyId"), string_field(d, "accessKeyId"))
    secret=first_nonempty(string_field(d, "SecretAccessKey"), string_field(d, "secretAccessKey"))
    key!==nothing && secret!==nothing || throw(AuthError("AWS token response lacks access keys"))
    expires=first_nonempty(
        get(d, "Expiration", nothing), get(d, "expiration", nothing), get(d, "expiresAt", nothing)
    )
    expires isa Real && expires>1e11 && (expires/=1000)
    return AwsCredentials(;
        access_key_id=key,
        secret_access_key=secret,
        session_token=first_nonempty(
            string_field(d, "SessionToken"),
            string_field(d, "sessionToken"),
            string_field(d, "Token"),
        ),
        expires_at=normalized_time(expires),
    )
end
function parse_ini(text)
    out=Dict{String,Dict{String,String}}()
    section=nothing
    text===nothing && return out
    for raw in split(text, '\n')
        line=strip(raw)
        isempty(line) && continue
        startswith(line, "#") ||
            startswith(line, ";") ||
            begin
                if startswith(line, "[") && endswith(line, "]")
                    section=String(strip(line[2:(end - 1)]))
                    get!(out, section, Dict{String,String}())
                else
                    pair=split(line, '='; limit=2)
                    section!==nothing && length(pair)==2 ||
                        throw(NotConfiguredError("malformed AWS profile configuration"))
                    out[section][lowercase(strip(pair[1]))]=String(strip(pair[2]))
                end
            end
    end
    return out
end
function aws_profiles(ctx)
    credentials=parse_ini(
        cloud_read(ctx, get(ctx.env, "AWS_SHARED_CREDENTIALS_FILE", "~/.aws/credentials"))
    )
    config=parse_ini(cloud_read(ctx, get(ctx.env, "AWS_CONFIG_FILE", "~/.aws/config")))
    return credentials, config, get(ctx.env, "AWS_PROFILE", "default")
end
function profile_section(config, name)
    return get(
        config, name=="default" ? name : "profile $name", get(config, name, Dict{String,String}())
    )
end
function aws_static(d)
    key=get(d, "aws_access_key_id", "")
    secret=get(d, "aws_secret_access_key", "")
    return if isempty(key) || isempty(secret)
        nothing
    else
        AwsCredentials(;
            access_key_id=key,
            secret_access_key=secret,
            session_token=string_field(d, "aws_session_token"),
        )
    end
end
function aws_environment(ctx)
    key=get(ctx.env, "AWS_ACCESS_KEY_ID", "")
    secret=get(ctx.env, "AWS_SECRET_ACCESS_KEY", "")
    return if isempty(key) || isempty(secret)
        nothing
    else
        AwsCredentials(;
            access_key_id=key,
            secret_access_key=secret,
            session_token=string_field(ctx.env, "AWS_SESSION_TOKEN"),
        )
    end
end
function cloud_on_path(ctx, command)
    separators=Sys.iswindows() ? ';' : ':'
    extensions=if Sys.iswindows()
        ("", split(get(ctx.env, "PATHEXT", ".COM;.EXE;.BAT;.CMD"), ';')...)
    else
        ("",)
    end
    if occursin('/', command) || occursin('\\', command)
        path=cloud_path(ctx, command)
        present=ctx.files===nothing ? isfile(path) : cloud_read(ctx, path)!==nothing
        return present ? path : nothing
    end
    for dir in split(get(ctx.env, "PATH", ""), separators; keepempty=false), ext in extensions
        path=joinpath(dir, command*ext)
        present=ctx.files===nothing ? isfile(path) : cloud_read(ctx, path)!==nothing
        present && return path
    end
    return nothing
end
function cloud_command(ctx, argv; timeout=30)
    ctx.command===nothing || return ctx.command(argv, timeout)
    executable=cloud_on_path(ctx, first(argv))
    executable===nothing && throw(AuthError("credential command is not available"))
    path, io=mktemp()
    process=nothing
    try
        chmod(path, 0o600)
        command=setenv(
            Cmd([executable; String.(argv[2:end])]), [string(k)*"="*string(v) for (k, v) in ctx.env]
        )
        process=run(pipeline(command; stdout=io, stderr=devnull); wait=false)
        deadline=time_ns()/1e9+timeout
        while process_running(process)
            time_ns()/1e9<deadline ||
                (kill(process); throw(AuthError("credential command timed out")))
            sleep(0.02)
        end
        success(process) || throw(AuthError("credential command failed"))
        flush(io)
        seekstart(io)
        filesize(path)<=16*1024*1024 ||
            throw(AuthError("credential command output exceeds size limit"))
        read(io, String)
    catch e
        e isa Union{AuthError,InterruptException} && rethrow()
        throw(AuthError("credential command failed"))
    finally
        process===nothing || !process_running(process) || kill(process)
        close(io)
        rm(path; force=true)
    end
end
function sts_credentials(raw)
    text=String(raw)
    m=match(r"<(?:\w+:)?Credentials>(.*?)</(?:\w+:)?Credentials>"s, text)
    m===nothing && throw(AuthError("STS response has no Credentials"))
    d=obj()
    for key in ("AccessKeyId", "SecretAccessKey", "SessionToken", "Expiration")
        item=match(Regex("<(?:\\w+:)?$key>(.*?)</(?:\\w+:)?$key>", "s"), m[1])
        item===nothing || (
            d[key]=replace(
                item[1], "&amp;"=>"&", "&lt;"=>"<", "&gt;"=>">", "&quot;"=>"\"", "&apos;"=>"'"
            )
        )
    end
    return aws_credential(d)
end
function sts_exchange(ctx, fields; source=nothing)
    region=get(
        ctx.settings,
        "region",
        get(ctx.env, "AWS_REGION", get(ctx.env, "AWS_DEFAULT_REGION", "us-east-1")),
    )
    url="https://sts.$region.amazonaws.com/"
    body=form_encode(fields)
    headers=Dict("content-type"=>"application/x-www-form-urlencoded")
    source===nothing || (
        headers=sigv4_sign(
            "POST", url, headers, codeunits(body), source, region, "sts"; now=ctx.clock()
        ).headers
    )
    status, _, raw=cloud_http(ctx, "POST", url, headers, body)
    200<=status<300 || throw(AuthError("STS token exchange was rejected"))
    return sts_credentials(raw)
end
function assume_role(ctx, section; seen=Set{String}())
    creds, config, _=aws_profiles(ctx)
    source=nothing
    if haskey(section, "source_profile")
        name=section["source_profile"]
        name in seen && throw(NotConfiguredError("cyclic AWS source_profile chain"))
        push!(seen, name)
        length(seen)<=16 || throw(NotConfiguredError("AWS source_profile chain too deep"))
        base=merge(profile_section(config, name), get(creds, name, Dict{String,String}()))
        source=haskey(base, "role_arn") ? assume_role(ctx, base; seen) : aws_static(base)
    else
        method=get(section, "credential_source", "")
        source=if method=="Environment"
            aws_environment(ctx)
        elseif method=="EcsContainer"
            aws_container(ctx)
        elseif method=="Ec2InstanceMetadata"
            aws_imds(ctx)
        else
            nothing
        end
    end
    source===nothing && throw(NotConfiguredError("AWS assume-role source has no credentials"))
    fields=obj(
        "Action"=>"AssumeRole",
        "Version"=>"2011-06-15",
        "RoleArn"=>section["role_arn"],
        "RoleSessionName"=>get(section, "role_session_name", "lm15-"*randstring(12)),
    )
    haskey(section, "external_id") && (fields["ExternalId"]=section["external_id"])
    haskey(section, "duration_seconds") && (fields["DurationSeconds"]=section["duration_seconds"])
    return sts_exchange(ctx, fields; source)
end
function web_identity_config(ctx, section)
    file=get(
        ctx.env, "AWS_WEB_IDENTITY_TOKEN_FILE", get(section, "web_identity_token_file", nothing)
    )
    role=get(ctx.env, "AWS_ROLE_ARN", get(section, "role_arn", nothing))
    return if file===nothing || role===nothing
        nothing
    else
        (
            file,
            role,
            get(
                ctx.env,
                "AWS_ROLE_SESSION_NAME",
                get(section, "role_session_name", "lm15-"*randstring(12)),
            ),
        )
    end
end
function aws_web_identity(ctx, config)
    file, role, session=config
    token=cloud_read(ctx, file)
    token===nothing && throw(NotConfiguredError("AWS web identity token file is unreadable"))
    return sts_exchange(
        ctx,
        obj(
            "Action"=>"AssumeRoleWithWebIdentity",
            "Version"=>"2011-06-15",
            "RoleArn"=>role,
            "RoleSessionName"=>session,
            "WebIdentityToken"=>strip(token),
        ),
    )
end
function sso_config(config, section)
    if haskey(section, "sso_session")
        session=section["sso_session"]
        sso=get(config, "sso-session $session", Dict{String,String}())
        haskey(sso, "sso_start_url") || return nothing
        return merge(sso, section, Dict("cache_key"=>bytes2hex(sha1(session))))
    elseif haskey(section, "sso_start_url")
        return merge(section, Dict("cache_key"=>bytes2hex(sha1(section["sso_start_url"]))))
    end
    return nothing
end
function aws_sso(ctx, cfg)
    d=cloud_json(ctx, "~/.aws/sso/cache/$(cfg["cache_key"]).json")
    d===nothing && throw(NotConfiguredError("no cached SSO token; run aws sso login"))
    token=string_field(d, "accessToken")
    expires=get(d, "expiresAt", nothing)
    region=get(cfg, "sso_region", "us-east-1")
    if token===nothing || (expires!==nothing && expiry_seconds(expires)-ctx.clock()<=300)
        all(k->string_field(d, k)!==nothing, ("refreshToken", "clientId", "clientSecret")) ||
            throw(NotConfiguredError("SSO token expired; run aws sso login"))
        refreshed=cloud_exchange(
            ctx,
            "https://oidc.$region.amazonaws.com/token",
            obj(
                "clientId"=>d["clientId"],
                "clientSecret"=>d["clientSecret"],
                "grantType"=>"refresh_token",
                "refreshToken"=>d["refreshToken"],
            ),
        )
        token=string_field(refreshed, "accessToken")
        token===nothing && throw(AuthError("SSO refresh returned no access token"))
    end
    all(k->haskey(cfg, k), ("sso_role_name", "sso_account_id")) ||
        throw(NotConfiguredError("SSO profile needs account and role"))
    url=with_query(
        "https://portal.sso.$region.amazonaws.com/federation/credentials",
        obj("role_name"=>cfg["sso_role_name"], "account_id"=>cfg["sso_account_id"]),
    )
    status, _, raw=cloud_http(ctx, "GET", url, Dict("x-amz-sso_bearer_token"=>token))
    status==200 || throw(AuthError("SSO role credentials rejected"))
    return aws_credential(getobject(JSON.parse(String(raw)), "roleCredentials"))
end
function aws_login_cached(ctx, section)
    session=get(section, "login_session", nothing)
    session===nothing && return nothing
    directory=get(ctx.env, "AWS_LOGIN_CACHE_DIRECTORY", "~/.aws/login/cache")
    d=cloud_json(ctx, directory*"/"*bytes2hex(sha256(session))*".json")
    d===nothing && return nothing
    raw=getobject(d, "accessToken")
    return isempty(raw) ? nothing : aws_credential(raw)
end
function container_url(ctx)
    relative=get(ctx.env, "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", nothing)
    if relative!==nothing
        startswith(relative, "/") &&
        !startswith(relative, "//") &&
        !occursin(r"[\\\r\n\t#]", relative) ||
            throw(NotConfiguredError("invalid container relative credential URI"))
        return "http://169.254.170.2"*relative
    end
    url=get(ctx.env, "AWS_CONTAINER_CREDENTIALS_FULL_URI", nothing)
    url===nothing && return nothing
    uri=HTTP.URI(url)
    host=lowercase(uri.host)
    uri.scheme in ("http", "https") &&
    !isempty(host) &&
    isempty(uri.userinfo) &&
    isempty(uri.fragment) || throw(NotConfiguredError("invalid container credential URI"))
    allowed=host in (
        "169.254.170.2",
        "169.254.170.23",
        "fd00:ec2::23",
        "[fd00:ec2::23]",
        "localhost",
        "::1",
        "[::1]",
    ) || occursin(r"^127\.\d{1,3}\.\d{1,3}\.\d{1,3}$", host)
    uri.scheme=="https" ||
        allowed ||
        throw(NotConfiguredError("container credential host is outside the metadata allowlist"))
    return url
end
function aws_container(ctx)
    url=container_url(ctx)
    url===nothing && return nothing
    headers=Dict{String,String}()
    token=get(ctx.env, "AWS_CONTAINER_AUTHORIZATION_TOKEN", nothing)
    if token===nothing && haskey(ctx.env, "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE")
        token=cloud_read(ctx, ctx.env["AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE"])
        token===nothing &&
            throw(NotConfiguredError("container authorization token file is unreadable"))
        token=strip(token)
    end
    token===nothing || (headers["Authorization"]=token)
    status, _, raw=cloud_http(ctx, "GET", url, headers; timeout=5)
    status==200 || throw(AuthError("container credential request rejected"))
    return aws_credential(JSON.parse(String(raw)))
end
function aws_imds(ctx)
    lowercase(get(ctx.env, "AWS_EC2_METADATA_DISABLED", ""))=="true" && return nothing
    ipv6=lowercase(get(ctx.env, "AWS_EC2_METADATA_SERVICE_ENDPOINT_MODE", ""))=="ipv6"
    base=rstrip(
        get(
            ctx.env,
            "AWS_EC2_METADATA_SERVICE_ENDPOINT",
            ipv6 ? "http://[fd00:ec2::254]" : "http://169.254.169.254",
        ),
        '/',
    )
    status, _, token=try
        cloud_http(
            ctx,
            "PUT",
            base*"/latest/api/token",
            Dict("X-aws-ec2-metadata-token-ttl-seconds"=>"21600");
            timeout=1,
        )
    catch e
        e isa AuthError || rethrow()
        return nothing
    end
    status==200 || return nothing
    headers=Dict("X-aws-ec2-metadata-token"=>String(token))
    status, _, raw=cloud_http(
        ctx, "GET", base*"/latest/meta-data/iam/security-credentials/", headers; timeout=1
    )
    status==200 || return nothing
    role=strip(String(raw))
    isempty(role) && return nothing
    status, _, raw=cloud_http(
        ctx,
        "GET",
        base*"/latest/meta-data/iam/security-credentials/"*percent_encode(first(split(role, '\n'))),
        headers;
        timeout=1,
    )
    status==200 || return nothing
    d=JSON.parse(String(raw))
    get(d, "Code", nothing) in (nothing, "Success") ||
        throw(AuthError("IMDS rejected credential request"))
    return aws_credential(d)
end

azure_scope(ctx) = get(ctx.settings, "scope", "https://ai.azure.com/.default")
function azure_token_url(ctx, tenant)
    return rstrip(
        get(
            ctx.settings,
            "authority_host",
            get(ctx.env, "AZURE_AUTHORITY_HOST", "https://login.microsoftonline.com"),
        ),
        '/',
    )*"/"*percent_encode(tenant)*"/oauth2/v2.0/token"
end
function azure_assertion(
    ctx, tenant, client, pem; jti=randstring(RandomDevice(), 32), send_chain=false
)
    der=certificate_der(pem)
    header=obj("alg"=>"RS256", "typ"=>"JWT", "x5t"=>base64url(sha1(der)))
    send_chain && (header["x5c"]=[base64encode(der)])
    now=floor(Int, ctx.clock())
    payload=obj(
        "aud"=>azure_token_url(ctx, tenant),
        "iss"=>client,
        "sub"=>client,
        "exp"=>now+600,
        "iat"=>now,
        "jti"=>jti,
    )
    return jwt_encode(header, payload, pem)
end
function azure_environment_fields(ctx; jti=nothing)
    e=ctx.env
    tenant=get(e, "AZURE_TENANT_ID", nothing)
    client=get(e, "AZURE_CLIENT_ID", nothing)
    if tenant===nothing || client===nothing
        throw(NotConfiguredError("Azure tenant and client id are required"))
    else
        nothing
    end
    fields=obj("client_id"=>client, "scope"=>azure_scope(ctx))
    if haskey(e, "AZURE_CLIENT_SECRET")
        fields["client_secret"]=e["AZURE_CLIENT_SECRET"]
    elseif haskey(e, "AZURE_CLIENT_CERTIFICATE_PATH")
        haskey(e, "AZURE_CLIENT_CERTIFICATE_PASSWORD") &&
            throw(NotConfiguredError("decrypt certificate before use with openssl pkey"))
        pem=cloud_read(ctx, e["AZURE_CLIENT_CERTIFICATE_PATH"])
        pem===nothing && throw(NotConfiguredError("Azure certificate is unreadable"))
        fields["client_assertion_type"]=CLIENT_ASSERTION
        fields["client_assertion"]=azure_assertion(
            ctx,
            tenant,
            client,
            pem;
            jti=jti===nothing ? randstring(RandomDevice(), 32) : jti,
            send_chain=lowercase(get(e, "AZURE_CLIENT_SEND_CERTIFICATE_CHAIN", "")) in
                       ("1", "true"),
        )
    else
        throw(NotConfiguredError("Azure environment needs a secret or certificate"))
    end
    fields["grant_type"]="client_credentials"
    return azure_token_url(ctx, tenant), fields
end
function azure_workload(ctx)
    e=ctx.env
    token=cloud_read(ctx, e["AZURE_FEDERATED_TOKEN_FILE"])
    token===nothing && throw(NotConfiguredError("Azure federated token file is unreadable"))
    fields=obj(
        "client_id"=>e["AZURE_CLIENT_ID"],
        "scope"=>azure_scope(ctx),
        "client_assertion_type"=>CLIENT_ASSERTION,
        "client_assertion"=>strip(token),
        "grant_type"=>"client_credentials",
    )
    return oauth_credential(
        cloud_exchange(ctx, azure_token_url(ctx, e["AZURE_TENANT_ID"]), fields; form=true), ctx
    )
end
function azure_msi_flavor(ctx)
    e=ctx.env
    if haskey(e, "IDENTITY_ENDPOINT")
        haskey(e, "IDENTITY_HEADER") &&
            return haskey(e, "IDENTITY_SERVER_THUMBPRINT") ? "service-fabric" : "app-service"
        haskey(e, "IMDS_ENDPOINT") && return "azure-arc"
    end
    return if haskey(e, "MSI_ENDPOINT")
        (haskey(e, "MSI_SECRET") ? "azure-ml" : "cloud-shell")
    else
        "imds"
    end
end
function azure_managed(ctx)
    flavor=azure_msi_flavor(ctx)
    e=ctx.env
    resource=replace(azure_scope(ctx), r"/\.default$"=>"")
    headers=Dict{String,String}()
    if flavor=="service-fabric"
        throw(
            NotConfiguredError(
                "Service Fabric thumbprint pinning is not implemented; supply an explicit bearer-token provider",
            ),
        )
    elseif flavor=="azure-arc"
        url=with_query(
            e["IDENTITY_ENDPOINT"], obj("api-version"=>"2019-11-01", "resource"=>resource)
        )
        status, h, _=cloud_http(ctx, "GET", url, Dict("Metadata"=>"true"); timeout=5)
        challenge=get(h, "www-authenticate", "")
        m=match(r"realm=\"?([^\"]+)", challenge)
        status==401 && m!==nothing || throw(AuthError("Azure Arc returned no expected challenge"))
        path=String(strip(m[1]))
        directory=if Sys.iswindows()
            joinpath(
                get(e, "PROGRAMDATA", "C:/ProgramData"), "AzureConnectedMachineAgent", "Tokens"
            )
        else
            "/var/opt/azcmagent/tokens"
        end
        normpath(dirname(path))==normpath(directory) && endswith(path, ".key") ||
            throw(AuthError("Azure Arc challenge file is outside its allowed directory"))
        if ctx.files===nothing
            isfile(path) && dirname(realpath(path))==realpath(directory) ||
                throw(AuthError("Azure Arc challenge file resolves outside its allowed directory"))
        end
        secret=cloud_read(ctx, path)
        secret!==nothing && ncodeunits(secret)<=4096 ||
            throw(AuthError("Azure Arc challenge file is unavailable"))
        status, _, raw=cloud_http(
            ctx, "GET", url, Dict("Metadata"=>"true", "Authorization"=>"Basic "*strip(secret))
        )
        status==200 || throw(AuthError("Azure Arc rejected credential request"))
        return oauth_credential(JSON.parse(String(raw)), ctx)
    elseif flavor=="cloud-shell"
        status, _, raw=cloud_http(
            ctx,
            "POST",
            e["MSI_ENDPOINT"],
            Dict("Metadata"=>"true", "content-type"=>"application/x-www-form-urlencoded"),
            form_encode(obj("resource"=>resource)),
        )
        status==200 || throw(AuthError("Cloud Shell credential request rejected"))
        return oauth_credential(JSON.parse(String(raw)), ctx)
    end
    version=if flavor=="app-service"
        "2019-08-01"
    elseif flavor=="azure-ml"
        "2017-09-01"
    else
        "2018-02-01"
    end
    query=obj("api-version"=>version, "resource"=>resource)
    haskey(e, "AZURE_CLIENT_ID") &&
        (query[flavor=="azure-ml" ? "clientid" : "client_id"]=e["AZURE_CLIENT_ID"])
    base=if flavor=="app-service"
        e["IDENTITY_ENDPOINT"]
    elseif flavor=="azure-ml"
        e["MSI_ENDPOINT"]
    else
        "http://169.254.169.254/metadata/identity/oauth2/token"
    end
    headers=if flavor=="app-service"
        Dict("X-IDENTITY-HEADER"=>e["IDENTITY_HEADER"])
    elseif flavor=="azure-ml"
        Dict("secret"=>e["MSI_SECRET"])
    else
        Dict("Metadata"=>"true")
    end
    response=try
        cloud_http(ctx, "GET", with_query(base, query), headers; timeout=flavor=="imds" ? 1 : 30)
    catch error
        flavor=="imds" && error isa AuthError && return nothing
        rethrow()
    end
    status, _, raw=response
    status==200 ||
        (flavor=="imds" ? (return nothing) : throw(AuthError("managed identity request rejected")))
    return oauth_credential(JSON.parse(String(raw)), ctx)
end
function azure_cli(ctx, name)
    if name=="az"
        args=["az", "account", "get-access-token", "--output", "json", "--scope", azure_scope(ctx)]
        haskey(ctx.env, "AZURE_TENANT_ID") &&
            append!(args, ["--tenant", ctx.env["AZURE_TENANT_ID"]])
        d=JSON.parse(cloud_command(ctx, args))
        value=string_field(d, "accessToken")
        value===nothing && throw(AuthError("Azure CLI returned no token"))
        return oauth_credential(
            obj("access_token"=>value, "expires_on"=>get(d, "expires_on", nothing)), ctx
        )
    elseif name=="azd"
        d=JSON.parse(
            cloud_command(
                ctx, ["azd", "auth", "token", "--output", "json", "--scope", azure_scope(ctx)]
            ),
        )
        value=string_field(d, "token")
        value===nothing && throw(AuthError("Azure Developer CLI returned no token"))
        return BearerToken(value; expires_at=normalized_time(get(d, "expiresOn", nothing)))
    end
    resource=replace(replace(azure_scope(ctx), r"/\.default$"=>""), "'"=>"''")
    script="Get-AzAccessToken -ResourceUrl '$resource' -AsSecureString:\$false | ConvertTo-Json -Compress"
    d=JSON.parse(cloud_command(ctx, ["pwsh", "-NoProfile", "-NonInteractive", "-Command", script]))
    value=string_field(d, "Token")
    value===nothing && throw(AuthError("Azure PowerShell returned no token"))
    return BearerToken(value)
end

function gcp_assertion(ctx, info; scope=GCP_SCOPE)
    now=floor(Int, ctx.clock())
    url=get(info, "token_uri", "https://oauth2.googleapis.com/token")
    header=obj("alg"=>"RS256", "typ"=>"JWT")
    haskey(info, "private_key_id") && (header["kid"]=info["private_key_id"])
    payload=obj(
        "iat"=>now, "exp"=>now+3600, "iss"=>info["client_email"], "aud"=>url, "scope"=>scope
    )
    return url, jwt_encode(header, payload, info["private_key"])
end
function gcp_impersonate(ctx, source, url, delegates=[])
    data=cloud_exchange(
        ctx,
        url,
        obj("delegates"=>delegates, "scope"=>[GCP_SCOPE], "lifetime"=>"3600s");
        headers=Dict("authorization"=>"Bearer "*source.value),
    )
    token=string_field(data, "accessToken")
    token===nothing && throw(AuthError("impersonation response has no access token"))
    return BearerToken(token; expires_at=normalized_time(get(data, "expireTime", nothing)))
end
function gcp_info(ctx, info; depth=0)
    depth<16 || throw(NotConfiguredError("nested GCP credential sources are too deep"))
    type=get(info, "type", nothing)
    if type=="authorized_user"
        all(k->string_field(info, k)!==nothing, ("refresh_token", "client_id", "client_secret")) ||
            throw(NotConfiguredError("authorized_user credential file lacks required fields"))
        fields=obj(
            "grant_type"=>"refresh_token",
            "client_id"=>info["client_id"],
            "client_secret"=>info["client_secret"],
            "refresh_token"=>info["refresh_token"],
        )
        return oauth_credential(
            cloud_exchange(
                ctx,
                get(info, "token_uri", "https://oauth2.googleapis.com/token"),
                fields;
                form=true,
            ),
            ctx,
        )
    elseif type=="service_account"
        url, assertion=gcp_assertion(ctx, info)
        return oauth_credential(
            cloud_exchange(
                ctx, url, obj("grant_type"=>JWT_BEARER, "assertion"=>assertion); form=true
            ),
            ctx,
        )
    elseif type=="impersonated_service_account"
        source=get(info, "source_credentials", nothing)
        source isa AbstractDict ||
            throw(NotConfiguredError("impersonation needs source_credentials"))
        return gcp_impersonate(
            ctx,
            gcp_info(ctx, source; depth=depth+1),
            info["service_account_impersonation_url"],
            get(info, "delegates", []),
        )
    elseif type!="external_account"
        throw(
            NotConfiguredError(
                "unsupported GCP credential file type; supply a bearer-token provider"
            ),
        )
    end
    source=getobject(info, "credential_source")
    format=getobject(source, "format")
    subject=nothing
    haskey(source, "environment_id") && throw(
        NotConfiguredError(
            "GCP external account with AWS source is not implemented; supply an explicit bearer-token provider",
        ),
    )
    if haskey(source, "file")
        subject=cloud_read(ctx, source["file"])
    elseif haskey(source, "url")
        status, _, raw=cloud_http(
            ctx, "GET", source["url"], Dict{String,String}(get(source, "headers", Dict()))
        )
        status==200 || throw(AuthError("external account token URL was rejected"))
        subject=String(raw)
    elseif haskey(source, "executable")
        get(ctx.env, "GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES", "")=="1" || throw(
            NotConfiguredError(
                "set GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES=1 to authorize credential commands",
            ),
        )
        executable=source["executable"]
        argv=String.(Base.shell_split(executable["command"]))
        d=JSON.parse(
            cloud_command(ctx, argv; timeout=get(executable, "timeout_millis", 30000)/1000)
        )
        get(d, "success", true)===true ||
            throw(AuthError("external account command reported failure"))
        subject=first_nonempty(string_field(d, "id_token"), string_field(d, "saml_response"))
        format=obj()
    end
    subject===nothing && throw(NotConfiguredError("external account subject token is unavailable"))
    subject=String(strip(subject))
    if get(format, "type", nothing)=="json"
        subject=string_field(JSON.parse(subject), get(format, "subject_token_field_name", ""))
        subject===nothing && throw(AuthError("subject token field is missing"))
    end
    payload=obj(
        "grantType"=>"urn:ietf:params:oauth:grant-type:token-exchange",
        "audience"=>info["audience"],
        "scope"=>GCP_SCOPE,
        "requestedTokenType"=>"urn:ietf:params:oauth:token-type:access_token",
        "subjectToken"=>subject,
        "subjectTokenType"=>info["subject_token_type"],
    )
    token=oauth_credential(
        cloud_exchange(ctx, get(info, "token_url", "https://sts.googleapis.com/v1/token"), payload),
        ctx,
    )
    return if haskey(info, "service_account_impersonation_url")
        gcp_impersonate(ctx, token, info["service_account_impersonation_url"])
    else
        token
    end
end
function adc_path(ctx)
    return rstrip(get(ctx.env, "CLOUDSDK_CONFIG", "~/.config/gcloud"), '/')*"/application_default_credentials.json"
end
function gcp_metadata(ctx)
    lowercase(get(ctx.env, "NO_GCE_CHECK", "")) in ("1", "true") && return nothing
    host=get(
        ctx.env, "GCE_METADATA_HOST", get(ctx.env, "GCE_METADATA_ROOT", "metadata.google.internal")
    )
    response=try
        cloud_http(
            ctx,
            "GET",
            "http://$host/computeMetadata/v1/instance/service-accounts/default/token",
            Dict("Metadata-Flavor"=>"Google");
            timeout=1,
        )
    catch e
        e isa AuthError || rethrow()
        return nothing
    end
    status, _, raw=response
    status==200 || return nothing
    return oauth_credential(JSON.parse(String(raw)), ctx)
end

# Each rung shares its availability decision between offline explanation and
# online resolution. A configured network rung is not claimed to be verified.
struct CredentialRung
    name::String
    mechanism::String
    availability::Symbol
    acquire::Function
    label::String
    detail::String
end
struct ResolvedCredential <: Exception
    value::CredentialValue
    rung::CredentialRung
end
# The human label of each rung (AUTH-1 provenance: what the doctor and an auth
# error print; never a value).
const RUNG_LABELS=Dict(
    "env:AWS_ACCESS_KEY_ID"=>"env \$AWS_ACCESS_KEY_ID (+SECRET, +SESSION_TOKEN)",
    "assume-role"=>"profile assume-role via STS",
    "web-identity"=>"web identity via STS",
    "sso"=>"IAM Identity Center (~/.aws/sso/cache)",
    "shared-credentials-file"=>"~/.aws/credentials",
    "login"=>"aws login session (~/.aws/login/cache)",
    "credential_process"=>"profile credential_process",
    "config-file"=>"~/.aws/config static keys",
    "container"=>"container credentials endpoint",
    "imds"=>"EC2 instance metadata (IMDSv2)",
    "environment"=>"Entra service principal from AZURE_* env",
    "workload-identity"=>"Entra workload identity",
    "managed-identity"=>"Azure managed identity",
    "az"=>"az account get-access-token",
    "pwsh"=>"Azure PowerShell Get-AzAccessToken",
    "azd"=>"azd auth token",
    "adc-env"=>"GOOGLE_APPLICATION_CREDENTIALS file",
    "adc-file"=>"gcloud application default credentials file",
    "metadata"=>"GCE metadata server",
    "gcloud"=>"gcloud auth print-access-token",
)
rung_label(name) = get(RUNG_LABELS, name, startswith(name, "env:") ? "env \$"*name[5:end] : name)
# AUTH-1 named credentials (2026-09-19): the same four words on every cloud, each
# covering these rungs only; a named credential never falls through.
const NAMED_CREDENTIALS=("platform", "workload", "environment", "cli")
const NAMED_RUNGS=Dict(
    "aws-chain"=>Dict(
        "platform"=>("container", "imds"),
        "workload"=>("web-identity",),
        "environment"=>("env:AWS_ACCESS_KEY_ID",),
        "cli"=>("assume-role", "sso", "shared-credentials-file", "login", "credential_process", "config-file"),
    ),
    "azure-chain"=>Dict(
        "platform"=>("managed-identity",),
        "workload"=>("workload-identity",),
        "environment"=>("environment",),
        "cli"=>("az", "pwsh", "azd"),
    ),
    "gcp-chain"=>Dict(
        "platform"=>("metadata",),
        "workload"=>("adc-env",),
        "environment"=>("adc-env",),
        "cli"=>("adc-file", "gcloud"),
    ),
)
const GCP_NAMED_TYPES=Dict("workload"=>("external_account",), "environment"=>("service_account", "impersonated_service_account"))
const NAMED_MEANING=Dict(
    "aws-chain"=>Dict(
        "platform"=>"the ECS/EKS container endpoint, else the EC2 instance role (IMDSv2)",
        "workload"=>"web identity (AWS_WEB_IDENTITY_TOKEN_FILE + AWS_ROLE_ARN) via STS",
        "environment"=>"AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY",
        "cli"=>"the active `aws` profile (assume-role, SSO, shared files, `aws login`, credential_process)",
    ),
    "azure-chain"=>Dict(
        "platform"=>"Azure managed identity",
        "workload"=>"Entra workload identity (AZURE_FEDERATED_TOKEN_FILE)",
        "environment"=>"an Entra service principal from AZURE_TENANT_ID / AZURE_CLIENT_ID + secret or certificate",
        "cli"=>"`az`, Azure PowerShell or `azd` sign-in",
    ),
    "gcp-chain"=>Dict(
        "platform"=>"the attached service account (GCE metadata server)",
        "workload"=>"workload identity federation (GOOGLE_APPLICATION_CREDENTIALS, type external_account)",
        "environment"=>"a service-account file (GOOGLE_APPLICATION_CREDENTIALS, type service_account)",
        "cli"=>"`gcloud auth application-default login` (the ADC file) or `gcloud auth print-access-token`",
    ),
)
named_meaning(policy, name) = NAMED_MEANING[policy.credential_policy][name]
function check_named(policy, name)
    name===nothing && return nothing
    endswith(policy.credential_policy, "-chain") || throw(NotConfiguredError(
        "$(policy.provider): a named credential ($(repr(name))) names a cloud identity; this door has no cloud chain";
        provider=policy.provider))
    name in NAMED_CREDENTIALS || throw(NotConfiguredError(
        "$(policy.provider): unknown named credential $(repr(name)); one of $(join(repr.(NAMED_CREDENTIALS), ", "))";
        provider=policy.provider))
    return name
end
"""
    CredentialSource

Where a resolved cloud credential came from (AUTH-1 provenance): the rung, its label,
the name that selected it when one did, and the expiry if known. Never the value.
"""
struct CredentialSource
    rung::String
    label::String
    named::Maybe{String}
    expires_at::Maybe{String}
end
function describe(s::CredentialSource; now=time())
    text=s.label
    s.named===nothing || (text*=" (named credential \"$(s.named)\")")
    if s.expires_at!==nothing
        left=floor(Int, expiry_seconds(s.expires_at)-now)
        text*=if left<=0
            ", expired"
        elseif left<3600
            ", expires in $(max(left ÷ 60, 1)) min"
        else
            ", expires in $(left ÷ 3600) h $((left % 3600) ÷ 60) min"
        end
    end
    return text
end
Base.show(io::IO, s::CredentialSource) = print(io, "CredentialSource(", describe(s), ")")
Base.showerror(io::IO, ::ResolvedCredential) = print(io, "credential resolved (<redacted>)")
function cloud_rungs(policy, ctx; acquire=false, named=nothing)
    rungs=CredentialRung[]
    e=ctx.env
    developer_failed=false
    wanted=named===nothing ? nothing : NAMED_RUNGS[policy.credential_policy][named]
    function add(name, mechanism, state, fn, detail="")
        wanted===nothing || name in wanted || return nothing
        rung=CredentialRung(name, mechanism, state, fn, rung_label(name), detail)
        push!(rungs, rung)
        if acquire && state!==:absent
            value=try
                fn()
            catch error
                if policy.credential_policy=="azure-chain" &&
                    name in ("az", "pwsh", "azd") &&
                    error isa AuthError
                    developer_failed=true
                    return nothing
                end
                error isa Union{LM15Error,InterruptException} && rethrow()
                throw(AuthError("cloud credential source failed"; provider=policy.provider))
            end
            value===nothing || throw(ResolvedCredential(value, rung))
        end
        return nothing
    end
    for key in policy.env_keys
        add(
            "env:$key",
            "env",
            isempty(get(e, key, "")) ? :absent : :usable,
            ()->key=="AWS_BEARER_TOKEN_BEDROCK" ? BearerToken(e[key]) : ApiKey(e[key]),
        )
    end
    if policy.credential_policy=="aws-chain"
        add(
            "env:AWS_ACCESS_KEY_ID",
            "env",
            aws_environment(ctx)===nothing ? :absent : :usable,
            ()->aws_environment(ctx),
        )
        credentials, config, profile=aws_profiles(ctx)
        section=profile_section(config, profile)
        role=haskey(section, "role_arn") &&
             (haskey(section, "source_profile") || haskey(section, "credential_source"))
        add("assume-role", "sigv4-sts", role ? :configured : :absent, ()->assume_role(ctx, section))
        web=web_identity_config(ctx, section)
        add(
            "web-identity",
            "unsigned-sts",
            if web===nothing || (role && !haskey(e, "AWS_WEB_IDENTITY_TOKEN_FILE"))
                :absent
            else
                :configured
            end,
            ()->aws_web_identity(ctx, web),
        )
        sso=sso_config(config, section)
        add("sso", "file-cache", sso===nothing ? :absent : :configured, ()->aws_sso(ctx, sso))
        shared=get(credentials, profile, Dict{String,String}())
        add(
            "shared-credentials-file",
            "ini-profile",
            aws_static(shared)===nothing ? :absent : :usable,
            ()->aws_static(shared),
        )
        login=aws_login_cached(ctx, section)
        add(
            "login",
            "file-cache",
            if !haskey(section, "login_session")
                :absent
            elseif login!==nothing && !is_expired(login; now=ctx.clock())
                :usable
            else
                :configured
            end,
            ()->begin
                login!==nothing && !is_expired(login; now=ctx.clock()) || throw(
                    NotConfiguredError(
                        "AWS login expired; run aws login (DPoP refresh is not implemented)"
                    ),
                )
                login
            end,
        )
        command=get(section, "credential_process", nothing)
        argv=command===nothing ? String[] : String.(Base.shell_split(command))
        add(
            "credential_process",
            "subprocess",
            isempty(argv) || cloud_on_path(ctx, first(argv))===nothing ? :absent : :configured,
            ()->begin
                d=JSON.parse(cloud_command(ctx, argv; timeout=60))
                get(d, "Version", nothing)==1 ||
                    throw(AuthError("credential_process needs Version 1"))
                aws_credential(d)
            end,
        )
        add(
            "config-file",
            "ini-profile",
            aws_static(section)===nothing ? :absent : :usable,
            ()->aws_static(section),
        )
        add(
            "container",
            "http-metadata",
            container_url(ctx)===nothing ? :absent : :configured,
            ()->aws_container(ctx),
        )
        add(
            "imds",
            "http-metadata",
            lowercase(get(e, "AWS_EC2_METADATA_DISABLED", ""))=="true" ? :absent : :configured,
            ()->aws_imds(ctx),
        )
    elseif policy.credential_policy=="azure-chain"
        narrowing=lowercase(strip(get(e, "AZURE_TOKEN_CREDENTIALS", "")))
        permitted(name, dev) = isempty(narrowing) || (
            if narrowing=="prod"
                !dev
            elseif narrowing=="dev"
                dev
            else
                narrowing==lowercase(name)
            end
        )
        envready=all(k->haskey(e, k), ("AZURE_TENANT_ID", "AZURE_CLIENT_ID")) &&
                 (haskey(e, "AZURE_CLIENT_SECRET") || haskey(e, "AZURE_CLIENT_CERTIFICATE_PATH"))
        add(
            "environment",
            "http-token-exchange",
            envready && permitted("EnvironmentCredential", false) ? :configured : :absent,
            ()->begin
                url, fields=azure_environment_fields(ctx)
                oauth_credential(cloud_exchange(ctx, url, fields; form=true), ctx)
            end,
        )
        ready=all(
            k->haskey(e, k), ("AZURE_TENANT_ID", "AZURE_CLIENT_ID", "AZURE_FEDERATED_TOKEN_FILE")
        )
        add(
            "workload-identity",
            "http-token-exchange",
            ready && permitted("WorkloadIdentityCredential", false) ? :configured : :absent,
            ()->azure_workload(ctx),
        )
        add(
            "managed-identity",
            "http-metadata",
            permitted("ManagedIdentityCredential", false) ? :configured : :absent,
            ()->azure_managed(ctx),
        )
        for (command, name) in (
            ("az", "AzureCliCredential"),
            ("pwsh", "AzurePowerShellCredential"),
            ("azd", "AzureDeveloperCliCredential"),
        )
            add(
                command,
                "subprocess",
                if permitted(name, true) && cloud_on_path(ctx, command)!==nothing
                    :configured
                else
                    :absent
                end,
                ()->azure_cli(ctx, command),
            )
        end
    elseif policy.credential_policy=="gcp-chain"
        for (name, path) in (
            ("adc-env", get(e, "GOOGLE_APPLICATION_CREDENTIALS", nothing)),
            ("adc-file", adc_path(ctx)),
        )
            info=path===nothing ? nothing : cloud_json(ctx, path)
            if name=="adc-env" && info!==nothing && named!==nothing && haskey(GCP_NAMED_TYPES, named)
                type=string(get(info, "type", ""))
                if !(type in GCP_NAMED_TYPES[named])
                    other=named=="workload" ? "environment" : "workload"
                    why="$path holds $type credentials; that is the named credential \"$other\", not \"$named\""
                    add(name, "json-file", :absent, ()->throw(NotConfiguredError(why; provider=policy.provider,
                        credential_hint="credentials=Dict(\"$(policy.provider)\" => \"$other\")")), why)
                    continue
                end
            end
            add(name, "json-file", info===nothing ? :absent : :configured, ()->gcp_info(ctx, info))
        end
        add(
            "metadata",
            "http-metadata",
            lowercase(get(e, "NO_GCE_CHECK", "")) in ("1", "true") ? :absent : :configured,
            ()->gcp_metadata(ctx),
        )
        add(
            "gcloud",
            "subprocess",
            cloud_on_path(ctx, "gcloud")===nothing ? :absent : :configured,
            ()->BearerToken(strip(cloud_command(ctx, ["gcloud", "auth", "print-access-token"]))),
        )
    end
    developer_failed && throw(
        AuthError(
            "Azure developer credentials failed; sign in with az, pwsh, or azd";
            provider=policy.provider,
        ),
    )
    return rungs
end
const NOTHING_FOUND_HINTS=Dict(
    "gcp-chain"=>"on a laptop: `gcloud auth application-default login`; elsewhere: set GOOGLE_APPLICATION_CREDENTIALS to a service-account or workload-identity file, run on Google Cloud with an attached service account, or pass api_keys=Dict(\"<provider>\" => <token or callable>)",
    "azure-chain"=>"on a laptop: `az login`; elsewhere: a managed identity, AZURE_TENANT_ID + AZURE_CLIENT_ID with a secret or certificate, or api_keys=Dict(\"<provider>\" => <token provider>)",
    "aws-chain"=>"on a laptop: `aws sso login` or `aws configure`; elsewhere: the instance or container role, AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY, or api_keys=Dict(\"<provider>\" => <credentials callable>)",
)
function probed_summary(rungs)
    return join(("$(r.label): $(r.availability===:absent ? (isempty(r.detail) ? "absent" : r.detail) : "tried")" for r in rungs), "; ")
end
"""Walk the chain online; the first rung that yields wins and is named (AUTH-1
provenance). With `named`, only that name's rungs run and nothing else is tried."""
function resolve_cloud_source(policy, ctx; named=nothing)
    rungs=CredentialRung[]
    try
        rungs=cloud_rungs(policy, ctx; acquire=true, named)
    catch error
        if error isa ResolvedCredential
            value=error.value
            expires=value isa Union{BearerToken,AwsCredentials} ? value.expires_at : nothing
            return value, CredentialSource(error.rung.name, error.rung.label, named, expires)
        end
        rethrow()
    end
    if named!==nothing
        throw(NotConfiguredError(
            "$(policy.provider): named credential \"$named\" — $(named_meaning(policy, named)) — answered nothing ($(probed_summary(rungs))). This door was told to use that identity only; it will not try the rest of the $(policy.credential_policy) chain.";
            provider=policy.provider,
            credential_hint="credentials=Dict(\"$(policy.provider)\" => \"<platform|workload|environment|cli>\") names one identity; omit it to walk the $(policy.credential_policy) chain",
        ))
    end
    hint=get(NOTHING_FOUND_HINTS, policy.credential_policy, nothing)
    hint===nothing || (hint=replace(hint, "<provider>"=>policy.provider))
    hint!==nothing && !isempty(policy.env_keys) && (hint="set $(first(policy.env_keys)), or $hint")
    return throw(NotConfiguredError(
        "$(policy.provider): no credential found in the $(policy.credential_policy) chain ($(probed_summary(rungs)))";
        provider=policy.provider, credential_hint=hint,
    ))
end
resolve_cloud(policy, ctx; named=nothing) = first(resolve_cloud_source(policy, ctx; named))
"""
A cloud chain credential provider (AUTH-2/AUTH-3): resolves once, hands out the value
until the refresh skew, then re-resolves. `source` names the rung that last answered.
"""
mutable struct CloudCredentialProvider
    policy::AccessPolicy
    ctx::ChainContext
    named::Maybe{String}
    lock::ReentrantLock
    cached::Union{Nothing,CredentialValue}
    source::Union{Nothing,CredentialSource}
end
Base.show(io::IO, p::CloudCredentialProvider) = print(io, "<cloud credential provider for ", p.policy.provider, ">")
function (p::CloudCredentialProvider)()
    return lock(p.lock) do
        current=p.cached
        if current===nothing || is_expired(current; now=p.ctx.clock())
            current, source=resolve_cloud_source(p.policy, p.ctx; named=p.named)
            p.source=source
            is_expired(current; now=p.ctx.clock()) && throw(AuthError(
                "cloud credential is expired; renew the configured credential source (credential came from: $(describe(source; now=p.ctx.clock())))";
                provider=p.policy.provider))
            p.cached=current isa BearerToken && current.expires_at===nothing ? nothing : current
        end
        return current
    end
end
function cloud_credential_provider(policy, ctx; named=nothing)
    check_named(policy, named)
    # A provider instance owns one identity. Ambient environment edits must not
    # change the principal underneath an already-cached access token.
    ctx=ChainContext(;
        env=copy(ctx.env),
        home=ctx.home,
        files=ctx.files===nothing ? nothing : copy(ctx.files),
        settings=copy(ctx.settings),
        clock=ctx.clock,
        http=ctx.http,
        command=ctx.command,
    )
    return CloudCredentialProvider(policy, ctx, named, ReentrantLock(), nothing, nothing)
end
const GCLOUD_CONFIG_NAME=r"^[a-z][-a-z0-9]*$"  # gcloud's own rule; keeps the name inside the directory
gcloud_config_dir(ctx) = rstrip(get(ctx.env, "CLOUDSDK_CONFIG", "~/.config/gcloud"), '/')
"""The project `gcloud config get project` prints, read from the files gcloud reads
(AUTH-10, amended 2026-09-26): CLOUDSDK_CORE_PROJECT, then `[core] project` in the active
configuration (CLOUDSDK_ACTIVE_CONFIG_NAME, else `active_config`, else `default`)."""
function gcloud_config_project(ctx)
    value=strip(get(ctx.env, "CLOUDSDK_CORE_PROJECT", ""))
    isempty(value) || return String(value), "env:CLOUDSDK_CORE_PROJECT"
    base=gcloud_config_dir(ctx)
    name=strip(get(ctx.env, "CLOUDSDK_ACTIVE_CONFIG_NAME", ""))
    isempty(name) && (name=strip(something(cloud_read(ctx, "$base/active_config"), "")))
    isempty(name) && (name="default")
    occursin(GCLOUD_CONFIG_NAME, name) || return nothing
    raw=cloud_read(ctx, "$base/configurations/config_$name")
    raw===nothing && return nothing
    config=try
        parse_ini(raw)
    catch
        return nothing
    end
    project=strip(get(get(config, "core", Dict{String,String}()), "project", ""))
    return isempty(project) ? nothing : (String(project), "gcloud-config")
end
"""`project/project-id` from the metadata server; offline (the doctor), unprobed."""
function gcp_metadata_project(ctx; offline=false)
    lowercase(get(ctx.env, "NO_GCE_CHECK", "")) in ("1", "true") && return nothing
    offline && return (nothing, "metadata")
    host=get(ctx.env, "GCE_METADATA_HOST", get(ctx.env, "GCE_METADATA_ROOT", "metadata.google.internal"))
    response=try
        cloud_http(ctx, "GET", "http://$host/computeMetadata/v1/project/project-id", Dict("Metadata-Flavor"=>"Google"); timeout=1)
    catch e
        e isa InterruptException && rethrow()
        return nothing
    end
    status, _, raw=response
    value=status==200 ? strip(String(copy(raw))) : ""
    return !isempty(value) && !occursin(r"[\s/?#]", value) ? (String(value), "metadata") : nothing
end
"""
    profile_settings(policy, ctx; offline=false)

The setting values the cloud's own configuration carries, after the caller and the
setting's env variables (AUTH-10), as `(value, from)`: the AWS profile's `region`; the
Google project from the credential file, gcloud's configuration, the ADC file, the
metadata server. `(nothing, "metadata")` means only the metadata server could answer.
"""
function profile_settings(policy, ctx; offline=false)
    return function (name)
        if policy.credential_policy=="aws-chain" && name=="region"
            creds, config, profile=aws_profiles(ctx)
            value=get(profile_section(config, profile), "region", get(get(creds, profile, Dict()), "region", nothing))
            return value===nothing || isempty(value) ? nothing : (value, "aws-profile")
        elseif policy.credential_policy=="gcp-chain" && name=="project"
            path=get(ctx.env, "GOOGLE_APPLICATION_CREDENTIALS", "")
            if !isempty(path)
                info=something(try cloud_json(ctx, path) catch; nothing end, obj())
                value=first_nonempty(string_field(info, "project_id"), string_field(info, "quota_project_id"))
                value===nothing || return (value, "adc-env")
            end
            found=gcloud_config_project(ctx)
            found===nothing || return found
            info=something(try cloud_json(ctx, adc_path(ctx)) catch; nothing end, obj())
            value=first_nonempty(string_field(info, "quota_project_id"), string_field(info, "project_id"))
            value===nothing || return (value, "adc-file")
            return gcp_metadata_project(ctx; offline)
        end
        return nothing
    end
end
function cloud_profile_settings(policy, given, env; files=nothing, home=get(env, "HOME", homedir()))
    # Kept for callers that need values only: the explicit ones, then the profile.
    out=Dict{String,String}(given)
    lookup=profile_settings(policy, ChainContext(; env, files, home); offline=true)
    for setting in (policy.host===nothing ? () : policy.host.settings)
        haskey(out, setting.name) && continue
        any(k->!isempty(get(env, k, "")), setting.env) && continue
        found=lookup(setting.name)
        found===nothing || found[1]===nothing || (out[setting.name]=found[1])
    end
    return out
end
"""
    explain_cloud(provider; env, api_keys, settings, files, home, credential, base_url)

The offline AUTH-7 walk of a cloud door: rung 0 (`api_keys`) then the chain, or only the
rungs a named `credential` covers; each host setting with where it came from; and the base
URL the door will send to with its origin (the explicit entry, the vendor's variable by
name, or the template).
"""
function explain_cloud(
    provider;
    env=ENV,
    api_keys=Dict(),
    settings=Dict(),
    files=nothing,
    home=get(env, "HOME", homedir()),
    credential=nothing,
    base_url=nothing,
)
    p=provider_definition(provider)
    source=explicit_source(p.id, api_keys)
    named=check_named(p.access, credential)
    named!==nothing && source!==nothing && throw(NotConfiguredError(
        "$(p.id): both an api_keys entry and the named credential \"$named\"; give one"; provider=p.id))
    selected=source!==nothing
    ctx=ChainContext(; env, files, home)
    endpoint, endpoint_from=if base_url!==nothing
        base_url, "explicit"
    else
        var_value, var=endpoint_from_env(p.access.host, env)
        var_value, var===nothing ? nothing : "env:$var"
    end
    origins=Dict{String,String}()
    problems=NotConfiguredError[]
    shown=resolve_settings(
        p.access, settings; env, profile=profile_settings(p.access, ctx; offline=true),
        endpoint, sources=origins, unprobed_ok=true, problems,
    )
    report_settings=OrderedDict{String,Any}()
    for setting in p.access.host.settings
        origin=get(origins, setting.name, nothing)
        origin===nothing && continue  # an optional setting an endpoint made unnecessary
        report_settings[setting.name]=if origin=="missing"
            (value=nothing, from=nothing, state=nothing)
        elseif startswith(origin, "unprobed:")
            (value=nothing, from=origin[10:end], state="unprobed")
        else
            (value=shown[setting.name], from=origin, state=nothing)
        end
    end
    url=nothing
    if isempty(problems)
        url=try
            render_base_url(p.access.host, shown, endpoint; provider=p.id)
        catch e
            e isa NotConfiguredError || rethrow()
            nothing
        end
    end
    ctx=ChainContext(; env, files, home, settings=shown)
    steps=[
        AuthStep(
            "api_keys",
            selected ? "provided via $source (value never shown)" : "not provided",
            selected ? :selected : :absent,
        ),
    ]
    for rung in cloud_rungs(p.access, ctx; named)
        state=if rung.availability===:absent
            :absent
        elseif selected
            :shadowed
        elseif rung.availability===:usable
            :selected
        else
            :unprobed
        end
        push!(
            steps,
            AuthStep(
                rung.name,
                if state===:unprobed
                    "$(rung.label): configured; $(rung.mechanism) runs at request time"
                elseif state===:absent && !isempty(rung.detail)
                    "$(rung.label): $(rung.detail)"
                else
                    "$(rung.label): $(rung.mechanism); values never shown"
                end,
                state,
            ),
        )
        selected |= state===:selected
    end
    return AuthReport(
        p.id, steps, selected || any(s->s.state===:unprobed, steps),
        Dict{String,String}(k=>v.value for (k, v) in report_settings if v.value!==nothing);
        settings_from=report_settings,
        base_url=url, base_url_from=url===nothing ? nothing : something(endpoint_from, "template"),
        named, problems=[e.message for e in problems],
    )
end
