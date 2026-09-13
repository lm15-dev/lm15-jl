struct PKCEPair
    verifier::String
    challenge::String
end
Base.show(io::IO, ::PKCEPair) = print(io, "PKCEPair(<verifier redacted>, method=S256)")
base64url(bytes) = rstrip(replace(base64encode(bytes), '+'=>'-', '/'=>'_'), '=')
pkce_challenge(verifier::AbstractString) = base64url(sha256(verifier))
function generate_pkce()
    verifier=base64url(rand(RandomDevice(), UInt8, 64))
    return PKCEPair(verifier, pkce_challenge(verifier))
end
struct DeviceCodeExpiredError <: AuthError
    metadata::ErrorMetadata
end
function DeviceCodeExpiredError(provider::AbstractString)
    return DeviceCodeExpiredError(
        ErrorMetadata(;
            provider=String(provider), message="device authorization expired; start login again"
        ),
    )
end
ERROR_CODES[DeviceCodeExpiredError] = "auth"
struct DeviceAuthorization
    user_code::String
    verification_uri::String
    verification_uri_complete::Maybe{String}
    interval::Float64
    expires_in::Float64
    device_code::String
end
function Base.show(io::IO, d::DeviceAuthorization)
    return print(
        io,
        "DeviceAuthorization(user_code=",
        repr(d.user_code),
        ", verification_uri=",
        repr(d.verification_uri),
        ", <device code redacted>)",
    )
end
const XAI_CLIENT_ID="b1a00492-073a-47ea-816f-4c329264a828"
const XAI_TOKEN_URL="https://auth.x.ai/oauth2/token"
function auth_form(url, fields)
    try
        r=http_request(
            "POST",
            url,
            ["Content-Type"=>"application/x-www-form-urlencoded", "Accept"=>"application/json"],
            form_encode(fields);
            status_exception=false,
            retry=false,
            redirect=false,
            readtimeout=30,
            connect_timeout=15,
        )
        d=JSON.parse(String(r.body))
        d isa AbstractDict || throw(ArgumentError("invalid envelope"))
        return Int(r.status), d
    catch e
        e isa InterruptException && rethrow()
        throw(AuthError("credential endpoint request failed"))
    end
end
function trusted_verification_uri(raw)
    raw isa AbstractString ||
        throw(AuthError("device authorization has no verification URI"; provider="xai"))
    uri=HTTP.URI(raw)
    uri.scheme=="https" && !isempty(uri.host) && isempty(uri.userinfo) ||
        throw(AuthError("device authorization returned an unsafe verification URI"; provider="xai"))
    return String(raw)
end
function start_xai_device_login()
    status, d=auth_form(
        "https://auth.x.ai/oauth2/device/code",
        obj(
            "client_id"=>XAI_CLIENT_ID,
            "scope"=>"openid profile email offline_access grok-cli:access api:access",
            "referrer"=>"lm15",
        ),
    )
    200<=status<300 || throw(AuthError("device authorization request was rejected"; provider="xai"))
    device=string_field(d, "device_code")
    user=string_field(d, "user_code")
    expires=get(d, "expires_in", nothing)
    device!==nothing &&
    user!==nothing &&
    expires isa Real &&
    !(expires isa Bool) &&
    isfinite(expires) &&
    expires>0 || throw(AuthError("device authorization is missing required fields"; provider="xai"))
    interval=get(d, "interval", 5)
    interval=if interval isa Real && !(interval isa Bool) && isfinite(interval) && interval>0
        Float64(interval)
    else
        5.0
    end
    full=string_field(d, "verification_uri_complete")
    return DeviceAuthorization(
        user,
        trusted_verification_uri(get(d, "verification_uri", nothing)),
        full===nothing ? nothing : trusted_verification_uri(full),
        interval,
        Float64(expires),
        device,
    )
end
function poll_xai_device_login(device::DeviceAuthorization; clock=()->time_ns()/1e9, sleep_fn=sleep)
    deadline=clock()+device.expires_in
    interval=device.interval
    while true
        clock()+interval>deadline && throw(DeviceCodeExpiredError("xai"))
        sleep_fn(interval)
        clock()>=deadline && throw(DeviceCodeExpiredError("xai"))
        status, d=auth_form(
            XAI_TOKEN_URL,
            obj(
                "grant_type"=>"urn:ietf:params:oauth:grant-type:device_code",
                "client_id"=>XAI_CLIENT_ID,
                "device_code"=>device.device_code,
            ),
        )
        if 200<=status<300
            access=string_field(d, "access_token")
            access===nothing &&
                throw(AuthError("device login returned no access token"; provider="xai"))
            lifetime=get(d, "expires_in", 3600)
            lifetime isa Real && !(lifetime isa Bool) && isfinite(lifetime) && lifetime>0 ||
                (lifetime=3600)
            return LocalOAuthCredential(
                access,
                string_field(d, "refresh_token"),
                round(Int64, (time()+lifetime)*1000),
                nothing,
            )
        end
        code=get(d, "error", nothing)
        code=="authorization_pending" && continue
        if code=="slow_down"
            hint=get(d, "interval", nothing)
            interval=if hint isa Real && !(hint isa Bool) && isfinite(hint) && hint>0
                Float64(hint)
            else
                interval+5
            end
        elseif code=="expired_token"
            throw(DeviceCodeExpiredError("xai"))
        elseif code in ("access_denied", "authorization_denied")
            throw(AuthError("device authorization was denied"; provider="xai"))
        else
            throw(AuthError("device authorization failed"; provider="xai"))
        end
    end
end
struct CredentialFileStore
    path::String
end
CredentialFileStore() = CredentialFileStore(default_credentials_path())
function Base.show(io::IO, s::CredentialFileStore)
    return print(io, "CredentialFileStore(path=", repr(s.path), ")")
end
function read_store(s::CredentialFileStore)
    isfile(s.path) || return obj()
    return read_json_object(
        s.path, "credential-store", "Restore a valid credential file or log in again"
    )
end
function mutate!(f::Function, s::CredentialFileStore, provider)
    hold_file_lock(s.path) do
        data=read_store(s)
        current=deepcopy(get(data, provider, nothing))
        replacement=f(current)
        replacement===nothing && return current
        replacement isa AbstractDict ||
            throw(ArgumentError("credential store mutation must return an object or nothing"))
        check_json(replacement)
        data[provider]=replacement
        write_private_json_atomic(s.path, data)
        return deepcopy(replacement)
    end
end
function delete_credential!(s::CredentialFileStore, provider)
    hold_file_lock(s.path) do
        data=read_store(s)
        if haskey(data, provider)
            delete!(data, provider)
            write_private_json_atomic(s.path, data)
        end
    end
    return nothing
end
function write_xai_credential(c::LocalOAuthCredential, path=default_credentials_path())
    mutate!(CredentialFileStore(path), "xai") do current
        entry=copy(asobject(current))
        entry["type"]="oauth"
        entry["access"]=c.access_token
        c.refresh_token===nothing || (entry["refresh"]=c.refresh_token)
        c.expires_at_ms===nothing || (entry["expires"]=c.expires_at_ms)
        return entry
    end
    return c
end
function login(provider::AbstractString; credentials_path=default_credentials_path(), echo=println)
    definition=provider_definition(provider)
    if definition.id!="xai"
        hint=first_nonempty(
            definition.access.login_hint, definition.console_url, "This is a keyless local server"
        )
        throw(
            UnsupportedFeatureError("login is owned by the provider: $hint"; provider=definition.id)
        )
    end
    device=start_xai_device_login()
    echo(
        "Open $(something(device.verification_uri_complete,device.verification_uri)) and enter code $(device.user_code)",
    )
    credential=poll_xai_device_login(device)
    return write_xai_credential(credential, credentials_path)
end
