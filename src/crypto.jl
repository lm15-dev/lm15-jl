# Native OpenSSL signing, not private-key arithmetic in Julia. Keys and signed
# assertions never appear in thrown errors or displayed objects.
function rsa_sign_sha256(pem::AbstractString, message::AbstractVector{UInt8})
    occursin("ENCRYPTED", pem) &&
        throw(NotConfiguredError("encrypted private keys require explicit decryption before use"))
    keybytes=Vector{UInt8}(codeunits(pem))
    bio=ccall(
        (:BIO_new_mem_buf, OpenSSL_jll.libcrypto),
        Ptr{Cvoid},
        (Ptr{UInt8}, Cint),
        keybytes,
        length(keybytes),
    )
    bio==C_NULL && throw(NotConfiguredError("cannot read RSA private key"))
    key=C_NULL
    ctx=C_NULL
    try
        key=ccall(
            (:PEM_read_bio_PrivateKey, OpenSSL_jll.libcrypto),
            Ptr{Cvoid},
            (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
            bio,
            C_NULL,
            C_NULL,
            C_NULL,
        )
        key==C_NULL && throw(
            NotConfiguredError(
                "no usable unencrypted PEM private key; convert PKCS#12 using openssl pkcs12 -nodes",
            ),
        )
        keytype=ccall((:EVP_PKEY_get_base_id, OpenSSL_jll.libcrypto), Cint, (Ptr{Cvoid},), key)
        keytype==6 || throw(NotConfiguredError("RS256 requires an RSA key"))
        ctx=ccall((:EVP_MD_CTX_new, OpenSSL_jll.libcrypto), Ptr{Cvoid}, ())
        ctx==C_NULL && throw(NotConfiguredError("cannot allocate RSA signing context"))
        digest=ccall((:EVP_sha256, OpenSSL_jll.libcrypto), Ptr{Cvoid}, ())
        ready=ccall(
            (:EVP_DigestSignInit, OpenSSL_jll.libcrypto),
            Cint,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
            ctx,
            C_NULL,
            digest,
            C_NULL,
            key,
        )
        ready==1 || throw(NotConfiguredError("cannot initialize RSA signer"))
        size=Ref{Csize_t}(0)
        rc=ccall(
            (:EVP_DigestSign, OpenSSL_jll.libcrypto),
            Cint,
            (Ptr{Cvoid}, Ptr{UInt8}, Ref{Csize_t}, Ptr{UInt8}, Csize_t),
            ctx,
            C_NULL,
            size,
            message,
            length(message),
        )
        rc==1 || throw(AuthError("RSA signing failed"))
        signature=Vector{UInt8}(undef, size[])
        rc=ccall(
            (:EVP_DigestSign, OpenSSL_jll.libcrypto),
            Cint,
            (Ptr{Cvoid}, Ptr{UInt8}, Ref{Csize_t}, Ptr{UInt8}, Csize_t),
            ctx,
            signature,
            size,
            message,
            length(message),
        )
        rc==1 || throw(AuthError("RSA signing failed"))
        resize!(signature, size[])
    finally
        ctx==C_NULL || ccall((:EVP_MD_CTX_free, OpenSSL_jll.libcrypto), Cvoid, (Ptr{Cvoid},), ctx)
        key==C_NULL || ccall((:EVP_PKEY_free, OpenSSL_jll.libcrypto), Cvoid, (Ptr{Cvoid},), key)
        ccall((:BIO_free, OpenSSL_jll.libcrypto), Cint, (Ptr{Cvoid},), bio)
        fill!(keybytes, 0)
    end
end
function jwt_encode(header, payload, pem)
    check_json(header)
    check_json(payload)
    signing=base64url(codeunits(JSON.serialize(header)))*"."*base64url(
        codeunits(JSON.serialize(payload))
    )
    return signing*"."*base64url(rsa_sign_sha256(pem, Vector{UInt8}(codeunits(signing))))
end
function certificate_der(pem)
    m=match(r"-----BEGIN CERTIFICATE-----([A-Za-z0-9+/=\s]+)-----END CERTIFICATE-----", pem)
    m===nothing && throw(NotConfiguredError("no PEM certificate block"))
    return base64decode(join(split(m[1])))
end
