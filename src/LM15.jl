"""
LM15 — Julia port of lm15, rebuilt module-by-module against the
lm15-contract corpus after the stale v1 implementation was removed
(2026-08-31).

Currently implemented: the auth surface (spec/auth.md AUTH-1/2/5/7 and the
AUTH-8 read side), fixture-verified against
`conformance/auth_resolution.json`.
"""
module LM15

include("json.jl")
include("auth.jl")

export AuthReport,
    AuthStep,
    Credential,
    LocalOAuthCredential,
    NotConfiguredError,
    StaticCredential,
    UnknownProviderError,
    access_token,
    account_id,
    canonical_provider,
    default_claude_credentials_path,
    default_codex_auth_path,
    describe,
    explain_auth,
    has_refresh_token,
    is_expired,
    known_providers,
    read_claude_code_credential,
    read_codex_cli_credential,
    resolve_credential,
    selected_step

end # module
