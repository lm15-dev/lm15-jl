# LM15.jl

Julia port of lm15, rebuilt module-by-module against the
[lm15-contract](https://github.com/lm15-dev/lm15-contract) corpus after the
stale v1 implementation was removed (2026-08-31). Stdlib-only
(`src/json.jl` is the port's own parser, recovered from the pre-removal
tree — a JSON parser is API-neutral).

## Status

| Module | Contract surface | State |
|---|---|---|
| `src/auth.jl` | spec/auth.md AUTH-1/2/5/7 + AUTH-8 read side | partial; current fixture suite fails (`conformance/auth_resolution.json`) |
| everything else | — | not yet rebuilt |

The auth module ships: the `Credential` union (static string or zero-arg
callable, plus `StaticCredential` with redacted `show`), the resolution
chain + `explain_auth` doctor report, and read-only borrowed-credential
loaders for the Claude Code and Codex CLI files. Not yet implemented
(stated, not absorbed): the AUTH-3/4 write side (locked double-checked
refresh, atomic 0600 writes) and the AUTH-9 login primitives.

The copied auth corpus now includes xAI subscription and cloud-chain cases.
The implementation and test driver do not support those cases yet. Keep the
failures visible; this port does not pass the current auth contract and has
no `CONTRACT_PIN` yet. Complete the auth upgrade before claiming conformance.

```julia
using Pkg; Pkg.test()
```
