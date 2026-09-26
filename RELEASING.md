# Releasing LM15.jl

## What 1.0.0 was checked with (2026-09-26)

```bash
julia --project=. -e 'using Pkg; Pkg.test()'                           # Julia 1.12.7 and 1.10.12
cd ../lm15-contract && python3 harness/check.py --shim julia --direction all   # 1,788 of 1,788, at the pin
python3 tools/managed_crossrun.py python julia typescript rust go      # 40 of 40
cd ../lm15-jl && bash docs/build.sh                                     # the manual, no network
set -a; . ../.env; set +a; julia --project=. examples/live_smoke.jl receipts/2026-09-26-live-smoke --managed
python3 ../lm15-contract/tools/check_secrecy.py --root receipts/2026-09-26-live-smoke
```

## Registering in Julia's General registry

1. Push `main` with the release commit; CI green (tests on three OSes and two Julia
   versions, the contract job, the lowest-compatible-versions job, the docs build).
2. On the release commit, comment `@JuliaRegistrator register` (the Registrator GitHub
   app must be installed on the repository). TagBot then tags `v1.0.0`.
3. **The name needs a human reviewer.** The registry's automatic merge refuses `LM15` on
   three name rules: shorter than 5 characters, all capitals, and close to `LMDB` and
   `MD5`. Registration still works, but the pull request waits for a registry
   maintainer; explain there that `lm15` is the project's name in every language
   (Python `lm15`, npm `@lm15/lm15`, crates.io `lm15`, Go `lm15-go`). Renaming the
   Julia package instead would change `using LM15` for every user and is a product
   decision, not a packaging one.

Later releases: bump `version` in `Project.toml`, update `CHANGELOG.md`, move
`CONTRACT_PIN` in the same commit as the code that needed it, and register again.
