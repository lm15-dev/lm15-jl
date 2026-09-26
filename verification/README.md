# Verification record (historical)

These are records of earlier source snapshots, kept unchanged. The checks for the
1.0.0 release (2026-09-26) are in [RELEASING.md](../RELEASING.md) and
[docs/src/project/status.md](../docs/src/project/status.md); its live receipts are in
[receipts/2026-09-26-live-smoke](../receipts/2026-09-26-live-smoke/SUMMARY.md).

**Latest executed record:** [function tools and scientific integrations](function-tools/README.md)
records 874 normal test assertions, 48 scientific integration assertions and all
1,380 shared contract passes. The files described below retain the earlier
725-assertion verification snapshot and its source digest.

These checks use synthetic credentials, temporary home directories and network
namespaces. Only the loopback interface is enabled for native HTTP/WebSocket tests;
no provider service or real credential store is contacted.

## Evidence

- `unit-tests.log`: the normal `Pkg.test(; allow_reresolve=false)` entry point,
  including API/serde regressions, event-stream lifecycle checks, real local HTTP
  and WebSocket traffic, redaction of exception stacks, private atomic writes,
  token-refresh reuse, and lock contention across Julia processes.
- `contract.log` and `contract/*.json`: every direction of the unchanged shared
  contract harness at the commit in `../CONTRACT_PIN`. Each report states whether
  the shim was network-sandboxed. No comparison rule or fixture was changed.
- `request-probes.json`: 42 additional request combinations compared against the
  Python reference using the contract's strict comparator. All differences are
  retained, not filtered into a passing count.
- `summary.json`: final counts, platform and source digest for the verified code.

## Outside-corpus findings

The 42 additional requests produced 38 identical results and four differences:

| Probe | Python reference | Julia | Existing rule |
|---|---|---|---|
| `openai.assistant-media` | Builds a request after dropping the assistant image | Refuses unsupported assistant-image replay | MAP-10: native content or an explicit refusal |
| `openai-chat.assistant-media` | Builds a request after dropping the assistant image | Refuses unsupported assistant-image replay | MAP-10 |
| `meta-anthropic.stored-cache` | Drops the supplied stored-cache reference | Refuses a resource this dialect cannot reference | MAP-6 rule 7 |
| `deepseek-anthropic.stored-cache` | Drops the supplied stored-cache reference | Refuses a resource this dialect cannot reference | MAP-6 rule 7 |

The two stored-cache cases initially agreed because both implementations dropped
the field. Reviewing the emitted request exposed the loss; the Julia mapping was
fixed and regression cases now cover resource/key refusals on the implicit-cache
Messages bindings. These are reference implementation findings, not grounds for
changing the contract. New provider-supported replay mappings require evidence;
none is guessed here.

## Reproduction on the development machine

The machine's shared Julia depot contained empty dependency source files. Checks
therefore used a fresh, separate depot at `/tmp/lm15-julia-clean-depot`. The shared
depot was not repaired or altered by this verification pass.

```sh
export JULIA_DEPOT_PATH=/tmp/lm15-julia-clean-depot
export JULIA_PKG_PRECOMPILE_AUTO=0

# The ordinary test entry point (uses local test servers, never providers).
julia --startup-file=no --project=. -e 'using Pkg; Pkg.test(; allow_reresolve=false)'

# Enforces a no-network sandbox and checks CONTRACT_PIN itself.
python3 tools/check_contract.py --direction all

# Returns nonzero when reference differences exist; inspect, do not discard them.
python3 tools/probe_requests.py
```

For the recorded unit run, Julia was additionally launched under `unshare -rn`,
with a new temporary HOME and only `lo` brought up using `ip link set lo up`.

## Limits of this evidence

This record proves neither successful authentication with a real cloud account
nor live model behavior. MacOS/Windows and Julia 1.10 need their own runs; the CI
matrix is configured for those targets but has not been run from this checkout.
The uncommon cloud-login mechanisms listed in the main README remain explicit
unsupported paths. Browser WebAssembly is not a supported Julia deployment.
