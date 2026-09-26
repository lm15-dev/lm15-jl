# Changelog

## 1.0.0 — 2026-09-26

The first release of lm15 for Julia. Graded by lm15-contract at the commit in
`CONTRACT_PIN`: 1,788 of 1,788 checks, the same count as Python, TypeScript, Rust and Go.
Brought up from the 2026-09-11 contract (`cfed007`) with everything ratified since:

- **Providers.** DeepInfra, Together AI, Fireworks AI and Parasail (registry, presets
  with per-model rules, litellm prefixes); TypeSafe (`TypeSafeLM`). The compat knob
  `reasoning_off`. Chat model listings read a bare array; a flat `cached_tokens` counts.
  Provider tables, routing rules, litellm prefixes and the MAP-15 "no such model" forms
  are copied from the reference as data (`tools/import_tables.py`).
- **Judgments (MAP-14).** `DataPart`/`data`, `Config.probabilities`, the schema helpers
  `judgments`, `choice`, `yes_no`, `score`; the Anthropic and Gemini schema rewrites;
  TypeSafe's classification; candidate-sequence likelihood on vLLM; `data(answer)`,
  `probabilities(answer)`, `data_part(answer)`, `expected_level`.
- **Gemini schemas (MAP-16).** A schema goes to `responseJsonSchema`/`parametersJsonSchema`
  exactly when Gemini's OpenAPI field cannot carry it.
- **Refusals name their field.** Message media a wire has no slot for, and stored-cache
  resources where there is no such tier, raise `UnsupportedFeatureError` with `feature`.
- **Cloud identity.** Named credentials (`platform`, `workload`, `environment`, `cli`),
  `CredentialSource` provenance on auth errors, endpoint roots and the vendors' endpoint
  variables, settings with their origin in `explain_auth`, the Google project from
  gcloud's configuration and the metadata server, API keys on `vertex`, token-shaped
  strings sent as bearer tokens, Google IAM guidance on 401/403.
- **Managed sign-in (AUTH-12–26).** `Auth` over the shared credential file or memory:
  xAI, Claude, ChatGPT, GitHub Copilot, Kimi Code, Meta and OpenRouter sign-ins
  (profiles copied from the contract), saved keys and recipes, replacement, logout with
  the suppression marker, durable cancellation, renewal under the cross-process lock,
  bound clients, `LM15.Interactive.connect`, managed routers and the managed doctor.
  R3: an unusable or signed-out xAI subscription blocks the ambient key.
- **Reply faults and diagnostics.** Compressed replies inflated or refused by name
  (INV-053); a non-JSON success is a `ProviderError` (INV-054); invalid Unicode refused
  before the wire (INV-055); rate-limit header snapshots, millisecond retry hints and
  more request-id headers on every HTTP error; handshake evidence on in-stream errors.
- **Transport.** `Timeouts` and `max_connections` with the ratified defaults.
- **Live sessions.** Bounded turn collection (`CollectionLimitError`).
- **Streams.** Scores kept for whole tokens before a client-side stop, coverage marked
  incomplete otherwise (`logprobs_complete`).
- **Routing.** Router-local provider declarations (`RouterConfig(providers=...)`);
  a router-made `CachedPrefix` keeps its provider; a client strips its own
  `provider:` prefix.
- **Package.** Registrable `Project.toml` (compat for every dependency, tested at the
  lower bounds on Julia 1.10), Aqua checks, CI for the contract harness, CompatHelper,
  TagBot; the manual builds in a network-free sandbox.

## 0.3.0 and earlier

Development versions, installed from GitHub only.
