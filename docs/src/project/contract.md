# The shared contract

LM15's shared contract describes the representation and provider mappings used by
the language implementations. This manual teaches the Julia API; it does not replace
that authority or grant the Python reference automatic precedence over fixtures.

## Read the exact pinned rules

The build resolves these links using this checkout's `CONTRACT_PIN`:

- [Authority](https://github.com/lm15-dev/lm15-contract/blob/@CONTRACT_PIN@/AUTHORITY.md)
- [Canonical types](https://github.com/lm15-dev/lm15-contract/blob/@CONTRACT_PIN@/spec/types.md)
- [Invariants](https://github.com/lm15-dev/lm15-contract/blob/@CONTRACT_PIN@/spec/invariants.md)
- [Vocabulary](https://github.com/lm15-dev/lm15-contract/blob/@CONTRACT_PIN@/spec/vocabularies.md)
- [Authentication](https://github.com/lm15-dev/lm15-contract/blob/@CONTRACT_PIN@/spec/auth.md)
- [Serialization](https://github.com/lm15-dev/lm15-contract/blob/@CONTRACT_PIN@/docs/serde-rules.md)
- [Mapping rules](https://github.com/lm15-dev/lm15-contract/blob/@CONTRACT_PIN@/docs/mapping-rules.md)

A wire claim needs provider evidence. A canonical representation claim follows the
normative rules. A failing implementation is not a reason to edit a fixture to match
its output. Record and review actual differences instead.

## Julia conveniences do not change the wire vocabulary

Julia constructors, tuples, typed function bindings and scoped `do` forms are local
API choices. Requests still serialize canonical messages, tool specifications and
configuration. A Julia function never enters a request.

Opaque schema, tool-input and replay dictionaries retain their contents; typed
optional fields have their own omission rules. Canonical indices, including part
and batch-entry indices, are zero-based even though Julia arrays usually start at 1.

Custom data codecs should produce existing JSON/content representations. Do not
introduce a new canonical Part subtype merely to serialize an application object.
A new protocol variant needs contract work, not just an extra Julia method.

## Conformance workflow

`bin/vet.jl` speaks the shared JSONL protocol. `tools/check_contract.py` uses the
sibling contract harness and its unchanged comparator. See
[contributing](contributing.md) for commands, and [status](status.md) for what was
actually executed. Writing a new command or workflow does not mean it has passed.
