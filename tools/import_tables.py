#!/usr/bin/env python3
"""Copy the reference's policy tables into src/data/*.json, as data.

Usage: python3 tools/import_tables.py [path/to/lm15-python]

The contract's port rule 2: mapping tables (compat presets, the provider
registry, access policies) are data, copied, never re-derived from memory or
provider documentation. This tool loads the reference package from the given
checkout and serializes the tables it holds at import time: the exact values
the reference uses, including entries it computes (a model list expanded into
per-model overrides). Only non-default fields are written, so a new field with a
default leaves the JSON unchanged until a table sets it.

It imports the reference's own modules (stdlib only, no network, no side effect
beyond defining tables); it does not run any request code.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "lm15-python"
sys.path.insert(0, str(REFERENCE))

from lm15 import compat as _compat  # noqa: E402
from lm15 import registry as _registry  # noqa: E402
from lm15 import router as _router  # noqa: E402

CONTRACT = ROOT.parent / "lm15-contract"


# Compat policies: None means "inherit", so an absent key is exactly the value.
# Access policies are written whole, so the port never depends on its own
# defaults agreeing with the reference's.
ELIDE_DEFAULTS = (_compat.OpenAIChatCompat, _compat.OpenAIResponsesCompat, _compat.AnthropicCompat)


def plain(value):
    """A JSON value for a table entry."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        out = {}
        for f in dataclasses.fields(value):
            v = getattr(value, f.name)
            if not isinstance(value, ELIDE_DEFAULTS):
                out[f.name] = plain(v)
                continue
            if f.default is not dataclasses.MISSING:
                default = f.default
            elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                default = f.default_factory()  # type: ignore[misc]
            else:
                default = dataclasses.MISSING
            if default is not dataclasses.MISSING and v == default:
                continue
            out[f.name] = plain(v)
        return out
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (frozenset, set)):
        return sorted(plain(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


compat = {
    name: plain(getattr(_compat, name))
    for name in (
        "OPENAI_CHAT_PRESETS",
        "OPENAI_CHAT_PRESET_BASE_URLS",
        "OPENAI_RESPONSES_PRESETS",
        "OPENAI_RESPONSES_PRESET_BASE_URLS",
        "ANTHROPIC_PRESETS",
        "ANTHROPIC_PRESET_BASE_URLS",
    )
}

rows = []
for definition in _registry.PROVIDERS.values():
    row = {"id": definition.id, "dialect": definition.dialect, "access": plain(definition.access)}
    if definition.compat is not None:
        row["compat"] = definition.compat
    elif definition.id == "xai":
        row["compat"] = "xai"  # XaiLM is the Chat dialect bound to the xai preset (providers/xai.py)
    for name in ("aliases", "placeholder_key", "console_url", "note"):
        v = getattr(definition, name, None)
        if v:
            row[name] = plain(v)
    rows.append(row)

routing = {
    "DEFAULT_RULES": [plain(rule) for rule in _router.DEFAULT_RULES],
    "LITELLM_PROVIDER_PREFIXES": plain(_router.LITELLM_PROVIDER_PREFIXES),
}
# MAP-15: the contract's own table, verbatim (the reference carries a copy too).
not_found = json.loads((CONTRACT / "spec" / "model-not-found.json").read_text())["forms"]

out = ROOT / "src" / "data"
out.mkdir(exist_ok=True)
for name, data in (("compat", compat), ("providers", rows), ("routing", routing), ("model_not_found", not_found)):
    (out / f"{name}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
print(f"{len(rows)} providers; " + ", ".join(f"{k}: {len(v)}" for k, v in compat.items()))
