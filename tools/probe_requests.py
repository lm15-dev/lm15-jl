#!/usr/bin/env python3
"""Offline request probes outside the pinned cases; report every reference difference.

Uses the contract's strict comparator. A difference is a finding, never a fixture
rewrite or an excuse to drop a canonical field.
"""
from __future__ import annotations
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT.parent / "lm15-contract"
spec = importlib.util.spec_from_file_location("lm15_contract_harness", CONTRACT / "harness/check.py")
harness = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = harness
spec.loader.exec_module(harness)


def probes():
    tool = {"type": "function", "name": "lookup", "parameters": {"type": "object", "properties": {}}}
    other = {"type": "function", "name": "weather", "parameters": {"type": "object", "properties": {}}}
    media = {"type": "image", "media_type": "image/png", "url": "https://example.org/image.png"}
    for provider, model in (("openai", "gpt-5.4-mini"), ("openai-chat", "gpt-5.4-mini"),
                            ("anthropic", "claude-sonnet-4-5"), ("gemini", "gemini-2.5-flash"),
                            ("deepseek", "deepseek-v4-flash"), ("xai", "grok-4.20")):
        base = {"model": model, "messages": [{"role": "user", "parts": [{"type": "text", "text": "hé🙂\nSay hello."}]}]}
        variants = {
            "sampling": {"config": {"max_tokens": 100, "temperature": 0.2, "top_p": 0.9}},
            "schema": {"config": {"response_format": {"type": "json_schema", "schema": {"type": "object", "properties": {}, "additionalProperties": False}, "strict": False}}},
            "allowed-auto": {"tools": [tool, other], "config": {"tool_choice": {"mode": "auto", "allowed": ["lookup"]}}},
            "cache-stable": {"system": "Reusable instructions", "config": {"cache": {"prefix": "stable", "retention": "long"}}},
            "reasoning-summary": {"config": {"reasoning": {"effort": "high", "summary": "auto"}, "max_tokens": 400}},
            "media-result": {"tools": [tool], "messages": base["messages"] + [
                {"role": "assistant", "parts": [{"type": "tool_call", "id": "call-1", "name": "lookup", "input": {}}]},
                {"role": "tool", "parts": [{"type": "tool_result", "id": "call-1", "is_error": True, "content": [media, {"type": "text", "text": "Image lookup failed."}]}]},
            ]},
        }
        for name, changes in variants.items():
            yield f"{provider}.{name}", provider, {**copy.deepcopy(base), **copy.deepcopy(changes)}
    # The corpus does not currently pin these MAP-10/MAP-6 edge combinations.
    for provider in ("openai", "openai-chat", "anthropic", "gemini"):
        yield f"{provider}.assistant-media", provider, {"model": "m", "messages": [
            {"role": "assistant", "parts": [{"type": "text", "text": "Here is the image"}, media]},
            {"role": "user", "parts": [{"type": "text", "text": "Describe it again"}]},
        ]}
    for provider in ("meta-anthropic", "deepseek-anthropic"):
        yield f"{provider}.stored-cache", provider, {"model": "deepseek-v4-flash" if provider.startswith("deepseek") else "muse-spark", "messages": [
            {"role": "user", "parts": [{"type": "text", "text": "Suffix"}]}
        ], "config": {"cache": {"resource": "cachedContents/opaque"}}}


def main():
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "verification/request-probes.json"
    depot = os.environ.get("JULIA_DEPOT_PATH", str(Path.home() / ".julia"))
    with tempfile.TemporaryDirectory(prefix="lm15-julia-probes-") as home:
        os.environ["HOME"] = home
        os.environ["JULIA_DEPOT_PATH"] = depot
        os.environ["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
        reference = harness.load_shim("python")
        julia = harness.Shim("julia", ["julia", "--startup-file=no", "--history-file=no", "--project=.", "bin/vet.jl"], ROOT)
        findings = []
        identical = 0
        try:
            if not (reference.sandboxed and julia.sandboxed):
                raise harness.HarnessError("network sandbox required")
            harness.check_pin(reference)
            harness.check_pin(julia)
            for name, provider, request in probes():
                fields = dict(provider=provider, canonical_request=request, api_key="probe-only", stream=False)
                left, right = reference.call("build_request", **fields), julia.call("build_request", **fields)
                def comparable(reply):
                    if not reply.get("ok"):
                        return {"ok": False, "type": reply["error"]["type"], "code": reply["error"].get("code")}
                    result = copy.deepcopy(reply["result"])
                    result["headers"] = {k: v for k, v in result["headers"].items() if k not in harness.DROP_HEADERS}
                    return {"ok": True, "result": result}
                diff = harness.first_difference(comparable(left), comparable(right))
                if diff is None:
                    identical += 1
                else:
                    findings.append({"id": name, "request": request, "difference": diff.to_dict(), "reference": left, "julia": right})
        finally:
            reference.close()
            julia.close()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"contract": harness.contract_head(), "identical": identical, "differences": findings}, indent=2) + "\n")
        print(f"Request probes: {identical} identical, {len(findings)} differences; {output}")
    return bool(findings)


if __name__ == "__main__":
    raise SystemExit(main())
