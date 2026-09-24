#!/usr/bin/env python3
"""Copy literal policy data; never import or execute the reference package.

Usage: python3 tools/import_tables.py [path/to/lm15-python/lm15]
"""
import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "lm15-python" / "lm15"
values = {}


def literal(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List)):
        return [literal(x) for x in node.elts]
    if isinstance(node, ast.Dict):
        return {literal(k): literal(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.Name):
        return values[node.id]
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "_access":
        return values[node.attr]
    if isinstance(node, ast.Subscript):
        return literal(node.value)[literal(node.slice)]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return literal(node.left) + literal(node.right)
    if isinstance(node, ast.JoinedStr):
        return "".join(str(literal(x.value)) if isinstance(x, ast.FormattedValue) else x.value for x in node.values)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        if name in ("OpenAIChatCompat", "OpenAIResponsesCompat", "AnthropicCompat", "EndpointSupport", "AccessPolicy", "HostSpec"):
            return {k.arg: literal(k.value) for k in node.keywords}
        if name == "HostSetting":
            return {"name": literal(node.args[0]), **{k.arg: literal(k.value) for k in node.keywords}}
    raise ValueError(ast.dump(node))


for filename in ("auth.py", "compat.py", "access.py"):
    tree = ast.parse((SOURCE / filename).read_text())
    for statement in tree.body:
        if isinstance(statement, ast.AnnAssign):
            target, value = statement.target, statement.value
        elif isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target, value = statement.targets[0], statement.value
        else:
            continue
        try:
            result = literal(value)
            if isinstance(target, ast.Name):
                values[target.id] = result
            elif isinstance(target, ast.Subscript):
                literal(target.value)[literal(target.slice)] = result
        except (KeyError, ValueError, TypeError):
            pass  # Only literal declarations are input data.

compat = {name: values[name] for name in (
    "OPENAI_CHAT_PRESETS", "OPENAI_CHAT_PRESET_BASE_URLS", "OPENAI_RESPONSES_PRESETS",
    "OPENAI_RESPONSES_PRESET_BASE_URLS", "ANTHROPIC_PRESETS", "ANTHROPIC_PRESET_BASE_URLS")}
owned = {"openai": "OPENAI_API", "openai-chat": "OPENAI_CHAT_API", "anthropic": "ANTHROPIC_API",
         "gemini": "GEMINI_API", "xai": "XAI", "claude-code": "CLAUDE_CODE", "openai-codex": "OPENAI_CODEX"}
rows = []
tree = ast.parse((SOURCE / "registry.py").read_text())
for statement in tree.body:
    if not isinstance(statement, ast.AnnAssign) or not isinstance(statement.target, ast.Name) or statement.target.id != "_DEFINITIONS":
        continue
    for call in statement.value.elts:
        helper = call.func.id
        kwargs = {k.arg: literal(k.value) for k in call.keywords}
        if helper == "_adapter_owned":
            provider, dialect = literal(call.args[0]), literal(call.args[1])
            if provider not in owned:
                continue  # a provider with a dialect of its own that Julia does not implement (typesafe)
            policy = values[owned[provider]]
        else:
            policy = literal(call.args[0])
            provider = policy["provider"]
            dialect = literal(call.args[1]) if helper == "_hosted" else {
                "_chat_bound": "openai-chat", "_responses_bound": "openai-responses", "_anthropic_bound": "anthropic"}[helper]
        if helper == "_chat_bound":
            kwargs.setdefault("compat", provider)
        if provider == "xai":
            kwargs["compat"] = "xai"
        rows.append({"id": provider, "dialect": dialect, "access": policy, **kwargs})

out = ROOT / "src" / "data"
out.mkdir(exist_ok=True)
for name, data in (("compat", compat), ("providers", rows)):
    (out / f"{name}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
