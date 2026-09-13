#!/usr/bin/env python3
"""Run the unchanged sibling contract harness against Julia.

No fixture edits, custom comparisons, or skip list. Uses the harness's own Shim,
pin check, runners and report writer; only the Julia launch command is supplied.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=ROOT.parent / "lm15-contract")
    parser.add_argument("--direction", default="all")
    parser.add_argument("--case")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "verification" / "contract")
    parser.add_argument("--julia", default="julia")
    args = parser.parse_args()
    path = args.contract.resolve() / "harness" / "check.py"
    spec = importlib.util.spec_from_file_location("lm15_contract_harness", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load contract harness: {path}")
    harness = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = harness
    spec.loader.exec_module(harness)
    if args.direction != "all" and args.direction not in harness.DIRECTIONS:
        parser.error(f"unknown direction; choose all or {', '.join(harness.DIRECTIONS)}")
    julia = shutil.which(args.julia)
    if julia is None:
        parser.error("Julia is not on PATH")
    depot = os.environ.get("JULIA_DEPOT_PATH", str(Path.home() / ".julia"))
    # Never make the caller's cloud/login environment part of a fixture.
    keep = {k: v for k, v in os.environ.items() if k.startswith(("NIX_", "LD_", "JULIA_")) or k in (
        "PATH", "SystemRoot", "WINDIR", "COMSPEC", "PATHEXT", "SSL_CERT_FILE", "SSL_CERT_DIR", "LANG", "LC_ALL",
    )}
    with tempfile.TemporaryDirectory(prefix="lm15-julia-vet-") as home:
        os.environ.clear()
        os.environ.update(keep)
        os.environ.update(HOME=home, USERPROFILE=home, JULIA_DEPOT_PATH=depot, JULIA_PKG_PRECOMPILE_AUTO="0")
        shim = harness.Shim("julia", [julia, "--startup-file=no", "--history-file=no", "--project=.", "bin/vet.jl"], ROOT)
        try:
            if not shim.sandboxed:
                raise harness.HarnessError("network sandbox unavailable; refusing to run the credential-sensitive harness")
            harness.check_pin(shim)
            capabilities = shim.call("capabilities")
            if not capabilities.get("ok"):
                raise harness.HarnessError(f"capabilities failed: {capabilities}")
            failed = False
            directions = harness.DIRECTIONS if args.direction == "all" else (args.direction,)
            for direction in directions:
                report = harness.run_direction(shim, direction, args.case, args.report_dir.resolve(), auth_scope="all")
                harness.write_reports(report, shim, capabilities["result"], args.report_dir.resolve())
                counts = report.counts
                print(f"{direction:>10}: pass {counts['pass']:3d} fail {counts['fail']:3d} skip {counts['skip']:3d}", flush=True)
                failed |= bool(counts["fail"])
            return int(failed)
        finally:
            shim.close()


if __name__ == "__main__":
    raise SystemExit(main())
