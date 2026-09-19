#!/usr/bin/env bash
# Linux-only isolated execution wrapper. Dependency installation is a separate,
# explicitly network-enabled step. This wrapper was written, not run, in this pass.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
julia_bin=$(command -v julia)
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-$HOME/.julia}"
export JULIA_PKG_PRECOMPILE_AUTO=0
home=$(mktemp -d)
trap 'rm -rf "$home"' EXIT
cd "$root"
# Do not fall back to unrestricted execution if namespaces are unavailable.
unshare -rn bash -c '
    set -euo pipefail
    ip link set lo up
    export HOME="$1" USERPROFILE="$1"
    export LM15_DOCS_ISOLATED=1 LM15_RUN_LIVE_EXAMPLES=no
    "$2" --startup-file=no --project=examples/scientific docs/scientific.jl
    "$2" --startup-file=no --project=docs docs/make.jl
' _ "$home" "$julia_bin"
