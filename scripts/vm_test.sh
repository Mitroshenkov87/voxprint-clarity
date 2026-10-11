#!/bin/bash
# Linux entry for the rented GPU VM. Same flags, JSON, and exit codes as
# scripts/vm_test.ps1. Exit 0 pass, 1 test failure, 2 setup, 3 no GPU.
# Flags: --smoke, --full, --json, --results DIR, --allow-other-gpu, --allow-other-os.
# See docs/TESTING-GPU.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

JSON=0
for arg in "$@"; do
    if [[ "$arg" == "--json" ]]; then
        JSON=1
    fi
done

fail() {
    local code="$1"
    local step="$2"
    local error="$3"
    echo "FAILED: step ${step}: ${error}" >&2
    if [[ "$JSON" -eq 1 ]]; then
        printf '{"type":"summary","app":"clarity","ok":false,"exit":%s,"step":"%s","error":"%s"}\n' \
            "$code" "$step" "$error"
    fi
    exit "$code"
}

PYTHON=""
if command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
fi
if [[ -z "$PYTHON" ]]; then
    fail 2 bootstrap "python3 was not found"
fi

exec "$PYTHON" "$ROOT/scripts/vm_test.py" "$@"
