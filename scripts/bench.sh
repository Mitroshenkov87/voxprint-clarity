#!/bin/sh
# One command for a rented RTX 4090 VM. See docs/BENCHMARK.md.
exec python -m clarity.bench "$@"
