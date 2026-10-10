# One command for a rented RTX 4090 VM. See docs/BENCHMARK.md.
python -m clarity.bench @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
