#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
: "${GLASS_DIAGRAMS_ROOT:=/Users/moamen/Documents/glass-diagrams}"
export PYTHONPATH="$GLASS_DIAGRAMS_ROOT${PYTHONPATH:+:$PYTHONPATH}"
BIN="${GLASS_DIAGRAMS_BIN:-$HOME/.local/share/glass-diagrams/uv-venv/bin}"
"$BIN/python" figures/gen_fig2.py
"$BIN/validate-diagram" figures/fig2-inference.json --profile schema --json > figures/fig2-schema.json
"$BIN/export-diagram" figures/fig2-inference.json --format animated-svg --duration 24 --output figures/fig2-inference.svg --json > figures/fig2-export-svg.json
"$BIN/build-diagram" figures/fig2-inference.json --output figures/fig2-inference.html --json > figures/fig2-build.json
"$BIN/python" figures/package_fig2.py
"$BIN/python" figures/qa_fig2.py > figures/fig2-qa-run.log
"$BIN/python" figures/fig2_replay_probe.py
"$BIN/python" figures/qa_fig2_electric.py > figures/fig2-electric-qa-run.log
"$BIN/validate-diagram" figures/fig2-inference.json --profile static --evidence-dir figures/fig2-validation/static --json > figures/fig2-static.json
