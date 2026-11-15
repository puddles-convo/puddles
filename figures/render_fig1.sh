#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
: "${GLASS_DIAGRAMS_ROOT:?Set GLASS_DIAGRAMS_ROOT to the glass-diagrams engine checkout}"
export PYTHONPATH="$GLASS_DIAGRAMS_ROOT${PYTHONPATH:+:$PYTHONPATH}"
BIN="${GLASS_DIAGRAMS_BIN:-$HOME/.local/share/glass-diagrams/uv-venv/bin}"
"$BIN/python" figures/gen_fig1.py
"$BIN/validate-diagram" figures/fig1-training.json --profile schema --json > figures/fig1-schema.json
"$BIN/export-diagram" figures/fig1-training.json --format animated-svg --duration 24 --output figures/fig1-training.svg --json > figures/fig1-export-svg.json
"$BIN/export-diagram" figures/fig1-training.json --format animated-html --output figures/fig1-training.html --json > figures/fig1-export-html.json
"$BIN/python" figures/package_fig1.py
"$BIN/python" figures/qa_fig1.py
"$BIN/validate-diagram" figures/fig1-training.json --profile static --evidence-dir figures/fig1-validation-v6/static --json > figures/fig1-static.json
