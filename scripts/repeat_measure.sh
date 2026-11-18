#!/bin/sh
# Measure the control (baseline), policy-v1 and policy-v4 N times each on the 20 held-out scenarios, with
# last night's models and judge (helper gpt-6-luna; judge gpt-6.1-sol, strict guide), then compare them.
# How much of last night's gain survives repeated sampling, and how much one set of 20 moves on its own.
#
#   N=10 P=6 sh scripts/repeat_measure.sh      # ~$0.33 a set: ~$10 for 30 sets; P sets at a time
set -eu
cd "$(dirname "$0")/.."
export HELPER_CHOICE=luna
. scripts/toolbox_env.sh
export PUDDLES_CONFIG=config/train-2026-11-15.toml
PY=${PY:-"uv run --no-project --python 3.12 --with-requirements requirements.txt python"}
M=results/training-2026-11-15
HELD=$M/measurements/heldout/heldout_ids.txt
OUT=runs/repeat-measure
N=${N:-10}
P=${P:-6}
mkdir -p $OUT
jobs=$OUT/jobs.txt
: > $jobs
for i in $(seq 1 "$N"); do
  echo "control $M/measurements/controls/baseline-plain-s2.md $i" >> $jobs
  echo "v1 policies/training-2026-11-15/policy-v1.md $i" >> $jobs
  echo "v4 $M/run/policies/policy-v4.md $i" >> $jobs
done
export PY HELD OUT
# one set: score_baselines writes <root>/<version>-run-<i>/{conversations,scores}.jsonl and result.json
xargs -P "$P" -L 1 sh -c '
  [ -f "$OUT/$0/$(basename "$1" .md)-done-$2" ] && exit 0
  $PY -m scripts.score_baselines --policy "$1" --ids "$HELD" --run "$2" --seed 7 --max-turns 6 \
    --root "$OUT/$0" >> "$OUT/$0.log" 2>&1 && touch "$OUT/$0/$(basename "$1" .md)-done-$2" \
    || echo "$0 run $2 INCOMPLETE"
' < $jobs
$PY scripts/repeat_stats.py $OUT
