#!/bin/sh
# Reproduce the training run of 2026-11-15 (results/training-2026-11-15/), at three levels.
#
#   sh scripts/reproduce_run.sh tables    # free: last night's tables from the saved scores (paired_stats.py)
#   sh scripts/reproduce_run.sh measure   # ~$1, ~10 min: fresh held-out sets of policy-v1, control, policy-v4
#   sh scripts/reproduce_run.sh check     # free: the rerun's root, prices and keys, no round
#   sh scripts/reproduce_run.sh loop      # ~$7, ~1.5 h: the whole loop again from policy-v1, rules only
#
# Same as last night: models (helper, simulated person, agents gpt-6-luna; judge and safety gpt-6.1-sol
# with the strict guide), the starting policy, seed 7, 8 training conversations a round, the 20 validation
# scenarios, noise 0.275, PUDDLES_GATE_REMEASURE_KEPT=1, PUDDLES_PARK_REVERTED=1, PUDDLES_PARK_AFTER=1,
# token cap 800, C8 protected, proposer rules only. Not the same: the code is the run's final commit
# (last night restarted four times on newer commits), conversations run 8 at a time, the models sample
# afresh (luna and sol take only their default temperature), and past.dev already holds last night's
# entries (recall keeps only records whose ID and hash match this run's own log).
set -eu
mode=${1:-check}
cd "$(dirname "$0")/.."
export HELPER_CHOICE=luna
. scripts/toolbox_env.sh
export PUDDLES_PROPOSER=v1 PUDDLES_TOKEN_CAP=800
export PUDDLES_CONFIG=${REPRO_CONFIG:-config/train-2026-11-15.toml}  # last night's caps: $11.00, no round cap
PY=${PY:-"uv run --no-project --python 3.12 --with-requirements requirements.txt python"}
M=results/training-2026-11-15
START=policies/training-2026-11-15/policy-v1.md
ROOT=runs/repro-2026-11-15
OUT=runs/repro-measure

case $mode in
  tables)
    $PY $M/measurements/tools/paired_stats.py ;;
  measure)
    mkdir -p $OUT
    for spec in "v1 $START" "control $M/measurements/controls/baseline-plain-s2.md" "v4 $M/run/policies/policy-v4.md"; do
      set -- $spec
      $PY -m scripts.score_baselines --policy "$2" --ids $M/measurements/heldout/heldout_ids.txt --run 1 --seed 7 \
        --max-turns 6 --root "$OUT/heldout-$1" > "$OUT/heldout-$1.log" 2>&1 &
    done
    wait
    tail -n 1 $OUT/heldout-*.log
    echo "last night (loop judge, held-out): v1 3.38, control 3.55, v4 3.90" ;;
  check)
    $PY -m scripts.run_loop --root $ROOT --policy $START --gate real --check ;;
  loop)
    $PY -m scripts.run_loop --root $ROOT --policy $START --gate real >> $ROOT.log 2>&1
    tail -n 3 $ROOT.log ;;
  *) echo "usage: $0 tables|measure|check|loop"; exit 2 ;;
esac
