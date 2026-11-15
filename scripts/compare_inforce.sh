#!/bin/sh
# Measure the policy in force of the toolbox arm and of the reproduction N times on the 20 held-out
# scenarios, with last night's models and judge, next to the 10 sets of control, policy-v1 and policy-v4
# already in runs/repeat-measure. Then compare all five (scripts/repeat_stats.py).
#
#   N=5 P=4 sh scripts/compare_inforce.sh       # ~$0.33 a set: ~$3.30 for 10 sets
set -eu
cd "$(dirname "$0")/.."
export HELPER_CHOICE=luna
. scripts/toolbox_env.sh
export PUDDLES_CONFIG=config/train-2026-11-15.toml
PY=${PY:-"uv run --no-project --python 3.12 --with-requirements requirements.txt python"}
START=policies/training-2026-11-15/policy-v1.md
HELD=results/training-2026-11-15/measurements/heldout/heldout_ids.txt
OUT=runs/repeat-measure
N=${N:-5}
P=${P:-4}
STAMP=$(date +%H%M)

inforce() {  # the policy in force of a loop root, replay-checked, copied so a later round cannot change it
  f=$($PY -c "import sys; from pathlib import Path; from puddles.decision_log import DecisionLog; \
from puddles import replay; r=Path(sys.argv[1]); print(replay.check(DecisionLog(r), r, Path(sys.argv[2])))" "$1" "$START")
  mkdir -p runs/compare-policies
  cp "$f" "runs/compare-policies/$2-$STAMP.md"
  echo "runs/compare-policies/$2-$STAMP.md"
}
P_TOOLBOX=$(inforce runs/luna-arm-toolbox toolbox)
P_REPRO=$(inforce runs/repro-2026-11-15 repro)
echo "toolbox arm in force: $P_TOOLBOX ($(head -1 $P_TOOLBOX))"
echo "reproduction in force: $P_REPRO ($(head -1 $P_REPRO))"

jobs=$OUT/jobs-inforce-$STAMP.txt
: > $jobs
for i in $(seq 1 "$N"); do
  echo "toolbox-arm $P_TOOLBOX $i" >> $jobs
  echo "repro-arm $P_REPRO $i" >> $jobs
done
export PY HELD OUT
xargs -P "$P" -L 1 sh -c '
  $PY -m scripts.score_baselines --policy "$1" --ids "$HELD" --run "$2" --seed 7 --max-turns 6 \
    --root "$OUT/$0" >> "$OUT/$0.log" 2>&1 || echo "$0 run $2 INCOMPLETE"
' < $jobs
$PY scripts/repeat_stats.py $OUT
