#!/bin/sh
# Step 4 of the toolbox plan: measure the starting policy and each arm's final policy the way the
# overnight run was measured: 2 fresh sets on the gatekeeper's 20 validation scenarios and 1 set on
# the 20 held-out scenarios, scored by the loop judge (gpt-6.1-sol) and again by the final judge
# (gpt-6-astra), which took no part in either loop. Then the paired comparison (scripts/arms_stats.py).
#
#   HELPER_CHOICE=qwen sh scripts/arms_measure.sh
set -eu
cd "$(dirname "$0")/.."
. scripts/toolbox_env.sh
PY=${PY:-"uv run --no-project --python 3.12 --with-requirements requirements.txt python"}
START=policies/training-2026-11-15/policy-v1.md
HELD=results/training-2026-11-15/measurements/heldout/heldout_ids.txt
VAL=fixtures/validation_ids.txt
OUT=runs/$choice-measure
mkdir -p "$OUT/astra"

final() {  # the policy in force at the end of an arm, from its decision log (replay-checked)
  $PY -c "import sys; from pathlib import Path; from puddles.decision_log import DecisionLog; \
from puddles import replay; r=Path(sys.argv[1]); print(replay.check(DecisionLog(r), r, Path(sys.argv[2])))" "$1" "$START"
}
if [ -d runs/$choice-arm-policy ]; then
  P_POLICY=$(final runs/$choice-arm-policy)
else  # no rules-only arm in this pipeline: last night's rules-only final policy (same models with luna)
  P_POLICY=results/training-2026-11-15/run/policies/policy-v4.md
  echo "no runs/$choice-arm-policy: the rules-only policy is last night's $P_POLICY (13 rounds, not 6)"
fi
P_TOOLBOX=$(final runs/$choice-arm-toolbox)
echo "start: $START | arm A final: $P_POLICY | arm B final: $P_TOOLBOX"
printf 'start %s\npolicy %s\ntoolbox %s\n' "$START" "$P_POLICY" "$P_TOOLBOX" > "$OUT/policies.txt"

sets() {  # sets <label> <policy file>: three sets, one after another
  for spec in "val 1 $VAL" "val 2 $VAL" "held 1 $HELD"; do
    set -- "$1" "$2" $spec
    $PY -m scripts.score_baselines --policy "$2" --ids "$5" --run "$4" --seed 7 --max-turns 6 \
      --root "$OUT/$1-$3" >> "$OUT/$1.log" 2>&1 || echo "$1 $3 run $4 INCOMPLETE (see $OUT/$1.log)"
  done
}
sets start "$START" &
sets policy "$P_POLICY" &
sets toolbox "$P_TOOLBOX" &
wait
grep -h -E "complete|INCOMPLETE" "$OUT"/start.log "$OUT"/policy.log "$OUT"/toolbox.log || true

# the final judge on the very same conversations
for f in "$OUT"/*-val/*/conversations.jsonl "$OUT"/*-held/*/conversations.jsonl; do
  label=$(echo "$f" | sed -E "s#$OUT/([a-z]+)-(val|held)/[^/]*-run-([0-9]+)/conversations.jsonl#\1-\2-\3#")
  [ -f "$OUT/astra/$label.summary.json" ] && continue
  PUDDLES_FALLBACK_JUDGE_MODEL=gpt-6-astra $PY results/training-2026-11-15/measurements/tools/judge_saved.py \
    "$label" "$f" "$OUT/astra" 10 50 2>&1 | grep -v -E "INFO|mlflow" | cut -c1-300
done
$PY scripts/arms_stats.py "$OUT"
