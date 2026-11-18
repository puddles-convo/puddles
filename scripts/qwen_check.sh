#!/bin/sh
# Step 1 of the toolbox plan: the overnight run's three policies on the 20 held-out scenarios,
# with the helper on our own Qwen3.5-9B (Modal) instead of gpt-6-luna. Same simulated person,
# judge, strict guide, seed and turns as results/training-2026-11-15/measurements/tools/recheck_ids.sh.
#
#   sh scripts/qwen_check.sh smoke    # policy-v4 on 2 held-out scenarios (a few cents)
#   sh scripts/qwen_check.sh full     # policy-v1, control, policy-v4 on all 20, in parallel
#
# Keys come from ENV_FILE (default ../puddles/.env), loaded without printing anything.
# Needs OPENAI_API_KEY and MODAL_HELPER_API_KEY (the key in the Modal Secret puddles-helper-key).
# HELPER_API_KEY is never used here: in .env it may hold another provider's key.
set -eu
mode=${1:-smoke}
cd "$(dirname "$0")/.."
[ -f .env ] && ENV_FILE=${ENV_FILE:-.env}  # this worktree first (git-ignored), then the main folder
ENV_FILE=${ENV_FILE:-../puddles/.env}
[ -f "$ENV_FILE" ] || { echo "no env file at $ENV_FILE (set ENV_FILE)"; exit 2; }
set -a; . "$ENV_FILE"; set +a

modal_key=${MODAL_HELPER_API_KEY:-}
[ -n "${OPENAI_API_KEY:-}" ] || { echo "OPENAI_API_KEY is not set in $ENV_FILE"; exit 2; }
[ -n "$modal_key" ] || { echo "MODAL_HELPER_API_KEY is not set in $ENV_FILE"; exit 2; }

# helper: our Qwen on Modal, thinking off
export HELPER_URL=${MODAL_HELPER_URL:?set MODAL_HELPER_URL in .env: https://<workspace>--puddles-helper-serve.us-east.modal.run/v1}
export HELPER_MODEL=qwen3.5-9b HELPER_API_KEY="$modal_key"
export HELPER_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}'
unset HELPER_MAX_TOKENS_PARAM 2>/dev/null || true
# simulated person: as in the overnight run (docs/specs/provider-settings.md)
export SIMUSER_URL=https://api.openai.com/v1 SIMUSER_MODEL=gpt-6-luna SIMUSER_API_KEY="$OPENAI_API_KEY"
export SIMUSER_MAX_TOKENS_PARAM=max_completion_tokens
export SIMUSER_EXTRA_BODY='{"temperature": 1, "reasoning_effort": "none"}'
# judge: the loop judge, single model, strict guide (gpt-6.1-sol takes only the default temperature)
export PUDDLES_JUDGE=fallback PUDDLES_FALLBACK_JUDGE_MODEL=gpt-6.1-sol PUDDLES_JUDGE_TEMPERATURE=none
export PUDDLES_TINKER_BASE_URL=https://api.openai.com/v1 TINKER_API_KEY="$OPENAI_API_KEY"
export PUDDLES_JUDGE_SKILL=calibration/judge-skill-strict/SKILL.md
export PUDDLES_CONFIG=config/qwen-2026-11-15.toml

PY=${PY:-"uv run --no-project --python 3.12 --with-requirements requirements.txt python"}
M=results/training-2026-11-15
HELD=$M/measurements/heldout/heldout_ids.txt
OUT=runs/qwen-check
mkdir -p "$OUT"

echo "waiting for the helper to be up (a cold start takes about 3.5 minutes)"
base=${HELPER_URL%/v1}
i=0
until curl -fsS -m 10 "$base/health" >/dev/null 2>&1; do
  i=$((i + 1)); [ $i -gt 60 ] && { echo "helper not up after 10 minutes"; exit 3; }
  sleep 10
done
echo "helper up"

one() {  # one() <policy file> <ids file> <root>
  $PY -m scripts.score_baselines --policy "$1" --ids "$2" --run 1 --seed 7 --max-turns 6 --root "$3"
}

case $mode in
  smoke)
    head -2 "$HELD" > "$OUT/smoke_ids.txt"
    one $M/run/policies/policy-v4.md "$OUT/smoke_ids.txt" "$OUT/smoke"
    ;;
  full)
    one policies/training-2026-11-15/policy-v1.md "$HELD" "$OUT/heldout-v1" > "$OUT/heldout-v1.log" 2>&1 &
    one $M/measurements/controls/baseline-plain-s2.md "$HELD" "$OUT/heldout-control" > "$OUT/heldout-control.log" 2>&1 &
    one $M/run/policies/policy-v4.md "$HELD" "$OUT/heldout-v4" > "$OUT/heldout-v4.log" 2>&1 &
    wait
    tail -n 1 "$OUT"/heldout-*.log
    ;;
  *) echo "usage: $0 smoke|full"; exit 2 ;;
esac

# the rule written before the run: no empty helper reply and no thinking text
$PY - "$OUT" <<'PY'
import glob, json, sys
bad, n = [], 0
for path in glob.glob(f"{sys.argv[1]}/**/conversations.jsonl", recursive=True):
    for line in open(path):
        rec = json.loads(line)
        for t in rec["turns"]:
            if t["speaker"] == "helper":
                n += 1
                text = (t.get("text") or "").strip()
                if not text or "<think" in text.lower() or "</think>" in text.lower():
                    bad.append((rec["id"], text[:60]))
print(f"helper replies checked: {n}, unusable: {len(bad)}")
for b in bad[:10]:
    print("  ", b)
PY
