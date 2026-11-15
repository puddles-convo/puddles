#!/bin/sh
# Step 3 of the toolbox plan: the same loop twice, in parallel, differing only in the proposer.
#   arm A  runs/<helper>-arm-policy   PUDDLES_PROPOSER=v1       rules only (last night's proposer)
#   arm B  runs/<helper>-arm-toolbox  PUDDLES_PROPOSER=toolbox  rules, skill cards, examples
# Folders are per helper, so a qwen and a luna pipeline can run at the same time.
# ARMS="toolbox" runs arm B only (e.g. luna, compared with last night's rules-only run).
# Same helper, simulated person, agents, judge, strict guide, starting policy, seed, scenarios,
# noise, token cap (1400 for both), remeasure/parking settings and round cap.
#
#   HELPER_CHOICE=qwen sh scripts/arms_run.sh smoke   # toolbox arm, 1 round, spend cap $1.50
#   HELPER_CHOICE=qwen sh scripts/arms_run.sh check   # verifies both roots, prices and keys; no round
#   HELPER_CHOICE=qwen sh scripts/arms_run.sh run     # both arms in the background; logs in runs/arm-*.log
#
# HELPER_CHOICE is qwen (our Qwen3.5-9B on Modal) or luna (gpt-6-luna), decided by step 1.
# Keys come from ENV_FILE (default ../puddles/.env), loaded without printing anything:
# OPENAI_API_KEY, EXA_API_KEY, PAST_API_KEY, and MODAL_HELPER_API_KEY for qwen.
set -eu
mode=${1:-check}
cd "$(dirname "$0")/.."
[ "$mode" = smoke ] && export ARMS_CONFIG=config/smoke-2026-11-15.toml
. scripts/toolbox_env.sh

PY=${PY:-"uv run --no-project --python 3.12 --with-requirements requirements.txt python"}
START=policies/training-2026-11-15/policy-v1.md
ROUNDS=${ROUNDS:-6}
ARMS=${ARMS:-policy toolbox}  # which arms to run: policy (rules only), toolbox, or both
proposer_of() { [ "$1" = policy ] && echo v1 || echo toolbox; }

if [ "$choice" = qwen ]; then
  echo "waiting for the helper to be up"
  i=0
  until curl -fsS -m 10 "${HELPER_URL%/v1}/health" >/dev/null 2>&1; do
    i=$((i + 1)); [ $i -gt 60 ] && { echo "helper not up after 10 minutes"; exit 3; }; sleep 10
  done
fi

arm() {  # arm <proposer> <root> <extra args...>
  p=$1; root=$2; shift 2
  PUDDLES_PROPOSER=$p $PY -m scripts.run_loop --root "$root" --policy $START --gate real "$@"
}

mkdir -p runs
case $mode in
  smoke)  # the toolbox arm only, one round, $1.50 cap
    rm -rf runs/$choice-smoke-toolbox
    arm toolbox runs/$choice-smoke-toolbox --check
    arm toolbox runs/$choice-smoke-toolbox --max-rounds 1 2>&1 | tee runs/$choice-smoke-toolbox.log | grep -v -E "INFO|mlflow" ;;
  check)
    for a in $ARMS; do arm "$(proposer_of $a)" runs/$choice-arm-$a --check; done ;;
  run)
    for a in $ARMS; do
      arm "$(proposer_of $a)" runs/$choice-arm-$a --max-rounds "$ROUNDS" >> runs/$choice-arm-$a.log 2>&1 &
      sleep 20  # the arms do not start their first conversations in the same second
    done
    wait
    for a in $ARMS; do tail -n 3 runs/$choice-arm-$a.log; done ;;
  *) echo "usage: $0 smoke|check|run"; exit 2 ;;
esac
