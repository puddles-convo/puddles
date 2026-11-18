# Shared by scripts/arms_run.sh and scripts/arms_measure.sh: the models, judge and loop settings
# of both arms. Sourced, not run. Needs HELPER_CHOICE (qwen or luna) and the repo root as cwd.
# Keys come from ENV_FILE (default ../puddles/.env), loaded without printing anything.
choice=${HELPER_CHOICE:?set HELPER_CHOICE to qwen or luna}
[ -f .env ] && ENV_FILE=${ENV_FILE:-.env}  # this worktree first (git-ignored), then the main folder
ENV_FILE=${ENV_FILE:-../puddles/.env}
[ -f "$ENV_FILE" ] || { echo "no env file at $ENV_FILE (set ENV_FILE)"; exit 2; }
set -a; . "$ENV_FILE"; set +a
missing=""
need="OPENAI_API_KEY EXA_API_KEY PAST_API_KEY"
[ "$choice" = qwen ] && need="$need MODAL_HELPER_API_KEY"
for k in $need; do
  eval "v=\${$k:-}"; [ -n "$v" ] || missing="$missing $k"
done
[ -z "$missing" ] || { echo "not set in $ENV_FILE:$missing"; exit 2; }

case $choice in
  qwen)
    [ -n "${MODAL_HELPER_API_KEY:-}" ] || { echo "MODAL_HELPER_API_KEY is not set in $ENV_FILE"; exit 2; }
    export HELPER_URL=${MODAL_HELPER_URL:?set MODAL_HELPER_URL in .env: https://<workspace>--puddles-helper-serve.us-east.modal.run/v1}
    export HELPER_MODEL=qwen3.5-9b HELPER_API_KEY="$MODAL_HELPER_API_KEY"
    export HELPER_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}'
    unset HELPER_MAX_TOKENS_PARAM 2>/dev/null || true ;;
  luna)  # docs/specs/provider-settings.md, as in last night's run
    export HELPER_URL=https://api.openai.com/v1 HELPER_MODEL=gpt-6-luna HELPER_API_KEY="$OPENAI_API_KEY"
    export HELPER_MAX_TOKENS_PARAM=max_completion_tokens
    export HELPER_EXTRA_BODY='{"temperature": 1, "reasoning_effort": "none"}' ;;
  *) echo "HELPER_CHOICE must be qwen or luna"; exit 2 ;;
esac
export SIMUSER_URL=https://api.openai.com/v1 SIMUSER_MODEL=gpt-6-luna SIMUSER_API_KEY="$OPENAI_API_KEY"
export SIMUSER_MAX_TOKENS_PARAM=max_completion_tokens
export SIMUSER_EXTRA_BODY='{"temperature": 1, "reasoning_effort": "none"}'
export PUDDLES_AGENT_MODEL=gpt-6-luna PUDDLES_AGENT_TOOL_REASONING_EFFORT=none
export PUDDLES_JUDGE=fallback PUDDLES_FALLBACK_JUDGE_MODEL=gpt-6.1-sol PUDDLES_JUDGE_TEMPERATURE=none
export PUDDLES_TINKER_BASE_URL=https://api.openai.com/v1 TINKER_API_KEY="$OPENAI_API_KEY"
export PUDDLES_JUDGE_SKILL=calibration/judge-skill-strict/SKILL.md
export PUDDLES_SAFETY_MODEL=gpt-6.1-sol
export PUDDLES_CONFIG=${ARMS_CONFIG:-config/qwen-2026-11-15.toml}
export PUDDLES_GATE_REMEASURE_KEPT=1 PUDDLES_PARK_REVERTED=1 PUDDLES_PARK_AFTER=1  # as run_train.sh
export PUDDLES_TOKEN_CAP=1400  # both arms: room for cards and examples, same for the rules-only arm
export PUDDLES_PROTECT_C8=1  # as in the run of 2026-11-15 (C8 is editable by default on integration)
export PUDDLES_CONVERSATION_WORKERS=${PUDDLES_CONVERSATION_WORKERS:-8}  # conversations at once (loop/runner.py)

