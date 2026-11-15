#!/bin/sh
# The real training loop: real gatekeeper, judge gpt-6.1-sol with the strict grading guide.
# Restarts itself after a failure stop (exit 3) or a crash, up to 6 times, 5 minutes apart.
# Stops for good on a cap (exit 0) or a refusal before start (exit 2).
cd <run-folder>/run || exit 9
export PUDDLES_JUDGE_SKILL=calibration/judge-skill-strict/SKILL.md
export PUDDLES_GATE_REMEASURE_KEPT=1
export PUDDLES_PARK_REVERTED=1
export PUDDLES_PARK_AFTER=1
n=0
while :; do
  date "+=== start %Y-%m-%d %H:%M:%S (restart $n)" >> <run-folder>/train.log
  caffeinate -i sh <run-folder>/with_env_train.sh env PUDDLES_CONFIG=<run-folder>/train.toml <venv>/bin/python -m scripts.run_loop --root <run-folder>/train-root2 --policy policies/training-2026-11-15/policy-v1.md --gate real >> <run-folder>/train.log 2>&1
  code=$?
  date "+=== exit $code at %H:%M:%S" >> <run-folder>/train.log
  [ "$code" = "0" ] && break
  [ "$code" = "2" ] && break
  n=$((n+1))
  [ $n -gt 6 ] && break
  sleep 300
done
echo "EXIT $code after $n restarts" > <run-folder>/train.exit
