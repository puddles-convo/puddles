#!/bin/sh
# The real training loop: real gatekeeper, judge gpt-6.1-sol with the strict grading guide.
# Restarts itself after a failure stop (exit 3) or a crash, up to 6 times, 5 minutes apart.
# Stops for good on a cap (exit 0) or a refusal before start (exit 2).
cd /Users/moamen/.hermes/cache/scratch/train/run || exit 9
export PUDDLES_JUDGE_SKILL=calibration/judge-skill-strict/SKILL.md
export PUDDLES_GATE_REMEASURE_KEPT=1
export PUDDLES_PARK_REVERTED=1
export PUDDLES_PARK_AFTER=1
n=0
while :; do
  date "+=== start %Y-%m-%d %H:%M:%S (restart $n)" >> /Users/moamen/.hermes/cache/scratch/train/train.log
  caffeinate -i sh /Users/moamen/.hermes/cache/scratch/train/with_env_train.sh env PUDDLES_CONFIG=/Users/moamen/.hermes/cache/scratch/train/train.toml /Users/moamen/Documents/puddles-runall-venv/bin/python -m scripts.run_loop --root /Users/moamen/.hermes/cache/scratch/train/train-root2 --policy policies/training-2026-11-15/policy-v1.md --gate real >> /Users/moamen/.hermes/cache/scratch/train/train.log 2>&1
  code=$?
  date "+=== exit $code at %H:%M:%S" >> /Users/moamen/.hermes/cache/scratch/train/train.log
  [ "$code" = "0" ] && break
  [ "$code" = "2" ] && break
  n=$((n+1))
  [ $n -gt 6 ] && break
  sleep 300
done
echo "EXIT $code after $n restarts" > /Users/moamen/.hermes/cache/scratch/train/train.exit
