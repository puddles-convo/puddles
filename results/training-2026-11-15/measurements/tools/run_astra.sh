#!/bin/sh
# The final judge (gpt-6-astra, strict guide) on the saved validation conversations of each policy version.
# usage: run_astra.sh <label> [<label> ...]   labels name files in astra/sets/<label>.conversations.jsonl
T=/Users/moamen/.hermes/cache/scratch/train
cd $T/fixD || exit 9
for label in "$@"; do
  [ -f $T/astra/$label.summary.json ] && { echo "$label: already judged"; continue; }
  PUDDLES_JUDGE_SKILL=$T/strict/SKILL-strict.md sh $T/with_env_train.sh env PYTHONPATH=$T/fixD PUDDLES_CONFIG=$T/train.toml PUDDLES_FALLBACK_JUDGE_MODEL=gpt-6-astra /Users/moamen/Documents/puddles-runall-venv/bin/python $T/judge_saved.py $label $T/astra/sets/$label.conversations.jsonl $T/astra 10 50 2>&1 | grep -v "INFO\|mlflow" | cut -c1-700
done
date "+done %H:%M:%S"
