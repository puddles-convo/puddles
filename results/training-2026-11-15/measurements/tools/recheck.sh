#!/bin/sh
# Side measurement, separate from the loop: score one policy file again on the 20 validation scenarios,
# with the loop's own models and strict judge. usage: recheck.sh <policy.md> <run number> <root name>
T=<run-folder>
cd $T/fixD || exit 9
export PUDDLES_JUDGE_SKILL=calibration/judge-skill-strict/SKILL.md
sh $T/with_env_train.sh env PUDDLES_CONFIG=$T/train.toml <venv>/bin/python -m scripts.score_baselines --policy "$1" --ids fixtures/validation_ids.txt --run "$2" --seed 7 --max-turns 6 --root "$T/$3" > "$T/$3.run$2.log" 2>&1
echo "run $2 exit $?"
