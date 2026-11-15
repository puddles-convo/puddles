#!/bin/sh
# Held-out check: the final policy, the like-for-like control and the starting policy on 20 validation-split
# scenarios that the gatekeeper never used. Loop judge only.
T=<run-folder>
H=$T/controls/heldout_ids.txt
date "+start %H:%M:%S"
sh $T/recheck_ids.sh $T/train-root2/policies/policy-v4.md 1 heldout-v4 $H &
sleep 8
sh $T/recheck_ids.sh $T/controls/baseline-plain-s2.md 1 heldout-control $H &
sleep 8
sh $T/recheck_ids.sh $T/fixD/policies/training-2026-11-15/policy-v1.md 1 heldout-v1 $H &
wait
date "+end %H:%M:%S"
python3 - <<PY
import json, glob
for name in ("heldout-v1", "heldout-control", "heldout-v4"):
    for f in sorted(glob.glob("$T/" + name + "/*/result.json")):
        d = json.load(open(f))
        print(name, {k: d[k] for k in ("n", "unscored", "empathy", "relevance", "quality", "emotion_change", "cost_usd") if k in d})
PY
