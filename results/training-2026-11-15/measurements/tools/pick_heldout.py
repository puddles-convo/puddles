"""Pick 20 held-out scenarios from the ESConv validation split: none of them is in the gatekeeper's 20."""
import os
import random

from puddles.eval.data import load_scenarios

val = load_scenarios("validation")
ids = [getattr(s, "id", None) or s.scenario_id for s in val]
used = {line.strip() for line in open("fixtures/validation_ids.txt") if line.strip()}
print("validation split size:", len(ids), "| in the gatekeeper's set:", len(used.intersection(ids)))
rest = sorted(set(ids) - used)
random.Random(20261115).shuffle(rest)
held = sorted(rest[:20])
path = os.path.expanduser("<run-folder>/controls/heldout_ids.txt")
with open(path, "w") as f:
    f.write("\n".join(held) + "\n")
print("held-out IDs written:", len(held), held[:4], "...")
print("overlap with the gatekeeper's set:", len(used.intersection(held)))
