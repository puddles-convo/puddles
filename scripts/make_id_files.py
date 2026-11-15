"""Write the scenario ID files the gate and the overnight run read.

    python -m scripts.make_id_files --seed 7

Writes `fixtures/validation_ids.txt` (N IDs sampled from the ESConv validation split,
sorted) and `fixtures/train_ids.txt` (every ESConv training ID). The files hold IDs only:
no ESConv text is committed. The test split is never read.

The same seed always draws the same validation IDs, so the file can be regenerated and
checked against what is committed.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import NoReturn, Sequence

from puddles.baselines import BaselineError, load_id_file

DEFAULT_N = 20


def pick_ids(all_ids: Sequence[str], n: int, seed: int) -> list[str]:
    """A deterministic seeded sample of n IDs, sorted. n larger than the list is an error."""
    if n > len(all_ids):
        raise BaselineError(f"asked for {n} IDs, only {len(all_ids)} available")
    return sorted(random.Random(seed).sample(sorted(all_ids), n))


def _write_ids(path: Path, ids: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")


def _print_error(message: str) -> None:
    print("error: " + " ".join(message.split()), file=sys.stderr)


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        _print_error(message)
        sys.exit(2)


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    p = _Parser(prog="python -m scripts.make_id_files", description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, required=True,
                   help="the seed that picks the validation IDs; written into the stage file")
    p.add_argument("--validation-n", type=int, default=DEFAULT_N)
    p.add_argument("--validation-out", type=Path, default=Path("fixtures/validation_ids.txt"))
    p.add_argument("--train-out", type=Path, default=Path("fixtures/train_ids.txt"))
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    try:
        from puddles.eval.data import load_scenarios

        validation_ids = pick_ids([s.scenario_id for s in load_scenarios("validation")],
                                  args.validation_n, args.seed)
        train_ids = sorted(s.scenario_id for s in load_scenarios("train"))
        overlap = set(validation_ids) & set(train_ids)
        if overlap:
            raise BaselineError(f"scenario IDs in both files: {sorted(overlap)[:3]}")
        if any(i.startswith("esconv-test-") for i in validation_ids + train_ids):
            raise BaselineError("a test-split ID was selected; the test split is never used")
        _write_ids(args.validation_out, validation_ids)
        _write_ids(args.train_out, train_ids)
        load_id_file(args.validation_out)  # the check the consumers will run
        load_id_file(args.train_out)
    except (BaselineError, OSError, ValueError) as exc:
        _print_error(f"{type(exc).__name__}: {exc}")
        return 2
    print(f"wrote {len(validation_ids)} validation IDs to {args.validation_out} "
          f"(seed {args.seed})")
    print(f"wrote {len(train_ids)} training IDs to {args.train_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
