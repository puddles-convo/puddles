"""Write the report of one training round, print it, and attach it to the round's MLflow run.

    python -m scripts.show_round --root runs/dev
    python -m scripts.show_round --root runs/dev --round 3 --mlflow sqlite:///runs/mlflow.db

It only reads what the round left behind; the one file it writes is <root>/reports/round-NN.md.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import NoReturn, Sequence

from puddles.round_view import UNREADABLE, ReportError, _why, build_report, name, read_rounds, report_path
from puddles.tracing import DEFAULT_TRACKING_URI, SQLITE_PREFIX

EXPERIMENT = "puddles"


def _print_error(message: str) -> None:
    print("error: " + " ".join(message.split()), file=sys.stderr)


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        _print_error(message)
        sys.exit(2)


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    p = _Parser(prog="python -m scripts.show_round", description=__doc__.splitlines()[0])
    p.add_argument("--root", default="runs/dev", help="the root the rounds were run on")
    p.add_argument("--round", type=int, default=None, help="the round; default: the last round in the log")
    p.add_argument("--mlflow", default=None, help="MLflow tracking address")
    return p.parse_args(argv)


def attach(path: Path, round_: int, tracking_uri: str, root: Path) -> str:
    """Attach the report to the newest run named round-NN whose tag `root` is this root.

    Several roots can share one tracking store, each with its own round-01, so a run with another
    root's tag, or with none, is never used. Returns why it was not attached, or ''.
    """
    if tracking_uri.startswith(SQLITE_PREFIX):
        db = tracking_uri[len(SQLITE_PREFIX):]
        if db and db != ":memory:" and not Path(db).exists():
            return f"there is no tracking store at {db}"  # opening it would create one
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        return f"there is no experiment {EXPERIMENT!r}"
    experiment_id = name(experiment.experiment_id)  # an ID from the store is shown or used only as a name
    if experiment_id == UNREADABLE:
        return f"the experiment {EXPERIMENT!r} has an ID that is unreadable"
    name_ = f"round-{round_:02d}"
    runs = client.search_runs([experiment_id], filter_string=f"attributes.run_name = '{name_}'",
                              order_by=["attributes.start_time DESC"])
    here = str(Path(root).resolve())
    runs = [r for r in runs if r.data.tags.get("root") == here]  # compared here: no quoting of the path needed
    if not runs:
        return f"there is no run named {name_} for the root {here}"
    run_id = name(runs[0].info.run_id)
    if run_id == UNREADABLE:
        return f"the run named {name_} for the root {here} has an ID that is unreadable"
    client.log_artifact(run_id, str(path), artifact_path="reports")
    return ""


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    root = Path(args.root)
    try:
        rounds = read_rounds(root)
        if args.round is not None:
            round_ = args.round
        elif rounds:
            round_ = max(r.round for r in rounds)
        else:
            raise ReportError(f"the log under {root} holds no round")
        text = build_report(root, round_)
    except ReportError as exc:
        _print_error(str(exc))
        return 2
    except Exception as exc:  # noqa: BLE001  build_report reads leniently; this is a fault of the view itself
        _print_error(f"the report could not be built ({_why(exc)})")
        return 1
    path = report_path(root, round_)
    print(text, end="")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as exc:  # printed all the same; with no file there is nothing to attach
        _print_error(f"the report could not be written to {path} ({_why(exc)})")
        print("The report was not attached to an MLflow run: it could not be written.")
        return 1
    uri = args.mlflow or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    try:
        why = attach(path, round_, uri, root)
    except Exception as exc:  # noqa: BLE001  a report that cannot be attached is still written and printed
        why = f"attaching it failed ({_why(exc)})"
    if why:
        print(f"The report was not attached to an MLflow run: {' '.join(why.split())}")
    else:
        print(f"The report was attached to the MLflow run round-{round_:02d} under reports/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
