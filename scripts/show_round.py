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

from puddles.round_view import ReportError, build_report, read_rounds, report_path
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


def attach(path: Path, round_: int, tracking_uri: str) -> str:
    """Attach the report to the newest run named round-NN. Returns why it was not attached, or ''."""
    if tracking_uri.startswith(SQLITE_PREFIX):
        db = tracking_uri[len(SQLITE_PREFIX):]
        if db and db != ":memory:" and not Path(db).exists():
            return f"there is no tracking store at {db}"  # opening it would create one
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        return f"there is no experiment {EXPERIMENT!r}"
    name = f"round-{round_:02d}"
    runs = client.search_runs([experiment.experiment_id], filter_string=f"attributes.run_name = '{name}'",
                              order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        return f"there is no run named {name}"
    client.log_artifact(runs[0].info.run_id, str(path), artifact_path="reports")
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
    path = report_path(root, round_)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(text, end="")
    uri = args.mlflow or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    try:
        why = attach(path, round_, uri)
    except Exception as exc:  # noqa: BLE001  a report that cannot be attached is still written and printed
        why = f"{type(exc).__name__}: {exc}"
    if why:
        print(f"The report was not attached to an MLflow run: {' '.join(why.split())}")
    else:
        print(f"The report was attached to the MLflow run round-{round_:02d} under reports/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
