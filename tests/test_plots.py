"""Stage 6 plots: every figure comes from a file; missing inputs skip, not fail."""

from pathlib import Path

from scripts.plots import main

FIXTURES = Path(__file__).parent / "fixtures"


def _run(out: Path, root: Path = FIXTURES / "plot-loop") -> list[str]:
    return main(
        [
            "--root",
            str(root),
            "--baseline-root",
            str(FIXTURES / "plot-baseline"),
            "--calibration",
            str(FIXTURES / "plot-calibration" / "pairs.jsonl"),
            "--final-eval",
            str(FIXTURES / "plot-final.json"),
            "--out",
            str(out),
        ]
    )


def test_all_four_plots_are_written(tmp_path):
    assert _run(tmp_path) == 0
    for name in ("quality.png", "rounds.png", "calibration.png", "baselines.png"):
        png = tmp_path / name
        assert png.exists() and png.stat().st_size > 1000, name


def test_missing_inputs_skip_without_failing(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    out = tmp_path / "out"
    assert (
        main(
            [
                "--root",
                str(empty),
                "--baseline-root",
                str(empty),
                "--calibration",
                str(empty / "none.jsonl"),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert not list(out.glob("*.png"))


def test_baselines_without_final_eval(tmp_path):
    out = tmp_path / "out"
    assert (
        main(
            [
                "--root",
                str(FIXTURES / "plot-loop"),
                "--baseline-root",
                str(FIXTURES / "plot-baseline"),
                "--calibration",
                str(FIXTURES / "plot-calibration" / "pairs.jsonl"),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert (out / "baselines.png").exists()


def test_pairs_file_with_only_bad_lines_is_skipped(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"unrelated": true}\nnot json at all\n')
    out = tmp_path / "out"
    assert (
        main(
            [
                "--root",
                str(FIXTURES / "plot-loop"),
                "--baseline-root",
                str(FIXTURES / "plot-baseline"),
                "--calibration",
                str(bad),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert not (out / "calibration.png").exists()
