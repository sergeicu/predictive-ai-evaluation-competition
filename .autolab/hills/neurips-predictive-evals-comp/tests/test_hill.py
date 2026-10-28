"""Checks on the hill itself, run by `hills check`.

A known-good submission scores sensibly, degenerate submissions score where the
math says they must, and broken ones are rejected with a readable reason.
"""

from pathlib import Path

from eval import BUDGETS, WEIGHTS, eval

HILL = Path(__file__).resolve().parents[1]
BASELINE = HILL / "examples" / "baseline"

SMALL = dict(max_eval_items=20, max_pairs_per_benchmark=3)


def _write(tmp_path: Path, body: str) -> Path:
    (tmp_path / "model.py").write_text(body)
    return tmp_path


def test_weights_sum_to_one():
    assert abs(sum(WEIGHTS[b] for b in BUDGETS) - 1.0) < 1e-9


def test_baseline_beats_constant_on_both_splits():
    for final in (False, True):
        report = eval(BASELINE, final=final, **SMALL)
        assert report["passed"], report["details"]
        assert report["metrics"][0]["name"] == "brier_alc"
        alc = report["metrics"][0]["value"]
        assert 0.0 < alc < 0.25, report["details"]
        b = report["details"]["brier_by_budget"]
        assert b["0"] == 0.25  # nothing revealed -> 0.5 everywhere
        assert b["31"] < b["0"]


def test_constant_half_scores_exactly_quarter(tmp_path):
    sub = _write(tmp_path, "def predict(input, labeled=None):\n    return 0.5\n")
    report = eval(sub, **SMALL)
    assert report["passed"], report["details"]
    assert abs(report["metrics"][0]["value"] - 0.25) < 1e-12


def test_missing_model_is_rejected(tmp_path):
    report = eval(tmp_path, **SMALL)
    assert not report["passed"]
    assert "model.py" in report["details"]["error"]


def test_out_of_range_probability_is_rejected(tmp_path):
    sub = _write(tmp_path, "def predict(input, labeled=None):\n    return 1.5\n")
    report = eval(sub, **SMALL)
    assert not report["passed"]
    assert "[0, 1]" in report["details"]["error"]


def test_nan_is_rejected(tmp_path):
    sub = _write(tmp_path, "def predict(input, labeled=None):\n    return float('nan')\n")
    report = eval(sub, **SMALL)
    assert not report["passed"]


def test_exception_is_rejected(tmp_path):
    sub = _write(tmp_path, "def predict(input, labeled=None):\n    raise RuntimeError('boom')\n")
    report = eval(sub, **SMALL)
    assert not report["passed"]
    assert "boom" in report["details"]["error"]


def test_labeling_py_is_used(tmp_path):
    _write(tmp_path, (BASELINE / "model.py").read_text())
    (tmp_path / "labeling.py").write_text(
        "def acquisition_function(input, prediction=None, labeled=None, context=None):\n"
        "    return abs(prediction - 0.5) < 0.2\n"
    )
    report = eval(tmp_path, **SMALL)
    assert report["passed"], report["details"]
    assert report["details"]["used_labeling_py"] is True


def test_seed_changes_sample_but_not_contract():
    a = eval(BASELINE, seed=0, **SMALL)
    b = eval(BASELINE, seed=1, **SMALL)
    assert a["passed"] and b["passed"]
    assert a["details"]["n_pairs"] == b["details"]["n_pairs"]
