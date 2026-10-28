"""Evaluator for neurips-predictive-evals-comp.

Replays the NeurIPS 2026 Predictive AI Evaluation Competition (PAIEC) protocol on
held-out benchmarks that live under private/. The climber can read this file; the
held-out labels cannot be read from anywhere but private/.

Protocol (mirrors https://aims.stanford.edu/competition):

* A *subject* is an AI model configuration; an *item* is one benchmark question;
  the label is 1 if the subject answered correctly, else 0.
* For every held-out (subject, benchmark) pair the items are split 50/50 into an
  acquisition pool and an evaluation pool. Labels are revealed from the
  acquisition pool at budgets 0, 1, 3, 7, 15, 31 (nested). Evaluation items are
  never revealed and acquisition items are never scored.
* At each budget the submission's ``predict([subject, item], labeled)`` is called
  for every evaluation item. ``labeled`` holds ``[[subject, item], label]`` for
  every revealed acquisition item across *all* sampled pairs in the run (same
  benchmark or not, same subject or not), exactly as Codabench supplies it.
* Brier score per pair, averaged equally over pairs, gives B_n for each budget.
  Brier ALC = 0.1*B0 + 0.2*B1 + 0.2*B3 + 0.2*B7 + 0.2*B15 + 0.1*B31. Lower is
  better. Expected calibration error (10 bins) is reported as a diagnostic.
* An optional ``labeling.py`` with ``acquisition_function(input, prediction=None,
  labeled=None, context=None) -> bool`` chooses which acquisition items to buy
  as the evaluator streams them; without it the order is random.

Any non-finite prediction, a value outside [0, 1], or an exception inside the
submission fails the whole report, as it would on Codabench.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

HILL = Path(__file__).resolve().parent
PRIVATE = HILL / "private"

BUDGETS = [0, 1, 3, 7, 15, 31]
WEIGHTS = {0: 0.1, 1: 0.2, 3: 0.2, 7: 0.2, 15: 0.2, 31: 0.1}
SUBJECT_FIELDS = [
    "normalized_name",
    "provider",
    "release_date",
    "access_date",
    "harness",
    "harness_version",
    "reasoning_effort",
    "subject_features_extra",
]
ITEM_FIELDS = ["item_content", "item_features", "interactors", "benchmark_id"]


def _fail(message: str, **extra) -> dict:
    return {"passed": False, "metrics": [], "config": [], "details": {"error": message, **extra}}


# --------------------------------------------------------------------------- data


def _load_holdout(split: str) -> list[dict]:
    """Rows of the held-out table for one split, as plain dicts."""
    import pyarrow.parquet as pq

    path = PRIVATE / "holdout.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; build it with tools/build_private.py")
    table = pq.read_table(path, filters=[("split", "==", split)])
    rows = table.to_pylist()
    for r in rows:
        r["subject"] = json.loads(r["subject"])
        r["item"] = {
            "item_content": r.pop("item_content"),
            "item_features": r.pop("item_features"),
            "interactors": json.loads(r["interactors"]) if r.get("interactors") else None,
            "benchmark_id": r["benchmark_id"],
        }
        r.pop("interactors", None)
    return rows


def _group_pairs(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    pairs: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        pairs.setdefault((r["subject_key"], r["benchmark_id"]), []).append(r)
    return pairs


def _pair_rng(seed: int, key: tuple[str, str]) -> random.Random:
    h = hashlib.sha256(f"{seed}|{key[0]}|{key[1]}".encode()).hexdigest()
    return random.Random(int(h[:16], 16))


# --------------------------------------------------------------------- submission


def _load_submission(submission: Path):
    """Import model.py (required) and labeling.py (optional) from the submission dir."""
    model_path = submission / "model.py"
    if not model_path.is_file():
        raise ValueError("submission must contain model.py with predict(input, labeled) -> float")

    def _import(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod

    sys.path.insert(0, str(submission))
    model = _import("paiec_submission_model", model_path)
    if not callable(getattr(model, "predict", None)):
        raise ValueError("model.py must define a callable predict(input, labeled)")
    acquire = None
    labeling_path = submission / "labeling.py"
    if labeling_path.is_file():
        labeling = _import("paiec_submission_labeling", labeling_path)
        acquire = getattr(labeling, "acquisition_function", None)
        if acquire is not None and not callable(acquire):
            raise ValueError("labeling.py defines acquisition_function but it is not callable")
    return model.predict, acquire


def _check_probability(p, where: str) -> float:
    if isinstance(p, bool) or not isinstance(p, (int, float)):
        raise ValueError(f"predict returned {p!r} ({type(p).__name__}) for {where}; a float in [0, 1] is required")
    p = float(p)
    if not math.isfinite(p) or p < 0.0 or p > 1.0:
        raise ValueError(f"predict returned {p} for {where}; a finite probability in [0, 1] is required")
    return p


# ----------------------------------------------------------------------- metrics


def _ece(preds: list[float], labels: list[int], bins: int = 10) -> float:
    if not preds:
        return float("nan")
    tot = 0.0
    n = len(preds)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(preds) if (lo <= p < hi) or (b == bins - 1 and p == 1.0)]
        if not idx:
            continue
        conf = sum(preds[i] for i in idx) / len(idx)
        acc = sum(labels[i] for i in idx) / len(idx)
        tot += len(idx) / n * abs(acc - conf)
    return tot


# -------------------------------------------------------------------------- eval


def eval(
    submission: Path,
    *,
    final: bool = False,
    seed: int = 0,
    max_eval_items: int = 100,
    max_pairs_per_benchmark: int = 12,
    time_budget_s: int = 3000,
) -> dict:
    submission = Path(submission).resolve()
    split = "final" if final else "validation"
    t0 = time.time()

    try:
        rows = _load_holdout(split)
    except Exception as e:  # noqa: BLE001
        return _fail(f"could not load held-out data: {e}")

    pairs = _group_pairs(rows)
    # Deterministic subset of pairs per benchmark so runs stay comparable and bounded.
    by_bench: dict[str, list[tuple[str, str]]] = {}
    for key in sorted(pairs):
        by_bench.setdefault(key[1], []).append(key)
    selected: list[tuple[str, str]] = []
    for bench, keys in sorted(by_bench.items()):
        rng = _pair_rng(seed, ("__bench__", bench))
        keys = sorted(keys)
        rng.shuffle(keys)
        selected.extend(keys[:max_pairs_per_benchmark])
    if not selected:
        return _fail(f"no (subject, benchmark) pairs in split {split!r}")

    # Split each pair into acquisition / evaluation pools.
    acq_pool: dict[tuple[str, str], list[dict]] = {}
    eval_pool: dict[tuple[str, str], list[dict]] = {}
    for key in selected:
        items = sorted(pairs[key], key=lambda r: r["item_id"])
        rng = _pair_rng(seed, key)
        rng.shuffle(items)
        half = len(items) // 2
        acq_pool[key] = items[:half]
        ev = items[half:]
        eval_pool[key] = ev[:max_eval_items]

    cwd = os.getcwd()
    try:
        os.chdir(submission)
        try:
            predict, acquire = _load_submission(submission)
        except Exception as e:  # noqa: BLE001
            return _fail(f"invalid submission: {e}", traceback=traceback.format_exc()[-2000:])

        # ---- acquisition: decide, per pair, the order labels are revealed in.
        max_budget = max(BUDGETS)
        acquired: dict[tuple[str, str], list[dict]] = {}
        for key in selected:
            pool = acq_pool[key]
            if acquire is None:
                acquired[key] = pool[:max_budget]
                continue
            chosen: list[dict] = []
            labeled_so_far: list = []
            try:
                for i, r in enumerate(pool):
                    remaining_labels = max_budget - len(chosen)
                    remaining_items = len(pool) - i
                    if remaining_labels <= 0:
                        break
                    inp = [r["subject"], r["item"]]
                    forced = remaining_items <= remaining_labels
                    buy = True
                    if not forced:
                        pred = predict(inp, labeled_so_far)
                        pred = _check_probability(pred, f"acquisition item {r['item_id']}")
                        buy = bool(
                            acquire(
                                inp,
                                prediction=pred,
                                labeled=labeled_so_far,
                                context={
                                    "labels_remaining": remaining_labels,
                                    "items_remaining": remaining_items,
                                    "budget": max_budget,
                                },
                            )
                        )
                    if buy:
                        chosen.append(r)
                        labeled_so_far = labeled_so_far + [[[r["subject"], r["item"]], int(r["label"])]]
            except Exception as e:  # noqa: BLE001
                return _fail(f"acquisition raised: {e}", traceback=traceback.format_exc()[-2000:])
            acquired[key] = chosen

        # ---- prediction at each budget.
        per_budget: dict[int, float] = {}
        per_budget_ece: dict[int, float] = {}
        n_calls = 0
        for n in BUDGETS:
            labeled: list = []
            for key in selected:
                for r in acquired[key][:n]:
                    labeled.append([[r["subject"], r["item"]], int(r["label"])])
            pair_briers = []
            all_p: list[float] = []
            all_y: list[int] = []
            for key in selected:
                se = 0.0
                ev = eval_pool[key]
                for r in ev:
                    if time.time() - t0 > time_budget_s:
                        return _fail(
                            f"time budget of {time_budget_s}s exceeded after {n_calls} predict calls",
                            budget_reached=n,
                        )
                    try:
                        p = predict([r["subject"], r["item"]], labeled)
                        p = _check_probability(p, f"item {r['item_id']} at budget {n}")
                    except Exception as e:  # noqa: BLE001
                        return _fail(
                            f"predict failed at budget {n}: {e}",
                            traceback=traceback.format_exc()[-2000:],
                        )
                    n_calls += 1
                    y = int(r["label"])
                    se += (p - y) ** 2
                    all_p.append(p)
                    all_y.append(y)
                pair_briers.append(se / len(ev))
            per_budget[n] = sum(pair_briers) / len(pair_briers)
            per_budget_ece[n] = _ece(all_p, all_y)
    finally:
        os.chdir(cwd)
        if str(submission) in sys.path:
            sys.path.remove(str(submission))

    alc = sum(WEIGHTS[n] * per_budget[n] for n in BUDGETS)
    n_eval_items = sum(len(v) for v in eval_pool.values())
    return {
        "passed": True,
        "metrics": [
            {"name": "brier_alc", "value": alc, "direction": "min"},
        ],
        "config": [
            {"name": "mode", "value": split, "primary": True},
            {"name": "seed", "value": seed, "primary": True},
            {"name": "max_eval_items", "value": max_eval_items, "primary": True},
            {"name": "max_pairs_per_benchmark", "value": max_pairs_per_benchmark, "primary": True},
        ],
        "details": {
            "brier_by_budget": {str(n): round(per_budget[n], 5) for n in BUDGETS},
            "ece_by_budget": {str(n): round(per_budget_ece[n], 5) for n in BUDGETS},
            "n_pairs": len(selected),
            "n_benchmarks": len(by_bench),
            "n_eval_items": n_eval_items,
            "n_predict_calls": n_calls,
            "used_labeling_py": acquire is not None,
            "constant_0.5_reference": 0.25,
            "seconds": round(time.time() - t0, 1),
        },
    }
