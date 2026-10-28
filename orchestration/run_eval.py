"""Score a submission with the hill's own eval.py and store the report.

    python run_eval.py --sub M0_pooled_beta --split validation --seeds 0,1,2
    python run_eval.py --sub M2_hier --holdout /path/dev_holdout.parquet --split validation

`--holdout` points eval.py at a different private table (the dev simulator); reports
then go under runs/<sub>/dev_*. Prints one summary line per report.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from common import HILL, RUNS, SUBS


def load_eval(holdout: str | None):
    spec = importlib.util.spec_from_file_location("hill_eval", HILL / "eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if holdout:
        h = Path(holdout).resolve()
        mod.PRIVATE = h.parent
        if h.name != "holdout.parquet":
            raise SystemExit("--holdout must point at a file named holdout.parquet")
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", required=True, help="name under subs/ or a path")
    ap.add_argument("--split", default="validation", choices=["validation", "final"])
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--max-eval-items", type=int, default=100)
    ap.add_argument("--max-pairs", type=int, default=12)
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--tag", default=None, help="label for the holdout (default: hill)")
    a = ap.parse_args()

    sub = Path(a.sub) if Path(a.sub).exists() else SUBS / a.sub
    if not (sub / "model.py").exists():
        raise SystemExit(f"no model.py in {sub}")
    mod = load_eval(a.holdout)
    tag = a.tag or ("hill" if not a.holdout else Path(a.holdout).parent.name)
    out_dir = RUNS / sub.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed in [int(s) for s in a.seeds.split(",")]:
        rep = mod.eval(
            sub,
            final=(a.split == "final"),
            seed=seed,
            max_eval_items=a.max_eval_items,
            max_pairs_per_benchmark=a.max_pairs,
        )
        name = f"{tag}_{a.split}_s{seed}_e{a.max_eval_items}_p{a.max_pairs}.json"
        (out_dir / name).write_text(json.dumps(rep, indent=1, default=str))
        d = rep.get("details", {})
        if rep.get("passed"):
            bb = d.get("brier_by_budget", {})
            curve = " ".join(f"B{k}={v:.4f}" for k, v in bb.items())
            print(
                f"{sub.name} {tag} {a.split} seed={seed} brier_alc={rep['metrics'][0]['value']:.5f} "
                f"| {curve} | pairs={d.get('n_pairs')} calls={d.get('n_predict_calls')} {d.get('seconds', 0):.1f}s"
            )
        else:
            print(f"{sub.name} {tag} {a.split} seed={seed} FAILED: {d.get('error')}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
