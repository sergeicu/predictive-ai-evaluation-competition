"""Estimate omega2, the variance of item difficulties (logit scale) within a benchmark
after removing subject effects, from local responses tables.

    python fit_items.py --benchmarks a,b,c  [--out omega2.json]

Per benchmark: one response per (subject, item); subjects with >= 20 items; items with
>= 5 subjects. Alternate a_s and delta_i (ridge 1.0, 40 Newton rounds), then
omega2 = var(delta) - mean sampling variance. Prints the median over benchmarks.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import HELD_OUT, bench_dir


def omega_for(name: str):
    d = bench_dir(name)
    rp = d / "responses.parquet"
    if not rp.exists():
        rp = d / "response.parquet"
    if not rp.exists():
        return None
    r = pd.read_parquet(rp, columns=["subject_id", "item_id", "trial", "response"])
    r = r[r["response"].isin([0, 1, 0.0, 1.0])]
    r = r.sort_values(["subject_id", "item_id", "trial"]).drop_duplicates(["subject_id", "item_id"])
    sc = r.groupby("subject_id").size(); r = r[r["subject_id"].isin(sc[sc >= 20].index)]
    ic = r.groupby("item_id").size(); r = r[r["item_id"].isin(ic[ic >= 5].index)]
    if r["item_id"].nunique() < 30 or r["subject_id"].nunique() < 5:
        return None
    y = r["response"].to_numpy(float)
    s_idx, s_keys = pd.factorize(r["subject_id"]); i_idx, i_keys = pd.factorize(r["item_id"])
    a = np.zeros(len(s_keys)); dl = np.zeros(len(i_keys))
    for _ in range(40):
        for tgt, idx, size in ((a, s_idx, len(s_keys)), (dl, i_idx, len(i_keys))):
            p = 1 / (1 + np.exp(-(a[s_idx] + dl[i_idx])))
            g = np.bincount(idx, weights=y - p, minlength=size) - 1.0 * tgt
            h = np.bincount(idx, weights=p * (1 - p), minlength=size) + 1.0
            tgt += np.clip(g / h, -1, 1)
    p = 1 / (1 + np.exp(-(a[s_idx] + dl[i_idx])))
    info = np.bincount(i_idx, weights=p * (1 - p), minlength=len(i_keys)) + 1.0
    omega2 = float(max(np.var(dl) - np.mean(1 / info), 0.0))
    return dict(benchmark=name, n_items=len(i_keys), n_subjects=len(s_keys), omega2=omega2, raw_var=float(np.var(dl)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--benchmarks", required=True); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    names = [b for b in a.benchmarks.split(",") if b and b not in HELD_OUT]
    rows = [x for x in (omega_for(n) for n in names) if x]
    for x in rows:
        print(f"{x['benchmark']}: items={x['n_items']} subjects={x['n_subjects']} omega2={x['omega2']:.3f}", file=sys.stderr)
    med = float(np.median([x["omega2"] for x in rows])) if rows else 0.0
    print(f"median omega2 = {med:.3f} over {len(rows)} benchmarks", file=sys.stderr)
    if a.out:
        Path(a.out).write_text(json.dumps({"median": med, "benchmarks": rows}, indent=1))
    print(f"{med:.4f}")


if __name__ == "__main__":
    main()
