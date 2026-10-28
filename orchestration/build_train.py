"""Aggregate local measurement-db tables into one pair-level training table.

    python build_train.py --out /scratch/.../train_pairs.parquet [--only list.txt] [--exclude a,b]

For every benchmark under DATA with subjects + responses tables: keep binary
item-level responses, one response per (subject, item) (first trial), and emit one
row per (benchmark, subject) with k = #correct, n = #items plus the subject fields.
Held-out benchmarks are always excluded; `--exclude` removes dev-holdout benchmarks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from common import DATA, HELD_OUT, SUBJECT_FIELDS, bench_dir


def load_pairs(name: str) -> pd.DataFrame | None:
    d = bench_dir(name)
    rp = d / "responses.parquet"
    if not rp.exists():
        rp = d / "response.parquet"
    sp = d / "subjects.parquet"
    if not (rp.exists() and sp.exists()):
        return None
    resp = pd.read_parquet(rp, columns=["subject_id", "item_id", "trial", "response"])
    resp = resp[resp["response"].isin([0, 1, 0.0, 1.0])]
    if resp.empty:
        return None
    resp = resp.sort_values(["subject_id", "item_id", "trial"]).drop_duplicates(["subject_id", "item_id"])
    agg = resp.groupby("subject_id")["response"].agg(k="sum", n="size").reset_index()
    subj = pd.read_parquet(sp)
    cols = ["subject_id"] + [c for c in SUBJECT_FIELDS if c in subj.columns]
    out = agg.merge(subj[cols], on="subject_id", how="left")
    out["benchmark"] = name
    out["n_items_bench"] = resp["item_id"].nunique()
    for c in SUBJECT_FIELDS:
        if c not in out.columns:
            out[c] = None
        out[c] = out[c].map(lambda v: None if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="file listing benchmarks to include")
    ap.add_argument("--exclude", default="", help="comma-separated benchmarks to drop (dev holdout)")
    a = ap.parse_args()
    names = sorted(p.name for p in DATA.iterdir() if p.is_dir()) if DATA.exists() else []
    if a.only:
        keep = {l.strip() for l in Path(a.only).read_text().splitlines() if l.strip() and not l.startswith("#")}
        names = [n for n in names if n in keep]
    excl = {b.strip() for b in a.exclude.split(",") if b.strip()} | HELD_OUT
    names = [n for n in names if n not in excl]
    frames = []
    for n in names:
        df = load_pairs(n)
        if df is None:
            print(f"{n}: skipped (missing tables or no binary responses)", file=sys.stderr)
            continue
        frames.append(df)
        print(f"{n}: subjects={len(df)} items={int(df['n_items_bench'].iloc[0])} responses={int(df['n'].sum())}", file=sys.stderr)
    if not frames:
        raise SystemExit("no training data found")
    train = pd.concat(frames, ignore_index=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    train.to_parquet(a.out, index=False)
    print(
        f"wrote {a.out}: {len(train)} pairs, {train['benchmark'].nunique()} benchmarks, "
        f"{train['normalized_name'].nunique()} subject names, {int(train['n'].sum())} responses",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
