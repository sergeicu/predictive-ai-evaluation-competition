"""Diagnostic: how many held-out subjects (names only, no labels) have a match in the
training table, exact key (name|effort) and by name. Never touches labels."""
import argparse, json
import pandas as pd
from common import HILL


def norm(v):
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() in ("nan", "none", "null") else s


ap = argparse.ArgumentParser(); ap.add_argument("--train", required=True); ap.add_argument("--holdout", default=str(HILL / "private" / "holdout.parquet"))
a = ap.parse_args()
tr = pd.read_parquet(a.train)
names = {norm(x) for x in tr["normalized_name"]}
keys = {f"{norm(n)}|{norm(e)}" for n, e in zip(tr["normalized_name"], tr["reasoning_effort"])}
ho = pd.read_parquet(a.holdout, columns=["split", "benchmark_id", "subject"]).drop_duplicates()
ho["s"] = ho["subject"].map(json.loads)
for split, g in ho.groupby("split"):
    subs = g["s"].tolist()
    exact = sum(f"{norm(s['normalized_name'])}|{norm(s['reasoning_effort'])}" in keys for s in subs)
    byname = sum(norm(s["normalized_name"]) in names for s in subs)
    miss = sorted({norm(s["normalized_name"]) for s in subs if norm(s["normalized_name"]) not in names})
    print(f"{split}: {len(subs)} subject-benchmark pairs; exact key match {exact}, name match {byname}; unmatched names ({len(miss)}): {miss[:30]}")
