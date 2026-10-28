"""Build a dev holdout table (same schema as the hill's private/holdout.parquet) from
local measurement-db tables, so eval.py can score leave-benchmarks-out on many more
benchmarks than the 4 in the hill's validation split.

    python build_dev_holdout.py --benchmarks a,b,c,d --out /scratch/.../dev/devA/holdout.parquet \
        [--min-items 80] [--max-subjects 40] [--max-items 400]

All listed benchmarks go to split "validation". item_content is left empty (no items
table is needed; the models under test do not read text). The salted anonymisation
mirrors tools/build_private.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from common import HELD_OUT, SUBJECT_FIELDS, bench_dir


def _h(salt, *parts):
    return hashlib.sha256("|".join([salt, *parts]).encode()).hexdigest()[:12]


def _clean(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-items", type=int, default=80)
    ap.add_argument("--max-subjects", type=int, default=40)
    ap.add_argument("--max-items", type=int, default=400)
    ap.add_argument("--salt", default="dev-salt")
    a = ap.parse_args()
    names = [b.strip() for b in a.benchmarks.split(",") if b.strip()]
    if set(names) & HELD_OUT:
        raise SystemExit("dev holdout must not contain the hill's held-out benchmarks")
    rows = []
    for name in names:
        d = bench_dir(name)
        rp = d / "responses.parquet"
        if not rp.exists():
            rp = d / "response.parquet"
        resp = pd.read_parquet(rp, columns=["subject_id", "item_id", "trial", "response"])
        resp = resp[resp["response"].isin([0, 1, 0.0, 1.0])].copy()
        resp["response"] = resp["response"].astype(int)
        resp = resp.sort_values(["subject_id", "item_id", "trial"]).drop_duplicates(["subject_id", "item_id"])
        items = sorted(resp["item_id"].unique())
        if len(items) > a.max_items:
            rng = random.Random(int(_h(a.salt, "items", name), 16))
            resp = resp[resp["item_id"].isin(set(rng.sample(items, a.max_items)))]
        counts = resp.groupby("subject_id").size()
        subj = pd.read_parquet(d / "subjects.parquet")
        subj = subj[subj["subject_id"].isin(counts[counts >= a.min_items].index)].copy()
        subj["_rd"] = pd.to_datetime(subj["release_date"], errors="coerce")
        subj = subj.sort_values("_rd", ascending=False).head(a.max_subjects)
        resp = resp[resp["subject_id"].isin(subj["subject_id"])]
        sd = {r["subject_id"]: {k: _clean(r.get(k)) for k in SUBJECT_FIELDS} for _, r in subj.iterrows()}
        anon = "b_" + _h(a.salt, "bench", name)
        for r in resp.itertuples(index=False):
            s = sd[r.subject_id]
            sj = json.dumps(s, sort_keys=True)
            rows.append(
                {
                    "split": "validation",
                    "benchmark_id": anon,
                    "subject_key": _h(a.salt, "subject", sj),
                    "subject": sj,
                    "item_id": "i_" + _h(a.salt, "item", name, str(r.item_id)),
                    # no items table is fetched; the anonymised id stands in for the text so
                    # item identity is still available to models (the hill has real text)
                    "item_content": "i_" + _h(a.salt, "item", name, str(r.item_id)),
                    "item_features": None,
                    "interactors": None,
                    "label": int(r.response),
                }
            )
        print(f"{name}: subjects={len(subj)} rows={int((resp.shape[0]))}", file=sys.stderr)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out, compression="zstd")
    (out.parent / "benchmarks.json").write_text(json.dumps(names))
    print(f"wrote {out} ({len(rows)} rows, {len(names)} benchmarks)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
