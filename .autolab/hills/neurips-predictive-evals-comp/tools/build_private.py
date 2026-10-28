"""Build private/holdout.parquet from public measurement-db benchmarks.

Usage (from the hill directory, with HF_TOKEN set and access to the gated
dataset https://huggingface.co/datasets/aims-foundations/measurement-db):

    python tools/build_private.py --benchmarks researchcodebench,real_webagents,... \
        [--min-items 80] [--max-subjects 40] [--salt <secret>]

For each benchmark it streams the formatted tables straight from the Hub (nothing
is cached on disk except the output), keeps subjects with at least --min-items
binary item-level responses, prefers the most recently released subjects, assigns
benchmarks alternately to the "validation" and "final" splits, anonymises the
benchmark and item ids with a salted hash, and writes one parquet table plus
holdout_benchmarks.json (public: which benchmarks climbers must exclude from
training) and private/benchmark_map.json (the anon-id mapping).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

HILL = Path(__file__).resolve().parents[1]
REPO = "datasets/aims-foundations/measurement-db/"
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


def _clean(v):
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    if hasattr(v, "tolist"):
        return v.tolist()
    return v


def _h(salt: str, *parts: str) -> str:
    return hashlib.sha256(("|".join([salt, *parts])).encode()).hexdigest()[:12]


def load_benchmark(fs: HfFileSystem, name: str, revision: str | None):
    base = REPO + name + "/formatted_tables/"
    rev = f"@{revision}" if revision else ""
    base = REPO.rstrip("/") + rev + "/" + name + "/formatted_tables/"
    rd = lambda t: pd.read_parquet(fs.open(base + t + ".parquet", "rb"))  # noqa: E731
    bench = rd("benchmarks").iloc[0]
    if not (bench["response_type"] == "binary" and bench["granularity"] == "item"):
        raise SystemExit(f"{name}: not binary item-level ({bench['response_type']}, {bench['granularity']})")
    subj = rd("subjects")
    items = rd("items")
    resp_name = "responses" if fs.exists(base + "responses.parquet") else "response"
    resp = rd(resp_name)
    return bench, subj, items, resp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", required=True, help="comma-separated measurement-db directory names")
    ap.add_argument("--min-items", type=int, default=80)
    ap.add_argument("--max-subjects", type=int, default=40)
    ap.add_argument(
        "--max-items",
        type=int,
        default=400,
        help="per benchmark, keep a fixed random subset of this many items (shared by all subjects)",
    )
    ap.add_argument("--revision", default=None, help="pin a measurement-db git revision")
    ap.add_argument("--salt", default=None, help="secret salt for anonymised ids (random if omitted)")
    ap.add_argument("--out", default=str(HILL / "private" / "holdout.parquet"))
    args = ap.parse_args()

    if not os.environ.get("HF_TOKEN"):
        print("HF_TOKEN is not set; the dataset is gated", file=sys.stderr)
        return 2
    salt = args.salt or secrets.token_hex(16)
    fs = HfFileSystem()
    names = [b.strip() for b in args.benchmarks.split(",") if b.strip()]

    out_rows: list[dict] = []
    bench_map: dict[str, dict] = {}
    public_list: list[dict] = []
    for i, name in enumerate(names):
        split = "validation" if i % 2 == 0 else "final"
        bench, subj, items, resp = load_benchmark(fs, name, args.revision)
        resp = resp[resp["response"].isin([0, 1, 0.0, 1.0])].copy()
        resp["response"] = resp["response"].astype(int)
        # one response per (subject, item): first trial wins
        resp = resp.sort_values(["subject_id", "item_id", "trial"]).drop_duplicates(["subject_id", "item_id"])
        all_items = sorted(resp["item_id"].unique())
        if len(all_items) > args.max_items:
            import random

            rng = random.Random(int(_h(salt, "items", name), 16))
            keep_items = set(rng.sample(all_items, args.max_items))
            resp = resp[resp["item_id"].isin(keep_items)]
        counts = resp.groupby("subject_id").size()
        keep = counts[counts >= args.min_items].index
        subj = subj[subj["subject_id"].isin(keep)].copy()
        subj["_rd"] = pd.to_datetime(subj["release_date"], errors="coerce")
        subj = subj.sort_values("_rd", ascending=False).head(args.max_subjects)
        resp = resp[resp["subject_id"].isin(subj["subject_id"])]
        items = items.set_index("item_id")
        anon_bench = "b_" + _h(salt, "bench", name)
        bench_map[anon_bench] = {"benchmark": name, "split": split}
        public_list.append(
            {
                "benchmark": name,
                "split": split,
                "n_subjects": int(subj.shape[0]),
                "n_items": int(resp["item_id"].nunique()),
                "n_responses": int(resp.shape[0]),
                "domain": _clean(bench.get("domain")),
            }
        )
        subj_dicts = {
            r["subject_id"]: {k: _clean(r.get(k)) for k in SUBJECT_FIELDS} for _, r in subj.iterrows()
        }
        for r in resp.itertuples(index=False):
            it = items.loc[r.item_id]
            sd = subj_dicts[r.subject_id]
            out_rows.append(
                {
                    "split": split,
                    "benchmark_id": anon_bench,
                    "subject_key": _h(salt, "subject", json.dumps(sd, sort_keys=True, default=str)),
                    "subject": json.dumps(sd, sort_keys=True, default=str),
                    "item_id": "i_" + _h(salt, "item", name, str(r.item_id)),
                    "item_content": None if it["content"] is None else str(it["content"]),
                    "item_features": _clean(it.get("item_features")),
                    "interactors": json.dumps(_clean(getattr(r, "interactors", None)), default=str)
                    if _clean(getattr(r, "interactors", None)) is not None
                    else None,
                    "label": int(r.response),
                }
            )
        print(f"{name}: split={split} subjects={subj.shape[0]} responses={resp.shape[0]}", file=sys.stderr)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(out_rows)
    pq.write_table(table, out, compression="zstd")
    (out.parent / "benchmark_map.json").write_text(json.dumps(bench_map, indent=1))
    (HILL / "holdout_benchmarks.json").write_text(json.dumps(public_list, indent=1))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(out_rows)} rows)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
