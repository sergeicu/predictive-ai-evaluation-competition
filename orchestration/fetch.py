"""Download selected formatted tables for a list of benchmarks into DATA.

    python fetch.py --benchmarks a,b,c [--tables subjects,responses] [--workers 8]
    python fetch.py --list stage1.txt

Never fetches held-out benchmarks. `responses` falls back to the older `response`
file name when needed. Idempotent: existing files are skipped.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

from common import DATA, HELD_OUT, REPO, bench_dir, hf_token


def fetch_one(name: str, tables: list[str], files: set[str] | None = None) -> str:
    out = bench_dir(name)
    out.mkdir(parents=True, exist_ok=True)
    got = []
    for t in tables:
        cands = [t] if t != "responses" else ["responses", "response"]
        for c in cands:
            dst = out / f"{c}.parquet"
            if dst.exists() and dst.stat().st_size > 0:
                got.append(c)
                break
            try:
                hf_hub_download(REPO, f"{name}/formatted_tables/{c}.parquet", repo_type="dataset", local_dir=DATA)
                got.append(c)
                break
            except EntryNotFoundError:
                continue
    missing = [t for t in tables if not any(g.startswith(t[:8]) for g in got)]
    return f"{name}: {','.join(got)}" + (f" MISSING {missing}" if missing else "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", default="")
    ap.add_argument("--list", default=None)
    ap.add_argument("--tables", default="subjects,responses,benchmarks")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    names = [b.strip() for b in a.benchmarks.split(",") if b.strip()]
    if a.list:
        names += [l.strip() for l in Path(a.list).read_text().splitlines() if l.strip() and not l.startswith("#")]
    bad = [n for n in names if n in HELD_OUT]
    if bad:
        raise SystemExit(f"refusing to fetch held-out benchmarks: {bad}")
    hf_token()
    files = None
    tables = [t.strip() for t in a.tables.split(",") if t.strip()]
    with ThreadPoolExecutor(a.workers) as ex:
        for msg in ex.map(lambda n: fetch_one(n, tables, files), names):
            print(msg, file=sys.stderr)
    total = sum(p.stat().st_size for p in DATA.rglob("*.parquet"))
    print(f"data dir now {total / 1e9:.2f} GB", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
