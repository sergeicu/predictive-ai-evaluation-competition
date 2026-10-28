"""Build catalog.csv: one row per measurement-db benchmark with its type, granularity,
release, domain, counts (whatever the benchmarks table carries) and the on-Hub size of
each formatted table. Reads ~250 tiny parquet files; no bulk data is downloaded.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from huggingface_hub import HfFileSystem

from common import CATALOG, HELD_OUT, REPO, hf_token


def main() -> int:
    hf_token()
    fs = HfFileSystem()
    root = f"datasets/{REPO}"
    # dataset_info's sibling list is truncated for this repo, so discover via the filesystem
    benches = sorted(e["name"].rsplit("/", 1)[-1] for e in fs.ls(root, detail=True) if e["type"] == "directory" and not e["name"].rsplit("/", 1)[-1].startswith("."))
    sizes: dict[tuple[str, str], int] = {}

    def size_one(name: str):
        try:
            for e in fs.ls(f"{root}/{name}/formatted_tables", detail=True):
                fn = e["name"].rsplit("/", 1)[-1]
                if fn.endswith(".parquet"):
                    sizes[(name, fn[:-8])] = int(e.get("size") or 0)
        except Exception:  # noqa: BLE001
            pass

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(size_one, benches))
    print(f"{len(benches)} top-level dirs, {sum(1 for b in benches if (b, 'benchmarks') in sizes)} with formatted tables", file=sys.stderr)

    def one(name: str) -> dict:
        row = {"benchmark": name}
        try:
            df = pd.read_parquet(fs.open(f"datasets/{REPO}/{name}/formatted_tables/benchmarks.parquet", "rb"))
            r = df.iloc[0]
            for c in df.columns:
                v = r[c]
                if hasattr(v, "tolist"):
                    v = v.tolist()
                row["b_" + str(c)] = str(v)[:200] if v is not None else None
        except Exception as e:  # noqa: BLE001
            row["error"] = str(e)[:200]
        for t in ("benchmarks", "subjects", "items", "responses", "response"):
            row[f"size_{t}"] = sizes.get((name, t), 0)
        row["held_out"] = name in HELD_OUT
        return row

    with ThreadPoolExecutor(16) as ex:
        rows = list(ex.map(one, [b for b in benches if (b, 'benchmarks') in sizes]))
    cat = pd.DataFrame(rows)
    CATALOG.parent.mkdir(parents=True, exist_ok=True)
    cat.to_csv(CATALOG, index=False)
    print(f"wrote {CATALOG} with {len(cat)} benchmarks", file=sys.stderr)
    cols = [c for c in cat.columns if c.startswith("b_")]
    print("benchmarks-table columns:", cols, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
