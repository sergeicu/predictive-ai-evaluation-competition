"""Aggregate runs/*/*.json into a markdown table (stdout) and RESULTS.md.

    python results.py [--filter hill_validation] [--out RESULTS.md]

One row per (submission, tag, split): mean brier_alc over seeds, per-budget means,
number of seeds. Sorted by tag then score.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from common import RUNS

PAT = re.compile(r"^(?P<tag>[^_]+)_(?P<split>validation|final)_s(?P<seed>\d+)_e(?P<e>\d+)_p(?P<p>\d+)\.json$")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    groups: dict[tuple, list] = defaultdict(list)
    for sub in sorted(RUNS.iterdir()) if RUNS.exists() else []:
        for f in sub.glob("*.json"):
            m = PAT.match(f.name)
            if not m:
                continue
            rep = json.loads(f.read_text())
            if not rep.get("passed"):
                groups[(sub.name, m["tag"], m["split"], m["e"], m["p"])].append(None)
                continue
            groups[(sub.name, m["tag"], m["split"], m["e"], m["p"])].append(rep)
    rows = []
    for (sub, tag, split, e, p), reps in groups.items():
        ok = [r for r in reps if r]
        if f"{tag}_{split}" and a.filter and a.filter not in f"{tag}_{split}":
            continue
        if not ok:
            rows.append((tag, split, 9, sub, "FAILED", "", len(reps), e, p))
            continue
        alc = sum(r["metrics"][0]["value"] for r in ok) / len(ok)
        budgets = ok[0]["details"]["brier_by_budget"].keys()
        curve = " ".join(
            f"{b}:{sum(r['details']['brier_by_budget'][b] for r in ok) / len(ok):.4f}" for b in budgets
        )
        rows.append((tag, split, alc, sub, f"{alc:.5f}", curve, len(ok), e, p))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    lines = ["| holdout | split | submission | brier_alc (mean) | per budget (B0 B1 B3 B7 B15 B31) | seeds | items/pairs |", "|---|---|---|---|---|---|---|"]
    for tag, split, _, sub, alc, curve, n, e, p in rows:
        lines.append(f"| {tag} | {split} | {sub} | {alc} | {curve} | {n} | {e}/{p} |")
    text = "\n".join(lines)
    print(text)
    if a.out:
        Path(a.out).write_text("# Results\n\nLower is better. Constant 0.5 = 0.25.\n\n" + text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
