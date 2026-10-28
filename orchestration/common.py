"""Shared paths and helpers for the orchestration scripts (run on aicr)."""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("PAIEC_ROOT", "/home/serge_ad_aihub/paiec"))
SCRATCH = Path(os.environ.get("PAIEC_SCRATCH", "/scratch/serge_ad_aihub/paiec"))
HILL = ROOT / "hill"
SUBS = ROOT / "subs"
RUNS = ROOT / "runs"
DATA = SCRATCH / "data" / "measurement-db"
CATALOG = SCRATCH / "catalog.csv"
REPO = "aims-foundations/measurement-db"
HELD_OUT = {b["benchmark"] for b in json.loads((HILL / "holdout_benchmarks.json").read_text())}

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


def hf_token() -> str:
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        p = Path.home() / ".secrets" / "hf_token"
        if p.exists():
            tok = p.read_text().strip()
            os.environ["HF_TOKEN"] = tok
    if not tok:
        raise SystemExit("HF_TOKEN missing")
    return tok


def bench_dir(name: str) -> Path:
    return DATA / name / "formatted_tables"
