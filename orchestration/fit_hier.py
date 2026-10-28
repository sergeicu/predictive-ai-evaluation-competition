"""Fit the hierarchical logit model on train_pairs.parquet and write a submission dir.

    python fit_hier.py --train /scratch/.../train_pairs.parquet --name M2_hier_stage1 \
        [--mode M1|M2] [--exclude dev1,dev2] [--sigma-mult 1.0] [--tau-mult 1.0] [--temp 1.0]
        [--min-n 5] [--ridge-a 1.0] [--ridge-d 1.0]

Model: k_sb ~ Binomial(n_sb, sigmoid(a_s + d_b)); ridge priors on a and d; fitted by
alternating Newton steps. sigma2 is the extra pair-level variance left after a_s + d_b.
The output dir gets models/hier/model.py + params.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import HELD_OUT, SUBS

TEMPLATE = Path(__file__).resolve().parent / "models" / "hier" / "model.py"


def _norm(v) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() in ("nan", "none", "null") else s


def skey(row) -> str:
    return f"{_norm(row['normalized_name'])}|{_norm(row['reasoning_effort'])}"


def year(s):
    s = str(s) if s is not None else ""
    return s[:4] if s[:4].isdigit() else None


def fit(df: pd.DataFrame, ridge_a: float, ridge_d: float, iters: int = 60, a_mult: float = 1.0):
    s_idx, s_keys = pd.factorize(df["skey"])
    b_idx, b_keys = pd.factorize(df["benchmark"])
    k = df["k"].to_numpy(float)
    n = df["n"].to_numpy(float)
    a = np.zeros(len(s_keys))
    d = np.zeros(len(b_keys))
    # start d at the benchmark mean logit, a at 0
    for _ in range(iters):
        for target, idx, size, ridge in ((a, s_idx, len(s_keys), ridge_a), (d, b_idx, len(b_keys), ridge_d)):
            p = 1 / (1 + np.exp(-(a[s_idx] + d[b_idx])))
            g = np.bincount(idx, weights=k - n * p, minlength=size) - ridge * target
            h = np.bincount(idx, weights=n * p * (1 - p), minlength=size) + ridge
            target += np.clip(g / h, -1.0, 1.0)
    p = 1 / (1 + np.exp(-(a[s_idx] + d[b_idx])))
    info_a = np.bincount(s_idx, weights=n * p * (1 - p), minlength=len(s_keys)) + ridge_a
    v_a = 1.0 / info_a
    # residual pair variance beyond a_s + d_b, corrected for binomial sampling noise
    m = n >= 20
    emp = np.log((k[m] + 0.5) / (n[m] - k[m] + 0.5))
    a_eff = a.mean() + a_mult * (a - a.mean())
    r = emp - (a_eff[s_idx[m]] + d[b_idx[m]])
    r = r - r.mean()
    noise = np.mean(1.0 / (n[m] * p[m] * (1 - p[m])))
    sigma2 = float(max(np.var(r) - noise, 0.05))
    return s_keys, a, v_a, b_keys, d, sigma2, dict(resid_var=float(np.var(r)), noise=float(noise), n_pairs_resid=int(m.sum()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", default="M2", choices=["M1", "M2", "M3"])
    ap.add_argument("--omega2", type=float, default=0.0, help="item-difficulty variance (logit^2) for M3; see fit_items.py")
    ap.add_argument("--omega-mult", type=float, default=1.0)
    ap.add_argument("--b0-half", action="store_true", help="predict exactly 0.5 when no label on the benchmark exists")
    ap.add_argument("--exclude", default="")
    ap.add_argument("--min-n", type=int, default=5)
    ap.add_argument("--ridge-a", type=float, default=1.0)
    ap.add_argument("--ridge-d", type=float, default=1.0)
    ap.add_argument("--sigma-mult", type=float, default=1.0)
    ap.add_argument("--tau-mult", type=float, default=1.0)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--a-mult", type=float, default=1.0, help="transfer coefficient on subject ability (0 = ignore subjects)")
    ap.add_argument("--out-root", default=str(SUBS))
    a = ap.parse_args()

    df = pd.read_parquet(a.train)
    excl = {b.strip() for b in a.exclude.split(",") if b.strip()} | HELD_OUT
    df = df[~df["benchmark"].isin(excl) & (df["n"] >= a.min_n)].copy()
    df["skey"] = df.apply(skey, axis=1)
    s_keys, ab, v_a, b_keys, d, sigma2, diag = fit(df, a.ridge_a, a.ridge_d, a_mult=a.a_mult)

    subjects = {str(kk): [float(x), float(v)] for kk, x, v in zip(s_keys, ab, v_a)}
    # fallbacks: name, provider|year, provider, global (mean and between-subject variance)
    meta = df.drop_duplicates("skey").set_index("skey")
    sa = pd.DataFrame({"a": ab, "v": v_a}, index=s_keys)
    sa["name"] = [_norm(x) for x in meta.loc[sa.index, "normalized_name"].values]
    sa["provider"] = [_norm(x) for x in meta.loc[sa.index, "provider"].values]
    sa["py"] = [f"{p}|{year(r)}" for p, r in zip(sa["provider"], meta.loc[sa.index, "release_date"].values)]
    gvar = float(np.var(ab)) if len(ab) > 1 else 1.0

    def grp(col):
        # mean ability of the group; variance = spread between members + their own uncertainty
        out = {}
        for key, g in sa.groupby(col):
            between = float(g["a"].var()) if len(g) > 1 else (gvar if col != "name" else 0.0)
            out[str(key)] = [float(g["a"].mean()), max(between + float(g["v"].mean()), 0.02)]
        return out

    params = {
        "mode": a.mode,
        "sigma2": sigma2,
        "sigma_mult": a.sigma_mult,
        "tau_mult": a.tau_mult,
        "temp": a.temp,
        "a_mult": a.a_mult,
        "omega2": a.omega2,
        "omega_mult": a.omega_mult,
        "b0_half": bool(a.b0_half),
        "mean_a": float(np.mean(ab)),
        "mu_d": float(np.mean(d)),
        "tau_d2": float(max(np.var(d), 0.05)),
        "subjects": subjects,
        "fallback": {"name": grp("name"), "provider_year": grp("py"), "provider": grp("provider"), "global": [float(np.mean(ab)), max(gvar + float(np.mean(v_a)), 0.05)]},
        "fit": {
            "n_pairs": int(len(df)),
            "n_benchmarks": int(len(b_keys)),
            "n_subjects": int(len(s_keys)),
            "excluded": sorted(excl),
            "ridge_a": a.ridge_a,
            "ridge_d": a.ridge_d,
            **diag,
        },
    }
    out = Path(a.out_root) / a.name
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(TEMPLATE, out / "model.py")
    (out / "params.json").write_text(json.dumps(params))
    print(
        f"{a.name}: pairs={len(df)} benchmarks={len(b_keys)} subjects={len(s_keys)} "
        f"sigma2={sigma2:.3f} mu_d={params['mu_d']:.3f} tau_d2={params['tau_d2']:.3f} "
        f"resid_var={diag['resid_var']:.3f} noise={diag['noise']:.3f}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
