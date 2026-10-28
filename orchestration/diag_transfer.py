"""Diagnostic (dev holdout only, never the hill's private data): does fitted subject
ability predict held-out pair accuracy within a benchmark?

    python diag_transfer.py --params subs/M2_s2_a1_sig1/params.json --holdout .../dev/holdout.parquet
"""
import argparse, json, importlib.util, sys
import numpy as np, pandas as pd
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--params", required=True); ap.add_argument("--holdout", required=True)
a = ap.parse_args()
assert "hill/private" not in a.holdout, "this diagnostic must not read the hill's private labels"
spec = importlib.util.spec_from_file_location("m", Path(a.params).parent / "model.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
ho = pd.read_parquet(a.holdout, columns=["benchmark_id", "subject", "label"])
g = ho.groupby(["benchmark_id", "subject"])["label"].agg(["sum", "size"]).reset_index()
g["acc"] = g["sum"] / g["size"]; g["logit"] = np.log((g["sum"] + 0.5) / (g["size"] - g["sum"] + 0.5))
g["a"] = [m.subject_prior(json.loads(s))[0] for s in g["subject"]]
g["known"] = [m.subject_key(json.loads(s)) in m.SUBJ for s in g["subject"]]
print(f"pairs={len(g)} known subjects={int(g.known.sum())}")
for b, gb in g.groupby("benchmark_id"):
    if len(gb) >= 5:
        r = np.corrcoef(gb["a"], gb["logit"])[0, 1]
        slope = np.polyfit(gb["a"], gb["logit"], 1)[0] if gb["a"].std() > 1e-6 else float("nan")
        print(f"{b}: n={len(gb)} mean_acc={gb.acc.mean():.3f} corr(a, logit)={r:+.3f} slope={slope:+.2f} sd(a)={gb.a.std():.2f} sd(logit)={gb.logit.std():.2f}")
gc = g.copy(); gc["logit_c"] = gc["logit"] - gc.groupby("benchmark_id")["logit"].transform("mean"); gc["a_c"] = gc["a"] - gc.groupby("benchmark_id")["a"].transform("mean")
print(f"pooled within-benchmark corr={np.corrcoef(gc.a_c, gc.logit_c)[0,1]:+.3f} slope={np.polyfit(gc.a_c, gc.logit_c, 1)[0]:+.2f}")
