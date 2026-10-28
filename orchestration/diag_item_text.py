"""Go/no-go diagnostic: do cheap text features of an item predict its difficulty within a
benchmark, and do benchmark-level text aggregates predict benchmark difficulty?

    python diag_item_text.py --benchmarks a,b,c

Per benchmark: Rasch-style item difficulties delta_i (as in fit_items.py), standardised
features of item_content, within-benchmark OLS -> R^2 (5-fold CV). Pooled: one linear
model on within-benchmark standardised features across all benchmarks -> CV R^2.
Benchmark level: regress mean logit accuracy on mean features across benchmarks.
"""
from __future__ import annotations
import argparse, re, sys
import numpy as np, pandas as pd
from common import HELD_OUT, bench_dir

CODE = re.compile(r"(def |class |import |\{|\}|;|```|SELECT |function )")
MATH = re.compile(r"(\$|\\frac|\\sum|=|\+|\d+\.\d+)")

def feats(text: str) -> list[float]:
    t = text or ""
    n = max(len(t), 1)
    return [np.log1p(len(t)), np.log1p(t.count("\n")), np.log1p(len(t.split())), sum(c.isdigit() for c in t) / n,
            len(CODE.findall(t)) / max(len(t.split()), 1), len(MATH.findall(t)) / max(len(t.split()), 1),
            t.count("?") , float(bool(re.search(r"\b(A|B|C|D)[\).:]", t))), sum(c.isupper() for c in t) / n]

FEAT_NAMES = ["loglen", "loglines", "logwords", "digit_frac", "code_rate", "math_rate", "n_q", "has_mc", "upper_frac"]

def deltas(name):
    d = bench_dir(name); rp = d / "responses.parquet"
    if not rp.exists(): rp = d / "response.parquet"
    ip = d / "items.parquet"
    if not (rp.exists() and ip.exists()): return None
    r = pd.read_parquet(rp, columns=["subject_id", "item_id", "trial", "response"])
    r = r[r["response"].isin([0, 1, 0.0, 1.0])].sort_values(["subject_id", "item_id", "trial"]).drop_duplicates(["subject_id", "item_id"])
    sc = r.groupby("subject_id").size(); r = r[r["subject_id"].isin(sc[sc >= 20].index)]
    ic = r.groupby("item_id").size(); r = r[r["item_id"].isin(ic[ic >= 5].index)]
    if r["item_id"].nunique() < 50 or r["subject_id"].nunique() < 5: return None
    y = r["response"].to_numpy(float); s_idx, s_keys = pd.factorize(r["subject_id"]); i_idx, i_keys = pd.factorize(r["item_id"])
    a = np.zeros(len(s_keys)); dl = np.zeros(len(i_keys))
    for _ in range(40):
        for tgt, idx, size in ((a, s_idx, len(s_keys)), (dl, i_idx, len(i_keys))):
            p = 1 / (1 + np.exp(-(a[s_idx] + dl[i_idx]))); g = np.bincount(idx, weights=y - p, minlength=size) - tgt
            h = np.bincount(idx, weights=p * (1 - p), minlength=size) + 1.0; tgt += np.clip(g / h, -1, 1)
    items = pd.read_parquet(ip, columns=["item_id", "content"]).set_index("item_id")
    content = items["content"].reindex(i_keys)
    X = np.array([feats(None if c is None else str(c)) for c in content])
    return name, X, dl, float(y.mean())

def cv_r2(X, yv, k=5):
    n = len(yv); idx = np.arange(n) % k; pred = np.zeros(n)
    for f in range(k):
        tr, te = idx != f, idx == f
        Xtr = np.c_[np.ones(tr.sum()), X[tr]]; beta = np.linalg.lstsq(Xtr.T @ Xtr + 1e-3 * np.eye(Xtr.shape[1]), Xtr.T @ yv[tr], rcond=None)[0]
        pred[te] = np.c_[np.ones(te.sum()), X[te]] @ beta
    return 1 - ((yv - pred) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--benchmarks", required=True); a = ap.parse_args()
    rows = [x for x in (deltas(n) for n in a.benchmarks.split(",") if n and n not in HELD_OUT) if x]
    Xs, ys, bl = [], [], []
    for name, X, dl, acc in rows:
        Xz = (X - X.mean(0)) / (X.std(0) + 1e-9); dz = dl - dl.mean()
        r2 = cv_r2(Xz, dz)
        corr = [np.corrcoef(Xz[:, j], dz)[0, 1] if Xz[:, j].std() > 0 else 0 for j in range(X.shape[1])]
        top = sorted(zip(FEAT_NAMES, corr), key=lambda t: -abs(t[1]))[:3]
        print(f"{name}: items={len(dl)} sd(delta)={dl.std():.2f} within-CV-R2={r2:+.3f} top={[(n, round(c, 2)) for n, c in top]}", file=sys.stderr)
        Xs.append(Xz); ys.append(dz); bl.append((name, acc, X.mean(0)))
    X = np.vstack(Xs); yv = np.concatenate(ys)
    print(f"POOLED within-benchmark CV R2 = {cv_r2(X, yv):+.3f} over {len(rows)} benchmarks, {len(yv)} items", file=sys.stderr)
    B = np.array([b[2] for b in bl]); acc = np.array([b[1] for b in bl]); lg = np.log(acc / (1 - acc))
    Bz = (B - B.mean(0)) / (B.std(0) + 1e-9)
    print(f"BENCHMARK-LEVEL: n={len(bl)} CV R2 of mean logit acc on mean text features = {cv_r2(Bz, lg, k=min(5, len(bl))):+.3f}", file=sys.stderr)
    for j, n in enumerate(FEAT_NAMES):
        print(f"  corr(bench {n}, logit acc) = {np.corrcoef(Bz[:, j], lg)[0,1]:+.2f}", file=sys.stderr)

if __name__ == "__main__":
    main()
