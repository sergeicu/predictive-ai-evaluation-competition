"""Hierarchical logit predictor for PAIEC (numpy + standard library).

    p(correct | subject s, benchmark B, item i) = sigmoid(a_s + d_B + e_sB + delta_i)

* a_s     subject ability from public measurement-db benchmarks (params.json) with a
          posterior variance v_s; unknown subjects fall back to provider x release-year,
          provider, or the global pool. `a_mult` shrinks ability differences toward the
          training mean (0 = ignore subjects entirely).
* d_B     benchmark difficulty, prior N(mu_d, tau_d2), updated from *every* revealed
          label on B in the run (all subjects).
* e_sB    pair deviation ~ N(0, sigma2 + v_s), updated from the pair's own labels.
* delta_i item difficulty ~ N(0, omega2), updated from other subjects' revealed labels
          on the same item (mode M3 only; items are identified by their content).

Everything is integrated numerically on grids, so the output is a posterior predictive
probability. Modes: M1 = budget-0 prior + pooled Beta afterwards (ablation);
M2 = subject + benchmark + pair; M3 = M2 + item effects.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "params.json").read_text())
MODE = P.get("mode", "M2")
SIGMA2 = float(P["sigma2"]) * float(P.get("sigma_mult", 1.0))
MU_D, TAU_D2 = float(P["mu_d"]), float(P["tau_d2"]) * float(P.get("tau_mult", 1.0))
OMEGA2 = float(P.get("omega2", 0.0)) * float(P.get("omega_mult", 1.0))
SUBJ = P["subjects"]  # key -> [a, v]
FALLBACK = P["fallback"]  # level -> {key: [mean, var]}
ND, NE, NI = int(P.get("n_d", 61)), int(P.get("n_e", 41)), int(P.get("n_i", 25))
TEMP = float(P.get("temp", 1.0))
A_MULT = float(P.get("a_mult", 1.0))
MEAN_A = float(P.get("mean_a", 0.0))
B0_HALF = bool(P.get("b0_half", False))
USE_ITEMS = MODE == "M3" and OMEGA2 > 0


def _std_grid(n):
    z = np.linspace(-4.0, 4.0, n)
    lw = -0.5 * z**2
    lw -= np.log(np.exp(lw).sum())
    return z, lw


_Z_D, _LOGW_D = _std_grid(ND)
_Z_E, _LOGW_E = _std_grid(NE)
_Z_I, _LOGW_I = _std_grid(NI)
_D = MU_D + math.sqrt(TAU_D2) * _Z_D  # d grid (ND,)
_DELTA = math.sqrt(OMEGA2) * _Z_I if OMEGA2 > 0 else np.zeros(1)

_cache_bench: dict = {}
_cache_pair: dict = {}
_cache_item: dict = {}


# ----------------------------------------------------------------------------- keys


def _norm(v) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() in ("nan", "none", "null") else s


def _year(s):
    s = _norm(s)
    return s[:4] if s[:4].isdigit() else None


def subject_key(subject: dict) -> str:
    return f"{_norm(subject.get('normalized_name'))}|{_norm(subject.get('reasoning_effort'))}"


def _sid(subject: dict) -> str:
    """Exact pair identity: the full subject dict (several subjects can share a name)."""
    return json.dumps(subject, sort_keys=True, default=str)


def _iid(item: dict) -> str | None:
    c = item.get("item_content")
    if c is None:
        return None
    return hashlib.sha1(str(c).encode("utf-8", "ignore")).hexdigest()


def _scale(a: float) -> float:
    """Shrink ability differences toward the training mean by the transfer coefficient."""
    return MEAN_A + A_MULT * (a - MEAN_A)


def subject_prior(subject: dict) -> tuple[float, float]:
    """(ability mean, ability variance) for a subject, with fallbacks."""
    k = subject_key(subject)
    if k in SUBJ:
        m, v = SUBJ[k]
    else:
        name = _norm(subject.get("normalized_name"))
        prov, yr = _norm(subject.get("provider")), _year(subject.get("release_date"))
        if name in FALLBACK.get("name", {}):
            m, v = FALLBACK["name"][name]
        elif yr and f"{prov}|{yr}" in FALLBACK.get("provider_year", {}):
            m, v = FALLBACK["provider_year"][f"{prov}|{yr}"]
        elif prov in FALLBACK.get("provider", {}):
            m, v = FALLBACK["provider"][prov]
        else:
            m, v = FALLBACK["global"]
    return _scale(float(m)), float(v) * A_MULT**2


# ------------------------------------------------------------------------ numerics


def _log_sigmoid(x):
    return -np.logaddexp(0.0, -x)


def _loglik(k: int, n: int, a: float, var_e: float):
    """log Binomial-kernel likelihood on the (d, e) grid, shape (ND, NE)."""
    e = math.sqrt(var_e) * _Z_E
    x = (a + _D)[:, None] + e[None, :]
    return k * _log_sigmoid(x) + (n - k) * _log_sigmoid(-x)


def _logsumexp(x, axis):
    m = x.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def _bench_summary(bench: str, labeled):
    """Counts per subject on this benchmark, per-item label lists, and the log posterior
    over d from everyone's counts. Cached by a fingerprint of the revealed labels."""
    counts: dict[str, list] = {}
    subjects: dict[str, dict] = {}
    items: dict[str, list] = {}
    for (s, it), y in labeled:
        if it.get("benchmark_id") != bench:
            continue
        sid = _sid(s)
        c = counts.setdefault(sid, [0, 0])
        c[0] += int(y)
        c[1] += 1
        subjects.setdefault(sid, s)
        if USE_ITEMS:
            iid = _iid(it)
            if iid is not None:
                items.setdefault(iid, []).append((sid, int(y)))
    fp = (bench, tuple(sorted((k, c[0], c[1]) for k, c in counts.items())), len(labeled))
    hit = _cache_bench.get(fp)
    if hit is not None:
        return hit
    marg: dict[str, np.ndarray] = {}
    for sid, (k, n) in counts.items():
        a, v = subject_prior(subjects[sid])
        marg[sid] = _logsumexp(_loglik(k, n, a, SIGMA2 + v) + _LOGW_E[None, :], axis=1)
    total = _LOGW_D.copy()
    for m in marg.values():
        total = total + m
    out = {"counts": counts, "subjects": subjects, "items": items, "marg": marg, "total": total, "fp": fp}
    if len(_cache_bench) > 32:
        _cache_bench.clear()
        _cache_pair.clear()
        _cache_item.clear()
    _cache_bench[fp] = out
    return out


def _pair_posterior(summary, subject):
    """Normalised weights over the (d, e) grid for this pair, the logit grid, and the
    posterior mean logit."""
    sid = _sid(subject)
    ck = (summary["fp"], sid)
    hit = _cache_pair.get(ck)
    if hit is not None:
        return hit
    a, v = subject_prior(subject)
    var_e = SIGMA2 + v
    log_post_d = summary["total"] - summary["marg"][sid] if sid in summary["marg"] else summary["total"]
    k, n = summary["counts"].get(sid, (0, 0))
    joint = log_post_d[:, None] + _LOGW_E[None, :] + _loglik(k, n, a, var_e)
    joint -= joint.max()
    w = np.exp(joint)
    w /= w.sum()
    x = (a + _D)[:, None] + (math.sqrt(var_e) * _Z_E)[None, :]
    out = (w, x, float((w * x).sum()))
    _cache_pair[ck] = out
    return out


def _item_posterior(summary, iid, exclude_sid):
    """Normalised weights over the delta grid for this item from other subjects' labels."""
    ck = (summary["fp"], iid)
    hit = _cache_item.get(ck)
    if hit is not None:
        return hit
    ll = _LOGW_I.copy()
    for sid, y in summary["items"].get(iid, []):
        if sid == exclude_sid:
            continue
        mu = _pair_posterior(summary, summary["subjects"][sid])[2]
        x = mu + _DELTA
        ll = ll + (_log_sigmoid(x) if y else _log_sigmoid(-x))
    ll -= ll.max()
    w = np.exp(ll)
    w /= w.sum()
    _cache_item[ck] = w
    return w


def _pooled_beta(subject, bench, labeled):
    S = N = s = n = 0
    for (obs_s, obs_i), y in labeled:
        if obs_i.get("benchmark_id") != bench:
            continue
        y = int(y)
        S += y
        N += 1
        if obs_s == subject:
            s += y
            n += 1
    if N == 0:
        return None
    p0 = (S + 2.0) / (N + 4.0)
    return (s + 2.0 * p0) / (n + 2.0)


def _clip(p):
    return float(min(0.999, max(0.001, p)))


# ------------------------------------------------------------------------- predict


def predict(input, labeled=None):
    subject, item = input
    bench = item.get("benchmark_id")
    labeled = labeled or []

    if MODE == "M1" and labeled:
        p = _pooled_beta(subject, bench, labeled)
        if p is not None:
            return _clip(p)

    if labeled:
        summary = _bench_summary(bench, labeled)
    else:
        summary = {"counts": {}, "subjects": {}, "items": {}, "marg": {}, "total": _LOGW_D, "fp": (bench, (), 0)}
    if not summary["counts"] and B0_HALF:
        return 0.5

    w, x, _ = _pair_posterior(summary, subject)
    if OMEGA2 > 0 and MODE == "M3":
        iid = _iid(item)
        if iid is not None and iid in summary["items"]:
            wi = _item_posterior(summary, iid, _sid(subject))
        else:
            wi = np.exp(_LOGW_I)  # unseen item: prior over its difficulty
        sig = 1.0 / (1.0 + np.exp(-(x[:, :, None] + _DELTA[None, None, :]) / TEMP))
        return _clip(float((w[:, :, None] * wi[None, None, :] * sig).sum()))
    sig = 1.0 / (1.0 + np.exp(-x / TEMP))
    return _clip(float((w * sig).sum()))
