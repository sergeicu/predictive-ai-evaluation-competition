"""Baseline: pooled Beta smoothing (standard library only).

For an item from anonymous benchmark B asked about subject S:

* p0 = (S_B + 2) / (N_B + 4), the smoothed success rate of *every* revealed label
  on benchmark B, from any subject. This is the benchmark base rate.
* p  = (s + 2 * p0) / (n + 2), where s, n count revealed labels for the exact
  (S, B) pair. Two pseudo-observations pull toward the base rate.
* 0.5 when nothing about B has been revealed (budget 0).

Participants in the live competition report this family of predictor at roughly
0.19 Brier ALC against 0.216 for the organizers' unsmoothed empirical mean and
0.25 for a constant 0.5.
"""


def predict(input, labeled=None):
    subject, item = input
    bench = item.get("benchmark_id")
    if not labeled or not bench:
        return 0.5

    S = N = 0  # all subjects on this benchmark
    s = n = 0  # this subject on this benchmark
    for (obs_subject, obs_item), y in labeled:
        if obs_item.get("benchmark_id") != bench:
            continue
        y = int(y)
        S += y
        N += 1
        if obs_subject == subject:
            s += y
            n += 1
    if N == 0:
        return 0.5
    p0 = (S + 2.0) / (N + 4.0)
    p = (s + 2.0 * p0) / (n + 2.0)
    return min(0.999, max(0.001, p))
