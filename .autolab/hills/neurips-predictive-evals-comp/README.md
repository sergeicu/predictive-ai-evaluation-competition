Predict whether an AI model will answer a benchmark question correctly, without running the model, on benchmarks it has never been evaluated on, and improve the prediction fast as a few true answers are revealed. Scored by Brier ALC, the official metric of the NeurIPS 2026 Predictive AI Evaluation Competition.

## The task

This hill is a local replica of the [Predictive AI Evaluation Competition](https://aims.stanford.edu/competition) (PAIEC, Stanford AIMS, NeurIPS 2026). A *subject* is an AI model configuration (name, provider, release date, harness, reasoning effort). An *item* is one benchmark question. The response is 1 if the subject answered it correctly, else 0. You must output `P(correct)` for a (subject, item) pair **without running the subject**.

Two twists, identical to the real competition:

1. **Benchmark-level cold start.** The items you are scored on come from benchmarks that are held out. Within a run you see an anonymous `benchmark_id`, so you can tell which items belong together, but you never learn the benchmark's name and you have no training rows for it. Subjects, by contrast, are mostly models that appear in public training data.
2. **Label budgets.** For every held-out (subject, benchmark) pair the items are split 50/50 into an acquisition pool and an evaluation pool. Labels are revealed from the acquisition pool at budgets 0, 1, 3, 7, 15 and 31 (nested). You are scored on the evaluation pool at every budget. The revealed labels for *all* sampled pairs are passed to you together, so at budget 3 you also see 3 labels for every other subject on the same benchmark and for every other benchmark in the run.

The metric stands in for: how much evidence does a decision maker need before they can trust a predicted score on an unevaluated benchmark? A submission that only wins by memorising the held-out items (see below) has missed the point and is also disqualified.

## Training data

Not bundled; it is 65 GB. Pull it yourself from Hugging Face, with `HF_TOKEN` set and gated access accepted:

```
aims-foundations/measurement-db        # 249 benchmarks: subjects / items / responses parquet per benchmark
aims-foundations/measurement-db-embed  # text-embedding-3-small vectors for a few benchmarks' items
```

Each benchmark directory has `formatted_tables/{benchmarks,subjects,items,responses}.parquet`. Filter the catalogue to `response_type == "binary"`, `granularity == "item"`, `release == "public"`; roughly 160 benchmarks qualify. The baseline repo's `tools/prepare_data.py` at https://github.com/aims-foundations/paiec_baseline shows one way to fetch them.

**You must exclude the held-out benchmarks listed in `holdout_benchmarks.json` from training, feature fitting, retrieval indices, and anything else.** Their items are what you are scored on. Using them, or matching held-out item text back to its public source to look up the answer, is cheating and will be treated as such. The real competition's test benchmarks are private; this hill can only approximate that by asking you not to look.

## What to submit

A directory with the same layout you would zip for Codabench:

```
model.py        required   def predict(input, labeled) -> float
labeling.py     optional   def acquisition_function(input, prediction=None, labeled=None, context=None) -> bool
<anything else> optional   fitted parameters, lookup tables, templates (.npz/.parquet/.json/...)
```

`predict` receives:

- `input = [subject, item]`.
  `subject` is a dict with keys `normalized_name, provider, release_date, access_date, harness, harness_version, reasoning_effort, subject_features_extra` (strings or `None`).
  `item` is a dict with keys `item_content` (the question text, may be long), `item_features` (a short string or `None`), `interactors` (`None` or a list), `benchmark_id` (anonymous string, stable within a run).
- `labeled`: a list of `[[subject, item], label]` with `label` in `{0, 1}`, one entry per revealed acquisition item across every sampled pair in the run. Empty at budget 0.

It must return a finite float in `[0, 1]`. Anything else, or an exception, fails the whole report. The evaluator runs `predict` with the current working directory set to your submission directory, so relative paths to your artifacts work.

If `labeling.py` is present, the evaluator streams each pair's acquisition pool to `acquisition_function` in random order and buys the item when it returns `True`; `context` carries `labels_remaining`, `items_remaining` and `budget`. Items are force-bought when the remaining items equal the remaining budget. Without `labeling.py` the first 31 items in random order are revealed.

Runtime: no network is needed and none should be assumed. Keep imports to the standard library plus numpy/pandas/pyarrow; heavier dependencies must be vendored or pre-fitted into artifacts. The evaluator has a total time budget (`time_budget_s`, default 3000 s) covering every `predict` call; default settings make about 5,000 to 7,000 calls per report.

`examples/baseline/` is a working submission: pooled Beta smoothing over the revealed labels, standard library only.

## How you are scored

For each pair and budget *n*, the Brier score is the mean of `(p - y)^2` over that pair's evaluation items. `B_n` is the mean over pairs (every pair weighs the same, however many items it has). Then

```
brier_alc = 0.1*B0 + 0.2*B1 + 0.2*B3 + 0.2*B7 + 0.2*B15 + 0.1*B31
```

Lower is better. A constant 0.5 scores exactly 0.25. The organizers' unsmoothed empirical mean scores about 0.216 on the real hidden set and the pooled-Beta baseline about 0.19; strong entries sit near 0.17 to 0.18. Expected calibration error (10 bins) per budget is reported in `details` as a diagnostic and does not rank.

| metric      | direction | meaning                                                      |
|-------------|-----------|--------------------------------------------------------------|
| `brier_alc` | min       | weighted area under the Brier learning curve over 6 budgets  |

The held-out benchmarks are split in two sets. While you work, the score comes from the `validation` set. `hills eval ... --final` scores the `final` set, which uses different benchmarks, so tuning to the validation benchmarks does not transfer by construction. Both sets are listed in `holdout_benchmarks.json`.

## Parameters

| param                     | default | meaning                                                                 |
|---------------------------|---------|-------------------------------------------------------------------------|
| `seed`                    | 0       | controls which pairs are sampled and the acquisition/evaluation split    |
| `max_eval_items`          | 100     | evaluation items scored per pair (deterministic subsample)              |
| `max_pairs_per_benchmark` | 12      | subjects sampled per held-out benchmark                                 |
| `time_budget_s`           | 3000    | total wall-clock allowed for all `predict` calls                        |

Reports rank only within matching `mode`, `seed`, `max_eval_items` and `max_pairs_per_benchmark`. Pass one with `hills eval <dir> -H neurips-predictive-evals-comp -p seed=3`.

## What the evaluator will not do

- It will not run the subject on the item, and neither may you.
- It will not read a metric your code computed; it computes Brier ALC itself from `private/`.
- It will not reveal evaluation labels or score acquisition items.
- Editing a report file invalidates its signature; `hills verify` catches it.
