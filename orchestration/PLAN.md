# Orchestration plan: climbing the PAIEC hill on the aicr cluster without AutoLab's agent

Claude (this session) is the orchestrator. The EC2 box only edits code and submits
Slurm jobs; every download, fit and evaluation runs on aicr. The hill's `eval.py` is
the single source of truth for scores. Nothing here uses AutoLab compute or LLM spend.

## Layout on aicr

```
/home/serge_ad_aihub/paiec/
  hill/            copy of the hill (eval.py, README, private/holdout.parquet)
  orchestration/   this directory, rsynced from the EC2 repo
  .venv/           python 3.12: pyarrow pandas numpy scipy huggingface_hub
  subs/<name>/     submissions (model.py + artifacts), one dir per experiment
  runs/<name>/     eval reports (json) per submission x split x seed
/scratch/serge_ad_aihub/paiec/
  data/measurement-db/<bench>/formatted_tables/{subjects,responses}.parquet
  hf_cache/        HF_HOME
  catalog.csv      one row per measurement-db benchmark: type, granularity, sizes
```

Slurm: account `sundai_aihub`, partition `cpu` (5 nodes x 128 cores, 1.1 TB RAM, 24 h).
Everything in this task is CPU-bound; GPUs are not needed.

## What we are optimising

`brier_alc = 0.1 B0 + 0.2 (B1 + B3 + B7 + B15) + 0.1 B31` over held-out benchmarks.
Validation split (what we tune on): editbench, appworld, hallusionbench, financebench.
Final split (touched once per milestone, never tuned on): algotune, brace, bird_sql,
elicitation_game. Reference points on validation at default params: constant 0.5 = 0.25,
pooled Beta = 0.225 (seed 0). Because only 4 benchmarks are in each split, every
decision is also checked on a *dev simulator*: the same `eval.py` pointed at a holdout
built from 12 to 16 training benchmarks that the candidate model never saw (leave
benchmarks out). Validation + dev agree before anything is called a win.

## The model ladder (each rung is one submission dir, scored in parallel)

| id | idea | why it should help | budget it targets |
|----|------|--------------------|-------------------|
| M0 | pooled Beta (hill baseline) | reference | all |
| M1 | subject ability prior from training data: logit p = a_s + mean difficulty | B0 is 10 % of the score and currently a constant 0.5 (Brier 0.25). Even a crude per-subject accuracy from public benchmarks should beat it. Unknown subjects fall back to provider x release-year means | B0, B1 |
| M2 | hierarchical logit: p_sb = sigmoid(a_s + d_b + e_sb), with the benchmark difficulty d_b inferred from *all* revealed labels of the run on that benchmark given the known a_s, and e_sb ~ N(0, sigma2) shrunk from the pair's own labels; sigma2 and the d_b prior fitted on training pairs | principled version of pooled Beta that uses the subject prior; the main lever reported by participants | B1 to B15 |
| M3 | M2 + item-level corrections: within-benchmark item difficulty from other subjects' labels on the same items when they overlap, plus cheap text features (length, code/math markers) normalised within benchmark | B15/B31 and ECE; expected small | B15, B31 |
| M4 | acquisition function (`labeling.py`): choose items that maximise information about d_b and e_sb (e.g. items already labelled by other subjects in the run, then uncertainty sampling) | cheaper labels at the same budget | B1 to B7 |
| M5 | ensembles / calibration: temperature on logits, mixture of M2 variants, tuned on dev only | squeeze the last 0.005 | all |

Each rung keeps the previous one's code path and adds a switch, so an ablation is a
flag, not a new file. Hyperparameters are swept as Slurm array jobs on the dev simulator.

## Data in stages (Serge's rule: small first, prove it learns, then scale)

1. **Catalogue** (one small job): read every benchmark's `benchmarks.parquet` and file
   sizes from the Hub. Output `catalog.csv`. Select the training pool: binary, item-level,
   public, not one of the 8 held-out benchmarks.
2. **Stage 1 (about 15 benchmarks, under 2 GB)**: subjects + responses tables only. Pick
   benchmarks with many recent (2025+) subjects and small response files. Fit M1/M2,
   score on validation and dev. Record the gain over M0.
3. **Stage 2 (about 50 benchmarks)**: refit M1/M2. If the score does not move, data
   volume is not the lever and we stop fetching.
4. **Stage 3 (all qualifying benchmarks, items tables only where M3 needs them)**.

## Protocol for every experiment

- One submission dir per variant; artifacts are fitted on the cluster and written next
  to `model.py`; `model.py` is standard library + numpy only and loads them relative to
  its own path.
- Scored by `orchestration/run_eval.py` on validation (seeds 0,1,2) and dev; the
  results table lives in `orchestration/RESULTS.md` and is updated after every batch.
- The final split is scored only when a rung is promoted, and the number is recorded,
  never acted on.
- Never read `private/` from model code; never include the 8 held-out benchmarks in
  the training pool or the dev holdout.

## Parallelism

- Fetch: one Slurm task per benchmark (array), each downloads two parquet files.
- Fit: one task per model variant x hyperparameter setting (array).
- Eval: one task per submission x split x seed; each is seconds to a minute.
- Login node runs only `sbatch`, `squeue`, rsync and tiny file reads.

## Findings log (2026-10-28)

Hill validation, mean Brier ALC over seeds 0 to 2; pooled Beta baseline = 0.2225, constant 0.5 = 0.25.

| rung | best variant | hill validation | notes |
|------|--------------|-----------------|-------|
| M0 | pooled Beta | 0.2225 | reference |
| M1 | subject prior at B0 + pooled Beta | 0.2241 | B0 gets *worse* than 0.5: public-benchmark ability does not place a model on an unseen benchmark |
| M2 | hierarchical logit, a_mult 0, sigma2 about 0.6 to 0.75 | 0.2195 | gain is all at budgets 1 to 7; the fitted pair variance (0.37 on 20 benchmarks, 2.3 on 111) is the wrong quantity, tune it on dev |
| M3 | M2 + item difficulty from other subjects' labels on the same item, omega_mult 3 to 6 | 0.2146 | biggest single lever; B31 drops from 0.203 to 0.187; also best on both dev holdouts |
| M4 | M3 + coordinated acquisition | 0.2149 | helps B1/B3, hurts B15/B31; hybrid rule (coordinate first k labels) under test |
| final split (touched once) | M3 0.1591, M2 0.1599, M0 0.1633 | | gains carry over |

Dead ends, each closed by one cheap experiment:
- Subject-ability transfer (a_mult > 0): within-benchmark correlation of fitted ability with held-out accuracy is only about 0.3 (slope 0.5); any a_mult > 0 is neutral to harmful.
- More training data: 111 benchmarks (12.6M responses) vs 20 small ones gives the same score; the training set matters for hyperparameters, not for the predictions.
- Text features of items (length, code/math markers, MC format, ...): pooled within-benchmark CV R² = 0.018 over 22 benchmarks / 178k items, signs flip across benchmarks; benchmark-level aggregates do not predict benchmark difficulty either. Not worth the 3 GB items corpus.
- Estimating sigma2 on recent subjects only: comes out even larger (6); the hill's subject pool behaves far more homogeneously than the public tables suggest.
