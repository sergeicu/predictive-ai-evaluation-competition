# Predictive AI Evaluation Competition (PAIEC), NeurIPS 2026: research deep dive

Research notes on the [Predictive AI Evaluation Competition](https://aims.stanford.edu/competition) run by Stanford's AI Measurement Science (AIMS) center at NeurIPS 2026. Compiled 2026-10-20, ten days before the 30 October 2026 submission deadline.

This repo contains three research reports, each produced by a dedicated research pass over the web, GitHub, Hugging Face, Codabench, and arXiv:

| Report | What it covers |
|---|---|
| [`reports/01_community_and_findings.md`](reports/01_community_and_findings.md) | Everything publicly said about the competition: official sources, Codabench stats, the baseline repo, measurement-db, ~10 participant repos with self-reported scores, the Spring 2026 predecessor challenge, and what was searched but not found. |
| [`reports/02_tools_and_libraries.md`](reports/02_tools_and_libraries.md) | Open-source libraries and repos usable for the task: organizer-adjacent code (torch_measure, reeval), IRT libraries, benchmark-prediction repos, collaborative filtering, active learning, calibration. Includes a comparison table and a recommended stack. |
| [`reports/03_literature_overview.md`](reports/03_literature_overview.md) | 70 papers across measurement science, IRT for LLMs, performance prediction, per-instance "assessors", cold start, active testing, calibration, and ALC metrics. Ends with a synthesis, open questions, and a top-10 reading list. |

## The competition in plain language

Evaluating AI models is expensive. Running every model on every benchmark question costs compute, time, and money. The competition asks: **can we predict how a model will do on a question without running it?**

Picture a giant spreadsheet. Rows are AI models. Columns are individual benchmark questions. Each cell is 1 if the model answered correctly, 0 if not. Most cells are filled from past evaluations. The task is to fill in the blanks with a probability between 0 and 1.

Two twists:

1. **Benchmark-level cold start.** The blank cells come from a benchmark nobody has seen in training. You know the model and have seen its results elsewhere. You can read the question text. But no model has been run on this benchmark.
2. **You can peek.** Before guessing, you may look at a few true answers from the new benchmark: 0, then 1, 3, 7, 15, 31. The score is the area under the Brier-score learning curve across those budgets (Brier ALC, lower is better). A good system learns a lot from very few revealed labels.

Key facts:

- **Submission:** a `submission.zip` to [Codabench](https://www.codabench.org/competitions/17828/) with `model.py` exposing `predict(input, labeled) -> float`, optional `labeling.py` with an acquisition function, optional `models.txt` and `requirements.txt`.
- **Metric:** Brier ALC = 0.1·B₀ + 0.2·B₁ + 0.2·B₃ + 0.2·B₇ + 0.2·B₁₅ + 0.1·B₃₁, averaged equally over subject–benchmark pairs. ECE is diagnostic only.
- **Data:** [`aims-foundations/measurement-db`](https://huggingface.co/datasets/aims-foundations/measurement-db) (gated) plus [`measurement-db-embed`](https://huggingface.co/datasets/aims-foundations/measurement-db-embed) (item embeddings, not gated). Only benchmarks in the [inventory sheet](https://docs.google.com/spreadsheets/d/1T69QyIZJRWX3d-Tmuh_lcFwtpqv9ytcLxfLWKn6gopI/edit?gid=0#gid=0) may be used for training.
- **Constraints:** 8 h per run; network only to listed LLM inference APIs; no web search; up to 5 HF model repos; pretrained models allowed with disclosure.
- **Deliverables:** code and a 4–8 page technical report (NeurIPS template, via [OpenReview](https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/PAIEC)) by **30 Oct 2026 23:59 AoE**. Session 11 Dec 2026, Sydney.
- **Baseline code:** https://github.com/aims-foundations/paiec_baseline

## Cross-cutting synthesis

### What the public evidence says about scores

No official leaderboard or baseline numbers exist. All numbers below are self-reported by participants in public repos and should be treated as anecdotal. Codabench reports 206 participants and 3,268 submissions as of 2026-10-20.

| Method family | Approximate Brier ALC on hidden formative set |
|---|---|
| Constant 0.5 | 0.250 |
| Organizers' empirical mean baseline | ≈0.216 (collapses at budget 1 because one label gives 0/1) |
| Beta-smoothed label mean, pooled across subjects on the same benchmark | ≈0.188–0.190 |
| Hierarchical IRT / logistic with online posterior updates | ≈0.18–0.19 |
| Single best lucky draw reported | ≈0.146 |

Single-submission noise is large (SD ≈0.01–0.03 on ~8–9 subject–benchmark pairs per draw), so individual formative scores barely distinguish methods. Several participants ran the same zip many times to average.

### What participants agree on

1. **Pooling revealed labels across subjects on the same benchmark is the dominant lever.** The `labeled` argument includes other sampled subjects' outcomes on the same benchmark. Using them to estimate benchmark difficulty is worth roughly 0.02–0.03 ALC.
2. **Cold-start priors transfer poorly.** Budget-0 Brier sits near 0.25 for most methods. Subject-metadata offsets sometimes hurt. Benchmark base rates differ by ~0.8 logits between benchmarks and cannot be guessed from text.
3. **Item-text features, sentence embeddings, and LLM difficulty ratings do not transfer across benchmarks.** Multiple independent negative results. LLM-generated "success notes" per item helped one team locally but were dropped.
4. **Local leave-one-benchmark-out validation did not predict hidden-test ranking** for at least one team. Pool-dependent methods that looked best locally lost on test.
5. **Learning curves flatten after ~7 labels** on hidden runs. Local simulations overstate late-budget gains.
6. **Uncertainty-based acquisition shows no measurable gain** in any reported experiment.
7. **Smoothing matters more than modelling.** The predecessor Spring 2026 challenge reached the same conclusion: "simple lookup-based statistics were more reliable than larger neural or LLM-based additions."

### What the literature predicts

- Aggregate-level predictability is high (model × task matrices are low-rank; ~100 items recover benchmark accuracy within ~2%).
- Per-item predictability is much lower. Prompt-only assessors on MMLU-Pro barely beat random; best AUROC ≈0.7. Interpretable demand rubrics (ADeLe) reach AUROC 0.75 on genuinely new benchmarks, while raw embeddings collapse to ≈0.48.
- Few-shot assessors saturate around ~100 reference labels. PAIEC's 31-label cap sits on the steep part of that curve, so the prior and shrinkage dominate.
- Benchmark-level cold start is largely untested in the IRT-for-LLM literature. Most IRT papers calibrate on the benchmark they evaluate.

### Where the gaps are

- Almost all teams use pooled-Beta or hierarchical-IRT. Nobody reports success with cross-benchmark item features. The ADeLe-style demand-rubric approach (LLM-annotated cognitive demands per item) is the one method with published evidence of new-benchmark transfer and seems untried here.
- The organizers' own [`torch_measure`](https://github.com/aims-foundations/torch_measure) library ships `AmortizedIRT` and a TabPFN cold-item predictor; no participant repo mentions using it.
- The `measurement-db-embed` dataset provides free text-embedding-3-small vectors for every public item. Only used by a few teams.
- Streaming Fisher-information acquisition (fluid-benchmarking, catsim) is untested; teams tried uncertainty sampling only.

### Suggested starting point

See §10 of the tools report. In short: a pooled-Beta or hierarchical-logistic predictor with Bayesian shrinkage toward the benchmark base rate, validated with leave-benchmark-out CV and many repeated formative submissions, is the robust floor. Anything added on top must be shown to beat it on the hidden set across many draws, not locally.

## Key links

- Competition page: https://aims.stanford.edu/competition
- Codabench: https://www.codabench.org/competitions/17828/
- Baseline: https://github.com/aims-foundations/paiec_baseline
- torch_measure: https://github.com/aims-foundations/torch_measure
- Data: https://huggingface.co/datasets/aims-foundations/measurement-db
- Embeddings: https://huggingface.co/datasets/aims-foundations/measurement-db-embed
- Inventory: https://docs.google.com/spreadsheets/d/1T69QyIZJRWX3d-Tmuh_lcFwtpqv9ytcLxfLWKn6gopI/edit?gid=0#gid=0
- Registration: https://docs.google.com/forms/d/e/1FAIpQLSevCdcDO1TaWqWUY7vRMljoYE97PJXC-X3Vg3ItD3kRi8XUmQ/viewform?usp=header
- OpenReview: https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/PAIEC

## Provenance

Research was carried out with Claude Code (Claude Fable 5.1) using three parallel research agents on 2026-10-20. Each report cites its sources inline and marks claims it could not verify. Participant scores are self-reported in their repos and have not been independently checked.
