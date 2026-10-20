# Open-source toolkit survey for PAIEC

*Predicting P(correct) for subject × item, benchmark-level cold start, Brier-scored. Researched 2026-10-20. All star counts and last-push dates were verified via `gh api` unless marked unverified; PyPI versions verified via the PyPI JSON API.*

## 0. Competition facts that shape tool choice (from the live rules page and baseline code)

Source: https://aims.stanford.edu/competition (the `aimslab.stanford.edu` URL redirects there) and `aims-foundations/paiec_baseline` code.

- Interface: `predict([subject, item], labeled) -> float` in `model.py`; optional `labeling.py` with `acquisition_function(input, prediction=None, labeled=None, context=None) -> bool` (streaming: evaluator walks a pool, you answer buy/skip; `context` has `labels_remaining`, `items_remaining`). Legacy one-arg form returns a numeric priority.
- Subject dict fields: `normalized_name, provider, release_date, access_date, harness, harness_version, reasoning_effort, subject_features_extra`. Item: `item_content, item_features, interactors, benchmark_id` (anonymous). Labels are only revealed for the same subject; "Only entries matching the complete target subject dictionary are included."
- Metric: "Brier ALC" (lower better), budgets 0/1/3/7/15/31 weighted 0.1/0.2/0.2/0.2/0.2/0.1; ECE is diagnostic only.
- Runtime: 8 h incl. setup; network only to Anthropic/OpenAI/Google/xAI/DeepSeek/Mistral/Together/Groq inference APIs; web search prohibited; no runtime model downloads except via `models.txt` (≤5 HF repos, ≤300B params, ≤1 TB each). The README says "Codabench does not install extra packages by default", **but** `check_submission_zip.py::_check_requirements` accepts a root `requirements.txt` with *named pip packages only* (no URLs/`-e`/local paths). Treat pip availability as "likely but confirm with a smoke submission"; safest plan is numpy/pandas/pyarrow-only inference with pre-fitted parameters shipped as `.npz/.parquet/.pt` (these suffixes are whitelisted as `SAFE_ARTIFACT_SUFFIXES`).
- Training data: `aims-foundations/measurement-db` (gated, CC BY-SA for curation; 249 benchmark migrations; `items.parquet`/`responses.parquet`/`subjects.parquet` per benchmark) **and** `aims-foundations/measurement-db-embed` (not gated; `text-embedding-3-small`, 1536-d, per-item vectors keyed by `item_id`+`content_sha1`). Ready-made item embeddings for every public item at zero API cost. This is the single most useful asset for cold-start item-difficulty models. At test time, embedding an unseen item requires an OpenAI call (allowed) or a local model listed in `models.txt` (the validator whitelists basic sentence-transformer model IDs).

## 1. The PAIEC baseline repo (what it does, what to swap)

**https://github.com/aims-foundations/paiec_baseline** — 2 stars, MIT, pushed 2026-10-15.

| Baseline | Method | Acquisition |
|---|---|---|
| `empirical_mean/` | mean of revealed labels for same (subject, benchmark_id); 0.5 at budget 0; no smoothing | evaluator default random |
| `ble/` (Bayesian Linguistic Evaluator) | GPT-5.6 "Luna" agent (default `openai/gpt-5.6-luna`, Responses API) does ≤10 turns of retrieval over the shipped measurement-db payload (keyword tools over tables + cosine retrieval over the embed index) and outputs a probability clipped to [0.02, 0.98]. **Fits no parameters.** | uncertainty sampling: buy if \|p−0.5\| ≤ 0.15 or forced |
| `empirical_mean_ble_acquisition/` | empirical mean | BLE probability for acquisition only |

Useful reusable pieces: `tools/prepare_data.py` (downloads measurement-db + embeddings into `payload/data/<bench>/{benchmarks,subjects,items,response,traces}.parquet` and `payload/embeddings/items/<bench>.parquet`), `ble/comp_pipeline/script/databank.py` (`MeasurementDB` loader), `embedding_index.py` (unit-norm float32 matrix, cosine search), `tools/streaming_ingestion.py` (the evaluator's `AcquisitionHook` so you can test your `labeling.py` signature locally), `check_submission_zip.py`, `tools/smoke_test.py --mock`.

What to swap in: (a) replace the BLE agent with a fitted latent-variable model (IRT/MF with item features) that uses the same payload; (b) replace `empirical_mean` with a Bayesian shrinkage posterior (Beta prior centred on the model's budget-0 prediction) so budgets 1–7 don't collapse to 0/1 (the README itself warns the mean "can result in a lower score than constant-0.5"); (c) replace the `labeling.py` policy with Fisher-information / expected-Brier-reduction selection; (d) optionally keep BLE as one feature/ensemble member for items with no comparable training benchmark.

## 2. Organizer-adjacent code (highest priority: same team, same data model)

| Repo | Stars / pushed / license | What it is | Maps to |
|---|---|---|---|
| **aims-foundations/torch_measure** https://github.com/aims-foundations/torch_measure — `pip install torch_measure` (0.1.0 on PyPI, py≥3.10) | 20 / 2026-07-04 / MIT | "PyTorch-native toolkit for predictive evaluation of AI systems." Models: `rasch, twopl, threepl, beta_rasch, beta_twopl, multifacet(_twopl), bifactor, testlet, bradley_terry, logistic_fm, ncf, amortized (AmortizedIRT: item params predicted from embeddings), tabpfn_predictor ("cold-item performance prediction"), llm_judge`. Fitting: `em, jml, mle, svi`. `cat/`: `fisher_information`, `MaxInfoStrategy`, `SpanningStrategy`, `RandomStrategy`, `calibration`. `metrics/calibration`, `metrics/reliability`, `data/response_matrix`, `data/masking`. Notably `NCF.predict(self, data: dict, labeled: list[dict]) -> float` mirrors the competition `predict(input, labeled)` signature. | zero-budget (AmortizedIRT/TabPFN), few-shot (ability update), acquisition (Fisher CAT) |
| **aims-foundations/reeval** (= sangttruong/reeval) https://github.com/aims-foundations/reeval — ICML 2025 "Reliable and Efficient Amortized Model-based Evaluation" (Truong, Tu, Liang, Li, Koyejo) | 8 / 2026-05-17 / no license file | Rasch IRT on 22 datasets × 172 LLMs; **amortized calibration = regress item difficulty from item-text embeddings** (Llama-3.1-8B / Mistral-7B embeddings); CAT in R + Python `cat/adap_test.py`; integrated into HELM. Sign convention: P = σ(θ + z). | zero-budget cold-start item difficulty; acquisition |
| **aims-foundations/irsl** https://github.com/aims-foundations/irsl — ICML 2026 "Item Response Scaling Laws" | 1 / 2026-05-26 / none | Binary-IRT and Beta-IRT (`utils.py`: calibration, CAT, losses); models ability as a function of pretraining FLOPs / test-time samples. Relevant to using `release_date`/provider as a smooth prior on θ for unseen subjects. | zero-budget subject prior |
| aims-foundations/measurement-db-embed (HF dataset, not gated) | lastModified 2026-09-13 | text-embedding-3-small vectors for all public items. | item features |

Caveats: torch_measure pulls `tabpfn<3,>=2.2`, `pyro-ppl`, `seaborn`, `tueplots`, heavy for a submission; fit offline and ship parameters, or vendor only the model file you need. reeval/irsl have no license files (research code; fine to read, re-implement rather than copy).

## 3. IRT libraries

| Tool | Verified maturity | License | Fit to task | Caveats |
|---|---|---|---|---|
| **py-irt** https://github.com/nd-ball/py-irt (`pip install py-irt` 0.7.1, 2026-03) | 175★, pushed 2026-10-08 | MIT | Pyro SVI 1PL/2PL/3PL/4PL, `multidim_2pl`, `amortized_1pl` (encoder from item text features → difficulty; ProdLDA-style). De-facto standard used by fluid-benchmarking, tinyBenchmarks, agent-psychometrics, benchpred (all vendor forks of it). Use for calibrating item params on training benchmarks. | `python <3.12` pin; torch+pyro dependency; SVI on ~10⁶ responses is slow-ish; no built-in cold-start for items without the amortized model. |
| **girth** https://github.com/eribean/girth (0.8.0, 2021) | 127★, pushed 2022-09-09 | MIT | numpy-only MML/JML estimation of 1–3PL, graded response, multidimensional; fast, no torch. Good for a dependency-light shipped model. | Effectively unmaintained; no missing-data support in some estimators (needs dense matrix). |
| pyirt https://github.com/17zuoye/pyirt | 111★, 2023-08 | MIT | EM 2PL for sparse long-format data. | Old (last PyPI 2019). |
| **EduCDM** https://github.com/bigdata-ustc/EduCDM (`pip install EduCDM` 1.0.1) | 201★, 2024-10-25 | Apache-2.0 | PyTorch IRT/MIRT/DINA/NCDM/KaNCD/ICD cognitive-diagnosis models; NCDM = neural IRT with knowledge-concept Q-matrix (could use `item_features`/benchmark tags as concepts). | Education-oriented APIs; no cold-start item path. |
| **catsim** https://github.com/douglasrizzo/catsim (`pip install catsim` 0.21.0, 2026-04) | 153★, pushed 2026-10-11 | BSD-3 | CAT engine: `MaxInfoSelector`, stratified/cluster/linear selectors, `NumericalSearchEstimator` (MLE/MAP θ), stopping rules; `catsim.irt` has 4PL ICC/information functions in numpy. Direct fit for `labeling.py` and for the few-shot θ update. | Pool-based (pick best item) whereas PAIEC is streaming (accept/reject this item); wrap info-threshold logic around it. |
| mirt (R) | — | GPL | Gold-standard MIRT; used by ATLAS, reeval, metabench. | R, not usable in a Python Codabench run; fine for offline calibration if you export params. |

## 4. IRT-for-LLM / benchmark-prediction repos

| Repo | Verified | What it gives you | Maps to |
|---|---|---|---|
| **felipemaiapolo/tinyBenchmarks** https://github.com/felipemaiapolo/tinyBenchmarks (`pip install git+…`, not on PyPI) | 185★, 2026-07-04, MIT | `tinyBenchmarks/tinyBenchmarks.py` + `tutorials/irt.py` (multidim 2PL via py-irt), anchor-point selection via k-means on IRT item embeddings, **p-IRT / gp-IRT** estimators (shrink IRT prediction toward observed mean with a sample-size-dependent weight). gp-IRT is precisely a principled few-shot (budget 1–31) combiner. | few-shot adaptation; acquisition (anchors) |
| **socialfoundations/benchmark-prediction (BenchPred)** https://github.com/socialfoundations/benchmark-prediction — NeurIPS 2025 "How Benchmark Prediction from Fewer Data Misses the Mark" | 6★, 2026-07-16, MIT | `pip install -e .`; one registry `benchpred.all_methods` with random, **AIPW**, PCA, Lasso, double-optimize, anchor_points (weighted/predictor), **PIRT, GPIRT**, metabench, tiny_bench; vendors `py_irt/models/multidim_2pl`. Key finding: all methods except Random-Sampling and AIPW fail on target models unlike training models, a warning for cold-start subjects. | few-shot; baselines |
| **allenai/fluid-benchmarking** https://github.com/allenai/fluid-benchmarking — COLM 2025 | 29★, 2025-09-16, Apache-2.0 | `fluid_benchmarking/{estimators,evaluation,irt_utils,indexing}.py`: MAP/MLE ability estimation, Fisher-information item selection, 2PL fitting via py-irt; HF dataset with pre-fit IRT params for 6 OLL benchmarks and 102 LMs (`allenai/fluid-benchmarking`). Clean, small, Apache. | acquisition; few-shot θ update |
| **Peiyu-Georgia-Li/ATLAS** https://github.com/Peiyu-Georgia-Li/ATLAS — ICML 2026 spotlight | 9★, 2026-07-12, none | 3PL chunked calibration + mean-sigma linking, WLE scoring, max-Fisher CAT, p-IRT accuracy; ~4k models × 5 benchmarks of item params in CSV. | reference; R pipeline |
| **adkipnis/metabench** https://github.com/adkipnis/metabench — ICLR 2025 | 9★, 2026-02-25, NOASSERTION | IRT item selection over >5000 LLMs; R analysis; data on Zenodo. | reference only (R) |
| **rvivek3/AnchorPoints** https://github.com/rvivek3/AnchorPoints — EACL 2024 | 8★, 2024-06-25, none | Anchor-points weighted/predictor code; "Estimating Model Predictions" baselines (nearest source neighbor). | few-shot via nearest-neighbour subjects |
| **Joe-Hall-Lee/PSN-IRT** https://github.com/Joe-Hall-Lee/PSN-IRT — AAAI 2026 oral | 3★, 2026-01-16, none | Neural 4PL IRT (`train.py`); ships 12 LLM × 41,871 item response matrix and 4PL item params. | neural IRT reference; extra training data (check inventory rules) |
| **richardzhuang0412/EmbedLLM** https://github.com/richardzhuang0412/EmbedLLM — ICLR 2025 spotlight | 34★, 2025-09-25, none | `algorithm/mf.py` (MF with noise regularisation, 232-d model embeddings), `algorithm/knn.py`; question-embedding tensors; HF dataset RZ412/EmbedLLM (correctness of many LLMs on many Qs). Correctness forecasting = this task minus cold-start. | zero-budget CF; routing-style item embeddings |
| **Mercidaiha/IRT-Router** https://github.com/Mercidaiha/IRT-Router — ACL 2025 | 36★, 2025-02-19, none | M-IRT/N-IRT with BERT query embeddings → item params; checkpoint included. | cold-start item params from text |
| JianhaoChen-nju/IrtNet (arXiv 2510.00844) | 2★, 2025-10-17 | MoE over model embeddings + query difficulty/discrimination. | reference |
| **dariakryvosheieva/agent-psychometrics** https://github.com/dariakryvosheieva/agent-psychometrics (arXiv 2604.00594) | 5★, 2026-08-31, MIT | IRT + rich task features; experiments for **new tasks, new agents, new benchmarks (held-out benchmark!)**, adaptive testing with IRT prior; decomposes agent ability into LLM + scaffold components (maps to PAIEC `harness`/`normalized_name` fields). Uses py-irt fork + sentence embeddings + LLM-judge features. | closest published analogue of the exact cold-start setting |
| **eliyahabba/growing-pains** https://github.com/eliyahabba/growing-pains (arXiv 2604.12843) | 2★, 2026-08-11, none | MIRT 2PL with fixed-parameter anchor calibration to link a *new benchmark* onto an existing scale with ~100 anchor items; `src/irt/` engine, `calibration.py`. Exactly the "new benchmark, few labels" linking problem. | few-shot benchmark linking |
| aaron-jx-li/DualEval https://github.com/aaron-jx-li/DualEval | 2★, 2026-06-26, MIT | Joint model–item calibration across binary + continuous signals; JSONL in, item params out. | reference |
| Rorschach1989/efficient-lm-eval https://github.com/Rorschach1989/efficient-lm-eval (arXiv 2510.04051) | 2★, 2025-10-07, MIT | LEGO-IRT: Rasch + continuous metrics + multi-benchmark (`LEGO-IRT-MB`). | multi-benchmark IRT |
| sambowyer/mrmr_eval https://github.com/sambowyer/mrmr_eval (NeurIPS 2026) | 2★, 2026-10-02, MIT | mRMR coreset + ridge/kernel-ridge regression on score matrices; fork of BenchPred. | subset selection |
| LoryPack/ReferenceInstancesPredictability https://github.com/LoryPack/ReferenceInstancesPredictability ("100 instances is all you need", Pacchiardi et al.) | 2★, 2024-10-16 | Predict instance-level success of a new LLM from ~100 reference instances + OpenAI embeddings; generic assessors trained across LLMs. | few-shot + item embeddings |
| JFMandujanoR/Laplace-PSN-IRT | 1★, 2026-07-27 | Laplace UQ for neural IRT. | calibration of IRT outputs |
| CyberAgentAILab/M3IRT | 1★, 2026-04-02, MIT | Multimodal IRT. | reference |
| smolix/benchmark-selection https://github.com/smolix/benchmark-selection | 7★, 2026-05-04, Apache-2.0 | Submodular benchmark selection + assembled score matrices. | side data |
| MilkThink-Lab/Awesome-Benchmark-Compression https://github.com/MilkThink-Lab/Awesome-Benchmark-Compression | 3★, 2026-10-14 | Curated catalogue (survey arXiv 2610.18248) of ~70 benchmark-prediction/adaptive-eval papers with code links; this is where most rows above came from. | literature map |

Papers with no public code found: C-MIRT "LLM Evaluation on Unseen Questions" (arXiv 2608.22295, text embedding → latent requirement vector; the closest method description to this task), Zeng & Papailiopoulos "You Don't Need to Run Every Eval" (arXiv 2606.24020; rank-2 logit-space completion of a model×benchmark score matrix; repo URL *unverified*), "Beyond Scores" (arXiv 2604.12191, cross-benchmark item prediction AUC 0.77–0.86; code *unverified*), Flash-HELM / "Efficient Benchmarking of LMs" (Perlitz; no repo found).

## 5. Collaborative filtering / matrix completion

| Tool | Verified | Fit | Caveats |
|---|---|---|---|
| **implicit** https://github.com/benfred/implicit (`implicit` 0.7.3) | 3830★, 2026-05-08, MIT | Fast ALS/BPR/logistic MF on sparse matrices; numpy/scipy only at inference. Good for a quick low-rank subject×item baseline. | No side information → useless for unseen items/benchmarks alone; combine with item-feature regression. |
| **LightFM** https://github.com/lyst/lightfm (1.17) | 5116★, 2024-07-24, Apache-2.0 | Hybrid MF with **user and item feature matrices** (cold-start by design: item embedding = sum of feature embeddings); logistic loss on binary outcomes. Can take embedding-derived item features + subject metadata features. | Dormant (last release 2023); compiles C; features must be sparse one-hot/bucketised (dense 1536-d embeddings need bucketising or PCA). |
| Cornac https://github.com/PreferredAI/cornac (3.0.1) | 1060★, 2026-09-14, Apache-2.0 | Many MF/CF models incl. text-aware (CTR, ConvMF, HFT) and cold-start-capable models; actively maintained. | Heavier API; recommendation-oriented metrics. |
| Surprise https://github.com/NicolasHug/Surprise (`scikit-surprise` 1.1.5) | 6819★, 2026-05-30, BSD-3 | SVD/SVD++/KNN baselines for explicit ratings. | No side info; rating-scale oriented. |
| RecBole https://github.com/RUCAIBox/RecBole (1.2.1) | 4588★, 2025-02-24, MIT | Large zoo incl. context-aware/FM/DeepFM models. | Too heavy for a submission; fine offline. |
| torch_measure `logistic_fm`, `ncf`, `bifactor` | (see §2) | Logistic factorization / neural CF written for response matrices. | Preferred over generic recsys libs here. |
| Hand-rolled: logistic low-rank MF with item side-regression in PyTorch (~100 lines) | — | Lets you implement exactly: logit p = θ_s·a_i − b_i with a_i,b_i = f(embedding_i, item_features) and θ_s = g(subject metadata) + free residual; this is what C-MIRT/reeval/AmortizedIRT do. | You own the code; easiest to make Codabench-safe. |

## 6. Active learning / acquisition (streaming accept/reject with budgets 1–31)

| Tool | Verified | Fit | Caveats |
|---|---|---|---|
| **catsim** / **fluid-benchmarking** / **torch_measure.cat** | §3/§2 | Fisher information at current θ̂ is the natural "buy" score; combine with a dynamic threshold so you spend the budget across the stream (secretary-style). | All pool-based; need streaming wrapper. |
| scikit-activeml https://github.com/scikit-activeml/scikit-activeml (1.0.1, pushed 2026-10-20) | 204★, BSD-3 | Pool **and stream-based** strategies (uncertainty, QBC, expected error reduction, stream budget managers). Stream module is directly relevant. | sklearn-estimator oriented; wrap your predictor. |
| modAL https://github.com/modAL-python/modAL (1.6.2) | 2373★, last push 2024-02-26 | Simple uncertainty/QBC; easy to copy. | Pool-based, dormant. |
| ALiPy https://github.com/NUAA-AL/ALiPy | 903★, 2025-07-23, BSD-3 | Many query strategies incl. cost-sensitive. | Older API. |
| libact https://github.com/ntucllab/libact | 791★, 2026-10-12, BSD-2 | ALBL (active learning by learning) bandit over strategies. | Needs compiled deps. |
| BAAL https://github.com/baal-org/baal (2.1.0) | 936★, 2025-12-03, Apache-2.0 | Bayesian AL (MC-dropout/BALD) for torch models. | Overkill for a 31-label budget. |
| GPyTorch https://github.com/cornellius-gp/gpytorch (1.15.2) / BoTorch https://github.com/pytorch/botorch (0.18.1) | 3928★ / 3630★, MIT, active | GP over subject θ or over item embeddings; BoTorch acquisition functions (expected improvement is not the right objective; use mutual information / expected Brier reduction instead). gp-IRT in tinyBenchmarks is a lightweight GP alternative. | heavy; python ≥3.11 for botorch. |
| gabrielleberrada/scaling-up-active-testing (NeurIPS 2025), amazon-science/NLPActiveTesting (2026) | 1★ / 0★ | Active *testing* (choose which test items to label to estimate accuracy), same objective family as PAIEC acquisition; AIPW/LURE estimators. | Research code. |

Acquisition design note: because labels are only revealed for the *same subject* and predictions are per-item, the value of a label is (i) updating the subject's θ on this benchmark (Fisher info) and (ii) calibrating the benchmark-level offset (the empirical-mean signal). A simple strong policy: buy when item discrimination × p(1−p) is above a running quantile that keeps the expected buy count = remaining budget.

## 7. Calibration (Brier rewards it)

| Tool | Verified | Fit |
|---|---|---|
| **netcal** https://github.com/EFS-OpenSource/calibration-framework (`netcal` 1.4.0, 2026-04) | 380★, 2026-04-16, Apache-2.0 | Temperature/Platt/Beta scaling, histogram binning, isotonic, BBQ, ENIR; ECE/MCE/NLL metrics; Bayesian variants. Use for post-hoc calibration of the fitted model's logits, fit per budget level on held-out *benchmarks* (benchmark-level CV, to match the test cold start). |
| gpleiss/temperature_scaling | 1180★, 2025-07-26, MIT | Reference implementation; trivially re-implemented. |
| scikit-learn `CalibratedClassifierCV`, `IsotonicRegression`, `brier_score_loss` | — | Likely already in the sandbox; isotonic on benchmark-grouped CV is the minimal safe option. |
| torch_measure `metrics/calibration`, `cat/calibration` | §2 | Organizer-consistent ECE/Brier utilities. |

Brier-specific tips: shrink toward the benchmark base rate under uncertainty (optimal under squared loss), never output 0/1 from tiny samples (the empirical-mean baseline does), and clip less aggressively than BLE's [0.02, 0.98] only if calibrated.

## 8. Text features for item difficulty (brief)

- Use the shipped `measurement-db-embed` vectors (text-embedding-3-small, 1536-d) for training items; for unseen test items either call OpenAI embeddings at runtime (allowed provider; BLE already does this in `embedding_index.py`) or ship a local `sentence-transformers` model via `models.txt` (whitelisted IDs include the basic BERT/ALBERT family; `all-MiniLM-L6-v2`-class models; *verify against the full list in `check_submission_zip.py`*). sentence-transformers 6.1.0 (19,200★, Apache-2.0).
- Cheap non-embedding features: length, numeric density, code/markdown presence, answer-format (MC vs free), `item_features` fields, benchmark_id-level aggregates of revealed labels.

## 9. Comparison table (usability in a Codabench submission)

| Tool | Sub-problem | pip-installable | Runtime deps | Maintained | License | Verdict |
|---|---|---|---|---|---|---|
| torch_measure | zero-budget, few-shot, acquisition, calibration | yes (0.1.0) | torch, pyro, tabpfn | yes (organizers) | MIT | **Core**; fit offline, consider vendoring only needed modules |
| py-irt | item/subject calibration | yes | torch, pyro | yes | MIT | **Core offline**; export params |
| tinyBenchmarks (p-IRT/gp-IRT) | few-shot combiner | git only | numpy, scipy | yes | MIT | **Copy the estimator (~50 lines)** |
| fluid-benchmarking | acquisition, θ MAP | git only | numpy; py-irt optional | moderate | Apache-2.0 | **Copy Fisher/MAP code** |
| catsim | acquisition, θ estimation | yes | numpy, scipy | yes | BSD-3 | Good light alternative |
| BenchPred | baselines (AIPW, anchors, PIRT) | `-e .` | torch | yes | MIT | Offline benchmarking of methods |
| reeval / irsl / agent-psychometrics / growing-pains | method references for amortized difficulty, new-benchmark linking | no | — | research | none/MIT | Read & reimplement |
| LightFM | hybrid CF with side info | yes (old) | C ext | dormant | Apache-2.0 | Optional baseline |
| implicit | plain MF | yes | numpy | yes | MIT | Quick baseline |
| netcal | calibration | yes | numpy, scipy, torch (some) | yes | Apache-2.0 | **Core** (or sklearn isotonic) |
| scikit-activeml | streaming AL | yes | sklearn | yes | BSD-3 | Optional; stream strategies |
| GPyTorch/BoTorch | GP priors | yes | torch | yes | MIT | Optional |
| sentence-transformers | item embeddings | yes | torch, transformers | yes | Apache-2.0 | Only if local embedding needed |
| EmbedLLM / PSN-IRT data | extra response matrices | no | — | research | none | Check "benchmark inventory" rule before using as training data |

## 10. Recommended stack

1. **Data/feature layer**: `paiec_baseline/tools/prepare_data.py` payload (parquet tables + text-embedding-3-small vectors) → pandas/pyarrow; derive item features (embedding PCA 64–256 dims + hand features + benchmark-level stats) and subject features (provider, release_date numeric, harness, reasoning_effort one-hots; target-encoded historical θ from training benchmarks).
2. **Zero-budget model (cold benchmark)**: amortized 2PL/3PL IRT: θ_s from subject history (free parameter per subject, prior from metadata), a_i, b_i from item features via a small MLP (torch_measure `AmortizedIRT`, or reeval-style ridge/MLP regression on top of py-irt-calibrated difficulties). Add a gradient-boosted per-cell classifier (LightGBM/TabPFN via `torch_measure.TabPFNPredictor`) on [subject feats, item feats, subject×benchmark-cluster interactions] and ensemble in logit space. Validate with **leave-benchmark-out CV** (this is what agent-psychometrics' `experiment_new_benchmarks` and growing-pains do).
3. **Few-shot adaptation (budgets 1–31)**: Bayesian update of (θ_s on this benchmark, benchmark offset) given revealed labels: MAP θ via fluid-benchmarking/catsim estimators with a prior from step 2, and a gp-IRT/p-IRT style shrinkage (tinyBenchmarks) toward the observed mean. Never return raw means.
4. **Acquisition (`labeling.py`)**: streaming max-Fisher-information with a budget-pacing threshold (catsim `MaxInfoSelector` logic or torch_measure `cat.fisher`), fallback to forced buys when `items_remaining <= labels_remaining` (as BLE does).
5. **Calibration**: per-budget isotonic or Beta scaling (netcal / sklearn) fit on leave-benchmark-out predictions; shrink toward base rate.
6. **Packaging**: fit everything offline, ship `.npz/.pt/.parquet` params; keep runtime imports to numpy/pandas/pyarrow (+torch only if you confirm it's in the image or allowed via `requirements.txt`); run `python check_submission_zip.py` and `tools/smoke_test.py --mock`.
7. **Optional LLM component**: keep BLE (or a cheaper prompt) as a feature for items whose embedding is far from all training items, since the API providers are allowed.

## Sources

[AIMS competition page](https://aims.stanford.edu/competition), [competition news](https://aimslab.stanford.edu/news/predictive-evaluation-competition-opens), [measurement-db](https://huggingface.co/datasets/aims-foundations/measurement-db), [paiec_baseline](https://github.com/aims-foundations/paiec_baseline), [torch_measure](https://github.com/aims-foundations/torch_measure), [reeval paper](https://arxiv.org/abs/2503.13335), [reeval code](https://github.com/aims-foundations/reeval), [irsl](https://github.com/aims-foundations/irsl), [py-irt](https://github.com/nd-ball/py-irt), [tinyBenchmarks](https://github.com/felipemaiapolo/tinyBenchmarks), [fluid-benchmarking](https://github.com/allenai/fluid-benchmarking), [Fluid paper](https://www.arxiv.org/pdf/2509.11106), [BenchPred](https://github.com/socialfoundations/benchmark-prediction), [ATLAS](https://github.com/Peiyu-Georgia-Li/ATLAS), [metabench](https://github.com/adkipnis/metabench), [AnchorPoints](https://github.com/rvivek3/AnchorPoints), [Anchor Points paper](https://aclanthology.org/2024.eacl-long.95/), [PSN-IRT](https://github.com/Joe-Hall-Lee/PSN-IRT), [EmbedLLM](https://github.com/richardzhuang0412/EmbedLLM), [IRT-Router](https://aclanthology.org/2025.acl-long.761.pdf), [agent-psychometrics](https://github.com/dariakryvosheieva/agent-psychometrics), [growing-pains](https://github.com/eliyahabba/growing-pains), [100 instances](https://github.com/LoryPack/ReferenceInstancesPredictability), [C-MIRT](https://arxiv.org/abs/2608.22295), [You Don't Need to Run Every Eval](https://arxiv.org/pdf/2606.24020), [Beyond Scores](https://arxiv.org/pdf/2604.12191), [Awesome-Benchmark-Compression](https://github.com/MilkThink-Lab/Awesome-Benchmark-Compression), [catsim](https://github.com/douglasrizzo/catsim), [EduCDM](https://github.com/bigdata-ustc/EduCDM), [netcal](https://github.com/EFS-OpenSource/calibration-framework), [scikit-activeml](https://github.com/scikit-activeml/scikit-activeml), [LightFM](https://github.com/lyst/lightfm), [implicit](https://github.com/benfred/implicit), [Cornac](https://github.com/PreferredAI/cornac).
