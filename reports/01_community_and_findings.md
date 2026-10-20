# PAIEC (NeurIPS 2026): public footprint and findings to date

*Research date: 2026-10-20. Everything below was fetched directly (web fetch, curl, GitHub API, HF API, Codabench API) or found via web search; where a source was inaccessible it is marked as such.*

## 0. Executive summary

- **Official discussion is thin and entirely organizer-owned**: the competition page, one AIMS news post (15 Aug 2026), the NeurIPS competitions announcement (28 Jul 2026), a Google Doc on data curation (5 Sep 2026), a benchmark-inventory Google Sheet, the Codabench listing, and an OpenReview venue group. **No paper or arXiv preprint describing PAIEC or measurement-db exists yet**; Sang Truong's publication page lists a NeurIPS 2026 bibtex entry "Predictive AI Evaluation Competition" with no PDF.
- **No public leaderboard.** The Codabench API returns `leaderboards: []`, `hide_score_output: true`; the forum is login-gated. Formative feedback is private per submitter. As of 2026-10-20: **206 participants, 3,268 submissions**.
- **Zero social-media footprint found** (X, Reddit, HN, Bluesky, LinkedIn beyond one repost snippet, Chinese platforms). The only HN thread is about the AIMS textbook (16 pts, 1 comment).
- **The substantive public knowledge comes from ~10 participant GitHub repos** (several with full technical-report drafts). They converge on: (a) the organizers' empirical-mean baseline scores ≈0.216 on platform (≈0.22–0.26 locally) and collapses at budget 1; (b) simple Beta-smoothed/hierarchical IRT models reach ≈0.18–0.19 mean formative ALC, with single-draw SD ≈0.01–0.03 (so individual formative scores are nearly uninformative); (c) pooling other subjects' revealed labels on the same benchmark is the dominant lever; (d) item-text features and LLM difficulty judgements do **not** transfer across benchmarks; (e) local LOBO rankings on the public pool did not predict hidden-test rankings (Yamasaki).
- **There was a predecessor**: the Spring 2026 "Predictive AI Evaluation Challenge" (Codabench 15934, Apr–Jun 2026, 166 participants, 5,369 submissions, NLL/AUC metric, tied to Stanford CS321M and the COLM 2026 AIMS workshop). Top scores: NLL −0.56 / AUC 0.74.

---

## 1. Official sources

### 1.1 Competition page — https://aims.stanford.edu/competition (aimslab.stanford.edu redirects here)
Organizers listed: Sang Truong, Wale Salaudeen (MSR/UIUC), Nhi Truong, Serena Wang (UBC/Google), Angelina Wang (Cornell), Luke Guerdan (CMU), Ziang Xiao (JHU), Xing Xie (MSR), Nick Haber, Sanmi Koyejo. Key content:
- Task: P(correct) for a *subject* (model + harness + harness version + reasoning effort) on an *item*, benchmark-level cold start; held-out items come from benchmarks in a private test pool; benchmarks randomly assigned to train/test pools *before* curation.
- Metric: Brier ALC = 0.1·B0 + 0.2·B1 + 0.2·B3 + 0.2·B7 + 0.2·B15 + 0.1·B31 (log2(1+n) axis, trapezoid), equal weight per subject–benchmark pair; ECE (10 bins) diagnostic only. Page gives **no numeric baseline**; learning-curve figure is illustrative.
- Test-time adaptation: pairs with ≥80 distinct items are split 50/50 into acquisition/evaluation pools; optional `labeling.py::acquisition_function`, else random reveal.
- Formative evaluation: freshly sampled subset of up to 1,000 subject–item pairs per submission; summative uses a common hidden subset. "Individual feedback is private to the submitter and organizers. Aggregate leaderboard visibility is announced separately."
- Deadlines: final codebase + technical report **30 Oct 2026 23:59 AoE**; report 4–8 pages NeurIPS template via OpenReview (`NeurIPS.cc/2026/Workshop/PAIEC`), must include selected Codabench submission ID. Session Fri 11 Dec 2026, 08:00–10:15 Sydney: 5 orals + poster session. No formal peer review planned. Organizers will rerun released methods.
- Rules: teams ≤6; only inventory-listed data for competition-specific training; runtime network limited to Anthropic/OpenAI/Google/xAI/DeepSeek/Mistral/Together/Groq inference APIs (no web search); 8 h per run; ≤5 model repos, ≤300B params each, ≤1,024 GB total; disclosure of LLM coding assistants required.
- Links extracted from the HTML: registration Google Form; benchmark inventory sheet `docs.google.com/spreadsheets/d/1T69QyIZJRWX3d-Tmuh_lcFwtpqv9ytcLxfLWKn6gopI`; methodology doc `docs.google.com/document/d/1LdUEiUG5tLhDdnIAtm4YKSf4w3LFb2HmI_ahiiuoRLU`; two Discord invites (`discord.gg/W2rmjJdcm`, `discord.gg/DQZgXCEJjc`); email aimslab@cs.stanford.edu.
- Page text contains **no** "prize"/"award" strings (grep confirmed).

### 1.2 AIMS news post — https://aims.stanford.edu/news/predictive-evaluation-competition-opens (15 Aug 2026)
Launch announcement: opened 15 Aug 2026; formative automatic feedback during development; summative = automatic predictive assessment + manual grading of codebase and report; report quality "an important factor". No numbers.

Note: search snippets show an earlier/variant AIMS wording "opened 15 August 2026 and remains open until **20 November 2026**" and a page titled "The Predictive AI Evaluation Challenge" with "deadline TBD", Track 1 = response prediction/matrix completion with AUC-ROC, torch_measure, three submission tracks (CSV, Docker, LLM featurizer). Those URLs now 404 or redirect to the current page; Wayback was offline when checked. Treat as stale predecessor material (see §6).

### 1.3 NeurIPS blog — https://blog.neurips.cc/2026/07/28/neurips-2026-competitions-announced/ (28 Jul 2026)
PAIEC is one of 16 accepted NeurIPS 2026 competitions (category "Foundation Models, Reasoning, and AI Evaluation", with Steerability Challenge, AIMO Interpretability, QuantiPhy). Only name + link. Organizers may publish post-competition analyses in the Datasets & Benchmarks track or a PMLR volume.

### 1.4 Codabench — https://www.codabench.org/competitions/17828/
The HTML page is a JS SPA. Public API `GET /api/competitions/17828/` (fetched 2026-10-20):
- Title "NeurIPS Predictive AI Evaluation Competition", owner `aims-lab` / "AIMS Foundations", created **2026-08-15**.
- One phase (id **29785**, "Competition"), 2026-08-15 → **2026-10-31T11:59Z**, status Current, `execution_time_limit: 28800` s, task "Amortized Prediction"; ingestion/scoring programs and starting kit uploaded 2026-09-16 (version tag `5f7b7bf`), not public.
- `max_submissions_per_day: 50`, `max_submissions_per_person: 1000`; `hide_score_output: true`, `hide_prediction_output: true`; `leaderboards: []` (none defined); `enable_detailed_results: false`.
- **participants_count: 206, submissions_count: 3268.**
- Docker image `us-west1-docker.pkg.dev/soe-cs321m-competition/paiec/submission:latest`; forum id 17518 (login-gated).
- Phase/dataset/results endpoints require auth. Overview text: "Can we predict how AI models will perform on unseen benchmark items from partial observations?"

### 1.5 OpenReview — https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/PAIEC
API shows venue "Predictive AI Evaluation Competition at NeurIPS 2026" / "PAIEC 2026", Sydney, contact aimslab@cs.stanford.edu, `public_submissions: false`, start_date ≈ 15 Sep 2026. No submissions visible.

### 1.6 Organizer bibliography — https://ai.stanford.edu/~sttruong/publications
Lists `@inproceedings{truong2026predictivecompetition}` "Predictive AI Evaluation Competition", NeurIPS 2026, authors: Sang Truong, Olawale Salaudeen, Nhi Truong, Serena Wang, **Yegor Denisov-Blanch, Fagun Patel** (not on the website list), Angelina Wang, Luke Guerdan, Ziang Xiao, Xing Xie, Nick Haber, Sanmi Koyejo. No PDF/arXiv link. The CV PDF also lists a 2026 preprint "The Predictive Evaluation Competition".

### 1.7 "Building an AI Measurement Data Bank" Google Doc (5 Sep 2026)
- Explicitly ties curation to the competition: contributors "will also be eligible for the **Data Curator Award** under the NeurIPS 2026 Predictive AI Evaluation Competition" (the only prize-like item found anywhere).
- "For the competition, eligible base models are currently restricted to releases from 2025–2026"; current coverage: Anthropic, Google, Moonshot, OpenAI, Zhipu, Alibaba, ByteDance, DeepSeek, xAI.
- Inventory built by scraping OpenReview for all papers at NeurIPS/ICLR/ICML, classifying benchmarks; schema definitions (response = grade for (subject, item, trial, interactor, test condition); null = attempted-but-ungraded); worked example SWE-rebench (Qwen3-Coder-480B via OpenHands v0.54.0).

### 1.8 Benchmark inventory sheet
1,261 rows (benchmark slug, authors, code/dataset URLs, OpenReview id, venue). Venue mix: ICML 2026 (217+18), NeurIPS 2025 D&B (202+18), ICLR 2026 (195+8), NeurIPS 2024 D&B (174+21), ICLR 2025 (163+23+8), ICML 2025 (77), ICLR 2024 (56+18), ICML 2024 (38). `curation_status` filled for only 4 rows ("success"); "released on GH/HF" columns empty.

---

## 2. Baseline repo — https://github.com/aims-foundations/paiec_baseline
- 2 stars / 3 forks / 5 commits; pushed 2026-10-15; MIT; cites "PAIEC Predictor Baselines (2026)" with the 10 organizers.
- Three predictors: **Bayesian Linguistic Evaluator (BLE)**: an LLM (default `openai/gpt-5.6-luna`, Responses API, reasoning_effort medium, ≤10 turns, 7 keyword/table retrieval tools + optional embedding search with `text-embedding-3-small`; beliefs clipped to [0.02, 0.98]; uncertainty acquisition when |p−0.5|<0.15); **Empirical Mean** (mean of acquired labels for the exact subject–benchmark pair, 0.5 otherwise; stdlib only); **Empirical Mean + BLE acquisition**.
- **No Brier ALC numbers are reported anywhere in the repo.** README states the acquisition benefit "requires empirical validation."
- `submit_api.py` for Codabench; `tools/check_submission_zip.py`, `run_smoke_test.py`.
- Issues (all open, all by `couldbeme`, 2 Oct 2026): #1 `prepare_data.py` aborts when one benchmark's embeddings are stale; #2 `test_mean_ble_acquisition` fails on macOS (symlinked TMPDIR); #3 repo copies of `streaming_ingestion.py`/`check_submission_zip.py` older than starting kit's. No comments.
- Participants' measurements of the empirical-mean baseline: platform **0.2161 ± 0.0215 (n=20; 95% CI [0.2067, 0.2255])** (Yamasaki); local 0.2213 (Yamasaki), 0.2311 (lucaluo925), 0.2453 (doctawho42), 0.2625 (aghasalim), 0.2635 (femartip). All note it collapses at budget 1 (B1 ≈ 0.31–0.39) because one label makes it predict 0/1.

---

## 3. measurement-db

### 3.1 HF dataset — https://huggingface.co/datasets/aims-foundations/measurement-db
- Gated (`gated: auto`, manual review per participants), created **2026-08-15**, last modified **2026-10-07**, 3,306 downloads/month, 25 likes, tags cc-by-sa-4.0, 1M–10M rows, tabular+text.
- Card: "curated collection of standardized, item-level AI evaluation results"; **571,921 recorded observations preserved in migration; 249 validated benchmark migrations; 6 benchmarks on schema v3; 1,191 archived source files (10.4 GB); total 740 GB**. Per-benchmark dirs with `formatted_tables/{benchmarks,items,responses,subjects,traces}.parquet`, `metadata.yaml`, `characterization.yaml`, `data_manifest.json`, `raw/`. Three benchmarks (ImageNet-Hard, VisualMemory, SWE-Together) withheld 2026-10-04 for source reproducibility; filter `release == "public"`. Alpha-SQL grading IDs changed in 2026-10-02 refresh. Citation: Truong N., Truong S.T., Koyejo S. (2026), *The AI Measurement Data Bank*.
- Discussions: PR #3 "Publish validated tabular benchmark collection on main" by sangttruong, merged **2026-10-06** (the big expansion); bot parquet conversion 2026-08-18.
- File listing shows 100,000+ files. Pinned revision used by several participants: `54074453da7b9911be0f48e4c560be547bc84d21`.
- Sibling datasets in the org: `measurement-db-gallery` (4,122 downloads, gated), `measurement-db-embed`, `safety-irt`, `ecosystem`, `eval_traces`. `measurement-db-public` resolves to the same dataset.

### 3.2 Website — https://aims.stanford.edu/measurement-db
Headline counts: **270 benchmarks (264 automated, 6 human-centered), 8.4M items, 781 subjects, 92.1M responses**; 35 releases tagged 2026, 80 tagged 2025. (These differ from the HF card's 571,921 "observations" figure, which refers to the validated migration.)

### 3.3 GitHub — https://github.com/aims-foundations/measurement-db
285 commits, 2 stars; six public example builders (MathArena, MMDocRAG, Multi-SWE-bench, REAL, ResearchCodeBench, SWE-rebench); schema v3; "download all tables from the same HF revision." No findings stated.

### 3.4 What participants report about the data (initial 6-benchmark release, Aug–Sep 2026)
From aghasalim's report and Yamasaki's draft (consistent):

| benchmark | binary? | subjects | items | responses | pairs ≥80 items | accuracy |
|---|---|---|---|---|---|---|
| matharena | partly (judge non-binary) | 104 | 1,751–1,755 | 92,861 | 81 | 0.65–0.70 |
| mmdocrag | no (fractional) | 68 | 2,000 | — | 68 | 0.78 |
| multi_swebench | yes | 82 | 2,126 | 57,975 | 82 | 0.148 |
| real_webagents | yes | 33 | 233 | 4,449 | 26 | 0.36–0.37 |
| researchcodebench | yes | 31 | 212 | 6,784 | 31 | 0.35–0.36 |
| swe_rebench | yes | **1** | 6,306 | 67,074 | 1 | 0.47–0.48 |

Observations: only 22 of 287 subject ids recur across benchmarks (femartip); a third of matharena items have no task text ("See image" / prompt fragments) and 2026 contests were only run against 2026 models, confounding naive difficulty (doctawho42); no item content repeats across benchmarks (Yamasaki); mmdocrag/swe_rebench groups exceed the 1,000-item draw rule (femartip). Later expansion: paiksca trains a prior on **182 public benchmarks**; 909tttttough reference AlgoTune, AppWorld, CMMLU as inventory benchmarks; doctawho42 evaluated refitting on the "expanded public pool" (no gain).

The Spring-2026 version of measurement-db was different: **16 benchmarks, 909 subjects, 70,873 items, 3.58M–4.4M binary responses** (galarcon0803, antoinemaechler-X READMEs), with a different schema (`benchmark/condition/subject_content/item_content`).

---

## 4. Papers / preprints
- **None about PAIEC or measurement-db.** arXiv API queries for "Predictive AI Evaluation Competition", "measurement-db", "AI Measurement Data Bank", "Predictive Evaluation Competition" return nothing relevant.
- Closest organizer work (precursors explicitly in the AIMS lineage):
  - *Reliable and Efficient Amortized Model-based Evaluation* (Truong, Tu, Liang, Li, Koyejo; ICML 2025, PMLR 267:60238; arXiv 2503.13335): IRT with text-based difficulty amortization, adaptive testing; 22 benchmarks, 183 LMs; code `aims-foundations/reeval`. https://proceedings.mlr.press/v267/truong25c.html
  - *Item Response Scaling Laws* (Truong, Tu, Schaeffer, Koyejo; ICML 2026, PMLR 306:122297; arXiv 2606.07616): factorizes scaling-law estimation O(M×N)→O(M+N); ~6,612 checkpoints, 37,682 questions, 10 benchmarks. https://proceedings.mlr.press/v306/truong26b.html
  - *AI Evaluation Should Require Standardized Item-Level Data Releases* (Jiang, Zhang, Zhu, Bai, Truong, Yi, Koyejo, Xie, Xiao; arXiv 2604.03244, Feb/Apr 2026): position paper; introduces OpenEval (10M responses, 155k items). Overlaps PAIEC organizers Xiao/Xie.
  - *A Measurement Science Roadmap: From Human Assessment to AI Evaluation* (Truong, Goodman, Brunskill, Domingue, Haber, Koyejo; preprint 25 Apr 2026). https://ai.stanford.edu/~sttruong/assets/pdf/roadmap.pdf
  - *Strategic Evaluation: Incentivizing AI Capability Coverage with Private Benchmarks* (Truong, S. Wang, Haber, Koyejo; NeurIPS 2026 poster; arXiv 2605.12530): private/randomized benchmarks framing relevant to the "private test pool" design.
  - *Guardians of the Measurement* (Truong, S. Wang, Koyejo; 2026 preprint listed on CV; no text found).
  - AIMS textbook *AI Measurement Science* (Truong & Koyejo, updated 7 Sep 2026; ch. 3 "prediction-powered evaluation … cold-start prediction of unseen models and items"; ch. 8 "Predictability"). Does not mention the competition or Brier ALC.
  - `aims-foundations/torch_measure` (20 stars): IRT/factor/amortized inference toolkit; no competition mention.
- Not organizer-affiliated but cited by searches as related: tinyBenchmarks (Polo et al.), "100 instances is all you need" (arXiv 2409.03563), contextual multidimensional IRT for unseen questions (arXiv 2608.22295), MedIRT, IRT-Router. None reference PAIEC.

---

## 5. Participant repositories and self-reported results (NeurIPS PAIEC, competition 17828)

All are self-reported; no official numbers exist. Formative draws are ~8–9 pairs per submission (≤1,000 subject–item pairs), redrawn each time.

| Repo | Who | Method | Key numbers |
|---|---|---|---|
| https://github.com/lucaluo925/paiec-2026 (pushed 2026-10-04) | "lucaluo925" (Claude-assisted) | Online hierarchical IRT: logit = a·θ_s + v_s − β − d_i − Σw_f; θ_s from public pool or regressed from provider/date/params; one Gaussian posterior per anonymous benchmark_id, assumed-density filtering per label; mixed with base rate per budget | LOBO (147 pairs, 19,621 targets): **ALC 0.16875** (B0 .2211, B1 .1932, B3 .1771, B7 .1593, B15 .1412, B31 .1248) vs empirical mean 0.23110; own-pair-labels-only 0.19450; cross-fitted 0.16515; folds 0.1509/0.1943/0.2728(swe_rebench, 1 pair)/0.1860. Same zip submitted 9× live: **0.1685–0.2062, mean 0.18817, SD 0.0113**. Reports "gap to the leading entry 6.0 SD" (implies some ranking visible to participants). 4 pre-registered variants falsified. |
| https://github.com/doctawho42/PAIEC (pushed 2026-10-20) | Nikita L. Polomoshnov (MSU) | "hier-ship": hierarchical IRT refit at each budget; attribute prior (provider/date/size/effort/harness) + new-benchmark level prior N(−2.5, 2.5²); replica of organizers' streaming evaluator; "pair-rate oracle" decomposition | Formative runs: ALC **0.2113 → 0.1926 → 0.1817** (final archive-3, frozen 2026-10-06); oracle V 0.1536/0.1797/0.1608; 78–89% of remaining error is item-level. Ceilings (public data): always-0.5 0.2500, pair true accuracy 0.176, true per-item prob 0.0433. Ladder: Beta(2,2)-smoothed mean 0.2089; pooled difficulty + attribute prior 0.1813; run-time predictor 0.1898 (0.1725 if budgets interleave per pair; **evaluation order worth 0.017; "nothing in the published rules settles which order the evaluator uses"**). Text→difficulty LOBO correlations 0.14/0.23/0.09/−0.22/0.16; LLM difficulty rating Pearson 0.48 on matharena, ≈0 elsewhere. Replica-run noise SD 0.023–0.030. |
| https://github.com/IvanYamasaki/paiec-yamasaki (pushed 2026-10-15) | "Team Yamasaki" (Portuguese-language bench) | Rasch-per-test-benchmark (line A) vs parameter-free Beta smoothing (J2: p0=(S+2)/(N+4) from other subjects, p=(s+2p0)/(n+2), 12 lines) | **Formative arms (hidden test, through 2026-10-15; 185 valid of 216 submissions)**: J4 0.1883 (n=39), J3 0.1888 (18), **J2 0.1892 (39)**, J1 0.1899 (12), A-v6 0.1946 (7), A-v7 0.1969 (40), empirical mean **0.2161 (20)**. All J arms beat baseline (adj. p ≤ 0.03); differences among J arms inconclusive. Best single run 0.1461 (J3), worst J 0.2341. **Local LOBO ranking (A v6 0.1640 best) did not transfer**: pool-dependent arms lost +0.016–0.029 on test; J2/J4 gained. Subject-metadata offset hurt B0 (0.318 vs 0.250). LLM "success note" (Sonnet) was the only item signal that transferred locally (0.1856→0.1626) but was dropped (API dependency). Final: J2, submission 989544 (0.1961 on its 9 pairs). |
| https://github.com/aghasalim/predictive-eval (2 stars, pushed 2026-10-11) | "aghasalim" | Shrunk logistic: ability by model name (w=0.5) + online MAP offsets for benchmark, subject, tag, question (Gaussian priors, 8 Newton steps) | Local (5 benchmarks × 4 seeds): const 0.2500, emp. mean 0.2625, v1 0.2203, v2 0.2171, v3 **0.2146**, v4 0.2143; v1 B0 = 0.2508 (ability transfer ≈ none). Platform: v1 0.1918–0.1980 (mean 0.1944, n=4), v2 0.1630–0.2133 (mean 0.1835, n=5), v3 0.1829 (n=1); 42-submission arm test of adaptation speed: no detectable difference (±0.005, SE ~0.0046). Platform per-budget (454 pairs): 0.2240/0.1994/…/0.1723(B7)/0.1595(B31). Text features hurt (+0.0071); MiniLM neighbours −0.0003 (no gain). Probe confirmed numpy/torch/sentence-transformers load in scoring container. |
| https://github.com/paiksca/paiec-2026 (pushed 2026-10-12) | Samuel Paik-Heintz (Inst. for Systems Biology; Claude-assisted) | Hierarchical logistic: μ_b (ridge on item text + harness feats, from **182 public benchmarks**), θ_s, δ_sb, run shift g, item-feature groups, integrated item residual; numpy only | Grouped 8-fold CV: smoothed label mean 0.2125; const-prior hier 0.2106; feature prior 0.2078; v2 0.2076 (hard regime 0.1985); **v3 (level centre −1.0, submitted) 0.2108 main / 0.1887 hard**. Feature-based level prior hurts with 5–20 training benchmarks, helps from ~80. Simulator matched to 55 public formative logs. |
| https://github.com/909tttttough-cyber/predictive-ai-competition-2026-909 (created 2026-10-15) | Chen Linhe, Liu Yanbo (St Petersburg State U; Codex-assisted) | V12: exact pair Beta(1,1) p=(k+1)/(n+2); V33: + conservative peer residual correction | No scores published; offline studies on AlgoTune and Multi-SWE-bench at pinned revision `5407445…`. |
| https://github.com/JadenChun5/paiec-2026 (created 2026-10-16) | "JadenChun5" | Frozen empirical-Bayes: (s+1.1814)/(n+1.1814+1.2785); 0.4803 with no labels | No scores in README ("Manuscript V4.3" not public). |
| https://github.com/femartip/predictive_ai_comp (1 star, pushed 2026-10-19; Spanish docs) | "femartip" | LightGBM "assessor" prior (R1 base rate, R2 model strength, S1 11 subject metadata feats, item text stats/bge embeddings/ADeLe demand levels) + test-time corrections T1/T2 | LOBO on 4 benchmarks: const 0.2500, official 0.2635, **labels-only B3 0.1984**, best trained prior 0.1962, oracle item-difficulty ceiling 0.1264. "Almost all improvement comes from test-time; trained priors add ≤0.002." Prior-only (no adaptation) 0.256–0.268, *worse than constant*. Protocol v2 (2026-10-15) adds ADeLe battery (13 sub-benchmarks). |
| https://github.com/SudoMishra/neurips-competition (2026-08-28) | — | starter workspace (empirical-Bayes baseline) | none |
| https://github.com/brettwagner87-cloud/infinity-analyze-paiec-v1.1 (2026-10-07) | "Infinity Analyze" | — | empty README |
| https://github.com/krosenfeld-IDM/aims-neurips-2026 | — | indexed by search as submission-tracking tooling | **now 404** (deleted/private) |
| https://github.com/jaideep1021/cs5814-coldcast (2026-09-22) | course project (CS5814) | metadata/TF-IDF/semantic logistic, "fusion" | On held-out *benchmarks* zero-label: constant 0.5 (0.2500) beat everything (metadata_logistic 0.2699, system_rate 0.2825, semantic 0.3335). |

Other evidence of participation: `mlcontests.github.io` lists it (added 28 Sep 2026, deadline 30 Oct, Codabench, no prize); `harryila/venue-radar` issue #131 (bot, 16 Sep 2026, "due Oct 31, 2026"); GitHub code search finds `aims-foundations/benchmark-caliper`, `benchmark-chisel`, `redteam-measurement` (AIMS-internal repos using the schema).

### 5.1 Cross-cutting findings participants agree on
1. Empirical-mean baseline ≈ 0.216 platform; a Beta prior alone gains ~0.03–0.04 (fixes the budget-1 collapse).
2. Reasonable hierarchical/IRT or pooled-Beta methods land at **~0.18–0.19 mean formative ALC**; the best per-draw numbers (~0.146–0.165) are luck of the draw. Single-submission SD ≈ 0.010–0.03; per-draw sample ≈ 8–9 pairs.
3. `labeled` includes outcomes from *other sampled subjects on the same benchmark*; pooling those is the main lever (lucaluo: 0.1945 own-pair-only → 0.1688 pooled; doctawho42: pooled difficulty worth ~0.02).
4. Cold-start priors from the public pool transfer poorly: B0 ≈ 0.25 for most methods; subject-metadata offsets can hurt B0 (Yamasaki J1 0.318); benchmark level does not transfer (doctawho42: levels differ by 0.81 logits).
5. Item-text features, sentence embeddings, and LLM difficulty ratings do not transfer across benchmarks (several independent negative results); LLM-generated per-item "success notes" helped locally for Yamasaki but were not shipped.
6. Hidden-run learning curves flatten after ~7 labels; local simulations overstate late-budget gains (doctawho42, aghasalim).
7. Uncertainty-based acquisition: no gain found (femartip T1-u; Yamasaki ceiling 0.0044 even with oracle difficulties).
8. Open ambiguity: evaluation/budget ordering of `predict` calls (doctawho42 estimates 0.017 ALC at stake).

---

## 6. Predecessor: Spring 2026 "Predictive AI Evaluation Challenge" (Codabench 15934)
- Codabench API: "Stanford Predictive AI Evaluation Competition", created 2026-04-24, phase 2026-04-01 → 2026-06-30, **166 participants, 5,369 submissions**, leaderboard columns Neg Log-Loss and AUC-ROC (hidden), same docker image lineage (`soe-cs321m-competition`).
- Tied to Stanford **CS321M** (Spring 2026; competition Apr 22 – Jun 5, 25% of grade; teams of 1–3; 4-page NeurIPS-style report; staff Koyejo, S. Truong, Hardy, Reuel, Sullivan, Unell, Chiou, Denisov-Blanch, Dullerud) — https://aims.stanford.edu/cs321m — and to the **COLM 2026 AI Measurement Science Workshop** (9 Oct 2026, SF; competition track deadline 15 Aug 2026; "Best Competition Paper" award) — https://aims.stanford.edu/workshop. No winners/results published on either page.
- Format then: 5,000 hidden items per submission, inputs `benchmark/condition/subject_content/item_content`, K=5 revealed labels per data category; metric NLL/AUC.
- Public results: BlakeMasters (https://blakemasters.github.io/aims-competition/, archived standings 14 Jun 2026) top-10: blakem31 −0.56/0.74, justinha −0.56/0.74, jdhouse −0.57/0.74, campeau, beingamanforever, saberz, mhayes3, ameya, shdkwok, ankagg94 (−0.58/0.72). Method: revealed-label anchoring + hierarchical shrinkage (τ=4) + calibration β=0.60; "calibrated priors −0.62, revealed-label anchoring −0.56". Other student repos: daphne-barretto −0.594/0.708 (item-kNN within subject); Joey-Ji −0.59 (Laplace subject ability → per-benchmark logistic calibration → blend); galarcon0803 −0.59/0.73 (joint 2PL IRT on 909 subjects + fingerprint mixture); antoinemaechler-X **−0.58 (organizer baseline −0.79, constant −0.69)**, temperature T=3 "tuned directly on the leaderboard because local validation did not transfer". Annika Kaul Singh (CS321M write-up): "simple lookup-based statistics were more reliable than larger neural or LLM-based additions… often performed worse on the hidden leaderboard" (distribution shift).

---

## 7. Social / community channels: searched, nothing substantive
- **X/Twitter**: site-restricted searches for "Predictive AI Evaluation Competition", "measurement-db", "Measurement Data Bank", "PAIEC" → no posts.
- **LinkedIn**: one search snippet of a Sang Truong post ("We are excited to share that we are launching the Predictive AI Evaluation Competition at NeurIPS 2026…", listing co-organizers) surfaced via Ramzi Dakhmouche's profile; LinkedIn returns HTTP 999 to fetches, so date/engagement unknown. Olawale Salaudeen's site confirms "co-organizing… Sang Truong leads."
- **Hacker News** (Algolia API): no hits for the competition/dataset; only "AI Measurement Science" textbook thread (2026-10-07, 16 points, 1 skeptical comment). https://news.ycombinator.com/item?id=49994142
- **Reddit**: web search found nothing; reddit JSON API blocked from this host.
- **Bluesky/Mastodon/Threads**: web search nothing; Bluesky public API returned non-JSON (blocked).
- **Discord**: two invite links on the page; not accessible without an account, not checked.
- **Codabench forum**: login-gated, not readable.
- **Chinese platforms** (知乎/CSDN/bilibili) and Medium/Substack: nothing.
- **Wayback Machine**: offline during research, so the page history (Nov 20 → Oct 30 deadline change; "Challenge" → "Competition" rename) could not be confirmed.
- Blog commentary on AIMS (xlnc.co, 13 Oct 2026, Matt Barney) discusses the textbook only.

## 8. Explicit gaps (looked, found nothing)
- No organizer-published baseline Brier ALC, no leaderboard, no participant counts beyond the Codabench API field, no interim results, no announced prizes (besides the Data Curator Award in the Google Doc).
- No arXiv/whitepaper for PAIEC or measurement-db; no slides/videos from organizers about the competition.
- No accessible content from Codabench forum, Discord, OpenReview submissions, or the gated HF README/`measurement-db-gallery`.
- Could not verify the "leading entry" lucaluo925 refers to (suggests some ranking is visible to logged-in participants, contradicting "aggregate visibility announced separately"; unresolved).
