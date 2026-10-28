#!/bin/bash
# Stage 1: fetch small pool, build train table, dev holdout, fit M1/M2. Run via run.sbatch-style job.
set -euo pipefail
source /home/serge_ad_aihub/paiec/orchestration/jobs/env.sh
cd $ORCH
S=$PAIEC_SCRATCH/stage1; mkdir -p $S/dev
$PY fetch.py --list stages/stage1_train.txt --tables subjects,responses,benchmarks --workers 8
$PY fetch.py --list stages/stage1_dev.txt   --tables subjects,responses,benchmarks --workers 8
$PY build_train.py --out $S/train_pairs.parquet --only <(cat stages/stage1_train.txt stages/stage1_dev.txt)
DEV=$(grep -v '^#' stages/stage1_dev.txt | paste -sd,)
$PY build_dev_holdout.py --benchmarks "$DEV" --out $S/dev/holdout.parquet
$PY coverage.py --train $S/train_pairs.parquet
$PY coverage.py --train $S/train_pairs.parquet --holdout $S/dev/holdout.parquet
# fits for the hill (dev benchmarks excluded so the same artifacts serve both scorers)
$PY fit_hier.py --train $S/train_pairs.parquet --name M1_prior_s1 --mode M1 --exclude "$DEV"
$PY fit_hier.py --train $S/train_pairs.parquet --name M2_hier_s1  --mode M2 --exclude "$DEV"
$PY fit_hier.py --train $S/train_pairs.parquet --name M2_hier_s1_sig2  --mode M2 --exclude "$DEV" --sigma-mult 2.0
$PY fit_hier.py --train $S/train_pairs.parquet --name M2_hier_s1_sig05 --mode M2 --exclude "$DEV" --sigma-mult 0.5
echo STAGE1_PREPARE_DONE
