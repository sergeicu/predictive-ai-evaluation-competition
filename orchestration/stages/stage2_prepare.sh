#!/bin/bash
# Stage 2: ~100 benchmarks (subjects + responses only), dev2 holdout, fits + sweep.
set -euo pipefail
source /home/serge_ad_aihub/paiec/orchestration/jobs/env.sh
cd $ORCH
S=$PAIEC_SCRATCH/stage2; mkdir -p $S/dev
$PY fetch.py --list stages/stage2_train.txt --tables subjects,responses,benchmarks --workers 8
$PY fetch.py --list stages/stage2_dev.txt   --tables subjects,responses,benchmarks --workers 8
$PY build_train.py --out $S/train_pairs.parquet
DEV=$(grep -v '^#' stages/stage2_dev.txt | paste -sd,)
$PY build_dev_holdout.py --benchmarks "$DEV" --out $S/dev/holdout.parquet --salt dev2-salt
$PY coverage.py --train $S/train_pairs.parquet
$PY coverage.py --train $S/train_pairs.parquet --holdout $S/dev/holdout.parquet
$PY fit_hier.py --train $S/train_pairs.parquet --name M1_prior_s2 --mode M1 --exclude "$DEV"
for am in 0 0.5 1; do for sm in 1 3; do
  $PY fit_hier.py --train $S/train_pairs.parquet --name M2_s2_a${am}_sig${sm} --mode M2 --exclude "$DEV" --a-mult $am --sigma-mult $sm
done; done
du -sh $PAIEC_SCRATCH/data
echo STAGE2_PREPARE_DONE
