#!/bin/bash
# Stage 1b: refit after the key fix; sweep ability-transfer and pair-variance multipliers.
set -euo pipefail
source /home/serge_ad_aihub/paiec/orchestration/jobs/env.sh
cd $ORCH
S=$PAIEC_SCRATCH/stage1
DEV=$(grep -v '^#' stages/stage1_dev.txt | paste -sd,)
rm -rf $PAIEC_ROOT/subs/M1_prior_s1 $PAIEC_ROOT/subs/M2_hier_s1* $PAIEC_ROOT/runs/M1_prior_s1 $PAIEC_ROOT/runs/M2_hier_s1*
$PY coverage.py --train $S/train_pairs.parquet
$PY fit_hier.py --train $S/train_pairs.parquet --name M1_prior_s1 --mode M1 --exclude "$DEV"
for am in 0 0.5 1; do for sm in 1 3; do
  $PY fit_hier.py --train $S/train_pairs.parquet --name M2_s1_a${am}_sig${sm} --mode M2 --exclude "$DEV" --a-mult $am --sigma-mult $sm
done; done
# quick smoke test of one fitted submission on a tiny eval
$PY run_eval.py --sub M2_s1_a1_sig1 --split validation --seeds 0 --max-eval-items 10 --max-pairs 2 --tag smoke
echo STAGE1B_FIT_DONE
