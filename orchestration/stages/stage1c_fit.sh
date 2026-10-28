#!/bin/bash
# Stage 1c (same small data): sigma refit per transfer coefficient, wider sweep, budget-0 fallback, item effects (M3).
set -euo pipefail
source /home/serge_ad_aihub/paiec/orchestration/jobs/env.sh
cd $ORCH
S=$PAIEC_SCRATCH/stage1
DEV=$(grep -v '^#' stages/stage1_dev.txt | paste -sd,)
TRAIN=$(grep -v '^#' stages/stage1_train.txt | paste -sd,)
$PY build_dev_holdout.py --benchmarks "$DEV" --out $S/dev/holdout.parquet
OMEGA=$($PY fit_items.py --benchmarks "$TRAIN" --out $S/omega2.json)
echo "omega2=$OMEGA"
rm -rf $PAIEC_ROOT/subs/M*_s1c* $PAIEC_ROOT/runs/M*_s1c*
for am in 0 0.25 0.5; do for sm in 1 2 4; do
  $PY fit_hier.py --train $S/train_pairs.parquet --name M2_s1c_a${am}_sig${sm} --mode M2 --exclude "$DEV" --a-mult $am --sigma-mult $sm
done; done
$PY fit_hier.py --train $S/train_pairs.parquet --name M2_s1c_a0_sig2_tau2 --mode M2 --exclude "$DEV" --a-mult 0 --sigma-mult 2 --tau-mult 2
$PY fit_hier.py --train $S/train_pairs.parquet --name M2_s1c_a0_sig2_b0half --mode M2 --exclude "$DEV" --a-mult 0 --sigma-mult 2 --b0-half
for om in 0.5 1 2; do
  $PY fit_hier.py --train $S/train_pairs.parquet --name M3_s1c_a0_sig2_om${om} --mode M3 --exclude "$DEV" --a-mult 0 --sigma-mult 2 --omega2 $OMEGA --omega-mult $om
done
$PY run_eval.py --sub M3_s1c_a0_sig2_om1 --split validation --seeds 0 --max-eval-items 10 --max-pairs 2 --tag smoke
echo STAGE1C_FIT_DONE
