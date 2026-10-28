#!/bin/bash
# Stage 1d (small data): larger item variances, b0 fallback with M3, coordinated acquisition (M4).
set -euo pipefail
source /home/serge_ad_aihub/paiec/orchestration/jobs/env.sh
cd $ORCH
S=$PAIEC_SCRATCH/stage1
DEV=$(grep -v '^#' stages/stage1_dev.txt | paste -sd,)
OMEGA=$($PY -c "import json;print(json.load(open('$S/omega2.json'))['median'])")
rm -rf $PAIEC_ROOT/subs/M*_s1d* $PAIEC_ROOT/runs/M*_s1d*
for om in 3 4 6; do
  $PY fit_hier.py --train $S/train_pairs.parquet --name M3_s1d_om${om} --mode M3 --exclude "$DEV" --a-mult 0 --sigma-mult 2 --omega2 $OMEGA --omega-mult $om
done
$PY fit_hier.py --train $S/train_pairs.parquet --name M3_s1d_om4_b0half --mode M3 --exclude "$DEV" --a-mult 0 --sigma-mult 2 --omega2 $OMEGA --omega-mult 4 --b0-half
$PY fit_hier.py --train $S/train_pairs.parquet --name M4_s1d_om4_acq --mode M3 --exclude "$DEV" --a-mult 0 --sigma-mult 2 --omega2 $OMEGA --omega-mult 4 --with-labeling
$PY fit_hier.py --train $S/train_pairs.parquet --name M4_s1d_om2_acq --mode M3 --exclude "$DEV" --a-mult 0 --sigma-mult 2 --omega2 $OMEGA --omega-mult 2 --with-labeling
$PY run_eval.py --sub M4_s1d_om4_acq --split validation --seeds 0 --max-eval-items 10 --max-pairs 2 --tag smoke
echo STAGE1D_FIT_DONE
