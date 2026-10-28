#!/bin/bash
# Stage 3 (stage-2 data, 111 benchmarks): joint grid over item variance x pair variance, acquisition, b0 fallback.
set -euo pipefail
source /home/serge_ad_aihub/paiec/orchestration/jobs/env.sh
cd $ORCH
S=$PAIEC_SCRATCH/stage2
DEV=$(grep -v '^#' stages/stage2_dev.txt | paste -sd,)
OMEGA=$($PY -c "import json;print(json.load(open('$S/omega2.json'))['median'])")
rm -rf $PAIEC_ROOT/subs/M*_s3_* $PAIEC_ROOT/runs/M*_s3_*
for om in 2 3 4 6; do for sg in 0.6 1.0; do
  $PY fit_hier.py --train $S/train_pairs.parquet --name M3_s3_om${om}_s${sg} --mode M3 --exclude "$DEV" --a-mult 0 --sigma2-abs $sg --omega2 $OMEGA --omega-mult $om
done; done
$PY fit_hier.py --train $S/train_pairs.parquet --name M3_s3_om4_s0.6_b0half --mode M3 --exclude "$DEV" --a-mult 0 --sigma2-abs 0.6 --omega2 $OMEGA --omega-mult 4 --b0-half
$PY fit_hier.py --train $S/train_pairs.parquet --name M4_s3_om4_s0.6_acq --mode M3 --exclude "$DEV" --a-mult 0 --sigma2-abs 0.6 --omega2 $OMEGA --omega-mult 4 --with-labeling
$PY fit_hier.py --train $S/train_pairs.parquet --name M4_s3_om3_s1.0_acq --mode M3 --exclude "$DEV" --a-mult 0 --sigma2-abs 1.0 --omega2 $OMEGA --omega-mult 3 --with-labeling
echo STAGE3_FIT_DONE
