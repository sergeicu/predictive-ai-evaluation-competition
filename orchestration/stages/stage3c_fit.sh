#!/bin/bash
# Stage 3c: push pair variance further, item variance 4 vs 5, temperature check.
set -euo pipefail
source /home/serge_ad_aihub/paiec/orchestration/jobs/env.sh
cd $ORCH
S=$PAIEC_SCRATCH/stage2
DEV=$(grep -v '^#' stages/stage2_dev.txt | paste -sd,)
OMEGA=$($PY -c "import json;print(json.load(open('$S/omega2.json'))['median'])")
rm -rf $PAIEC_ROOT/subs/M3_s3c_* $PAIEC_ROOT/runs/M3_s3c_*
for sg in 1.5 2.5; do for om in 4 5; do
  $PY fit_hier.py --train $S/train_pairs.parquet --name M3_s3c_om${om}_s${sg} --mode M3 --exclude "$DEV" --a-mult 0 --sigma2-abs $sg --omega2 $OMEGA --omega-mult $om
done; done
for t in 0.9 1.1; do
  $PY fit_hier.py --train $S/train_pairs.parquet --name M3_s3c_om4_s1.0_t${t} --mode M3 --exclude "$DEV" --a-mult 0 --sigma2-abs 1.0 --omega2 $OMEGA --omega-mult 4 --temp $t
done
echo STAGE3C_FIT_DONE
