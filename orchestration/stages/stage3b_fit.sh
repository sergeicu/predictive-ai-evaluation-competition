#!/bin/bash
# Stage 3b: hybrid acquisition variants on stage-2 data (om4, sigma 0.6).
set -euo pipefail
source /home/serge_ad_aihub/paiec/orchestration/jobs/env.sh
cd $ORCH
S=$PAIEC_SCRATCH/stage2
DEV=$(grep -v '^#' stages/stage2_dev.txt | paste -sd,)
OMEGA=$($PY -c "import json;print(json.load(open('$S/omega2.json'))['median'])")
rm -rf $PAIEC_ROOT/subs/M4_s3b_* $PAIEC_ROOT/runs/M4_s3b_*
for cu in 3 7 15; do for c in 1.0 1.5; do
  $PY fit_hier.py --train $S/train_pairs.parquet --name M4_s3b_cu${cu}_c${c} --mode M3 --exclude "$DEV" --a-mult 0 --sigma2-abs 0.6 --omega2 $OMEGA --omega-mult 4 --with-labeling --acq-coord-until $cu --acq-c $c
done; done
echo STAGE3B_FIT_DONE
