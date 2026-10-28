#!/bin/bash
# Stage 5: candidate submissions = best M3 variance settings + hybrid acquisition (first 3 labels coordinated).
set -euo pipefail
source /home/serge_ad_aihub/paiec/orchestration/jobs/env.sh
cd $ORCH
S=$PAIEC_SCRATCH/stage2
DEV=$(grep -v '^#' stages/stage2_dev.txt | paste -sd,)
OMEGA=$($PY -c "import json;print(json.load(open('$S/omega2.json'))['median'])")
rm -rf $PAIEC_ROOT/subs/CAND_* $PAIEC_ROOT/runs/CAND_*
for sg in 1.0 1.5; do
  $PY fit_hier.py --train $S/train_pairs.parquet --name CAND_om4_s${sg}_acq3 --mode M3 --exclude "$DEV" --a-mult 0 --sigma2-abs $sg --omega2 $OMEGA --omega-mult 4 --with-labeling --acq-coord-until 3 --acq-c 1.5
done
echo STAGE5_DONE
