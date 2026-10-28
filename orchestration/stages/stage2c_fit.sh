#!/bin/bash
# Stage 2c: same 111-benchmark data; sigma2 from recent subjects / absolute sweep, M3 item effects, diagnostics.
set -euo pipefail
source /home/serge_ad_aihub/paiec/orchestration/jobs/env.sh
cd $ORCH
S=$PAIEC_SCRATCH/stage2
DEV=$(grep -v '^#' stages/stage2_dev.txt | paste -sd,)
TRAIN=$(grep -v '^#' stages/stage2_train.txt | paste -sd,)
$PY build_dev_holdout.py --benchmarks "$DEV" --out $S/dev/holdout.parquet --salt dev2-salt
$PY diag_transfer.py --params $PAIEC_ROOT/subs/M2_s2_a1_sig1/params.json --holdout $S/dev/holdout.parquet
OMEGA=$($PY fit_items.py --benchmarks "$TRAIN" --out $S/omega2.json)
echo "omega2=$OMEGA"
rm -rf $PAIEC_ROOT/subs/M*_s2c* $PAIEC_ROOT/runs/M*_s2c*
$PY fit_hier.py --train $S/train_pairs.parquet --name M2_s2c_a0_recent --mode M2 --exclude "$DEV" --a-mult 0 --recent-since 2024-06-01
$PY fit_hier.py --train $S/train_pairs.parquet --name M2_s2c_a0.5_recent --mode M2 --exclude "$DEV" --a-mult 0.5 --recent-since 2024-06-01
for sg in 0.3 0.6 1.0; do
  $PY fit_hier.py --train $S/train_pairs.parquet --name M2_s2c_a0_s${sg} --mode M2 --exclude "$DEV" --a-mult 0 --sigma2-abs $sg
done
$PY fit_hier.py --train $S/train_pairs.parquet --name M2_s2c_a0.25_s0.6 --mode M2 --exclude "$DEV" --a-mult 0.25 --sigma2-abs 0.6
$PY fit_hier.py --train $S/train_pairs.parquet --name M2_s2c_a0_s0.6_b0half --mode M2 --exclude "$DEV" --a-mult 0 --sigma2-abs 0.6 --b0-half
for om in 0.5 1 2; do
  $PY fit_hier.py --train $S/train_pairs.parquet --name M3_s2c_a0_s0.6_om${om} --mode M3 --exclude "$DEV" --a-mult 0 --sigma2-abs 0.6 --omega2 $OMEGA --omega-mult $om
done
echo STAGE2C_FIT_DONE
