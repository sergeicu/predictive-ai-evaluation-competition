#!/bin/bash
# Stage 4 diagnostic: fetch items tables for the small stage-1 pool only; test text features -> item/benchmark difficulty.
set -euo pipefail
source /home/serge_ad_aihub/paiec/orchestration/jobs/env.sh
cd $ORCH
$PY fetch.py --list stages/stage1_train.txt --tables items --workers 8
$PY fetch.py --list stages/stage1_dev.txt --tables items --workers 8
ALL=$(grep -hv '^#' stages/stage1_train.txt stages/stage1_dev.txt | paste -sd,)
$PY diag_item_text.py --benchmarks "$ALL"
echo STAGE4_DIAG_DONE
