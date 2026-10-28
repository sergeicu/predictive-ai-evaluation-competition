#!/bin/bash
#SBATCH -A sundai_aihub -p cpu -J paiec-evals
#SBATCH -c 4 --mem=16G -t 02:00:00
#SBATCH -o /home/serge_ad_aihub/paiec/logs/%x-%A_%a.out
# usage: sbatch --array=1-N jobs/eval_array.sbatch <tasks-file>; line i = "<sub> <split> <seeds> [extra args]"
source /home/serge_ad_aihub/paiec/orchestration/jobs/env.sh
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")
[ -z "$LINE" ] && exit 0
set -- $LINE
SUB=$1; SPLIT=$2; SEEDS=$3; shift 3
cd $ORCH && $PY run_eval.py --sub "$SUB" --split "$SPLIT" --seeds "$SEEDS" "$@"
