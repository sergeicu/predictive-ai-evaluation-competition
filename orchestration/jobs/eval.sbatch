#!/bin/bash
#SBATCH -A sundai_aihub -p cpu -J paiec-eval
#SBATCH -c 4 --mem=16G -t 02:00:00
#SBATCH -o /home/serge_ad_aihub/paiec/logs/%x-%j.out
# usage: sbatch eval.sbatch <sub> <split> <seeds> [extra run_eval args...]
source /home/serge_ad_aihub/paiec/orchestration/jobs/env.sh
SUB=$1; SPLIT=$2; SEEDS=$3; shift 3
cd $ORCH && $PY run_eval.py --sub "$SUB" --split "$SPLIT" --seeds "$SEEDS" "$@"
