#!/bin/bash
#SBATCH -A sundai_aihub -p cpu -J paiec-run
#SBATCH -c 8 --mem=32G -t 04:00:00
#SBATCH -o /home/serge_ad_aihub/paiec/logs/%x-%j.out
# usage: sbatch run.sbatch <script.py> [args...]   (generic: fits, builds, sweeps)
source /home/serge_ad_aihub/paiec/orchestration/jobs/env.sh
cd $ORCH && $PY "$@"
