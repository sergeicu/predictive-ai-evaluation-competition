#!/bin/bash
#SBATCH -A sundai_aihub -p cpu -J paiec-fetch
#SBATCH -c 8 --mem=16G -t 04:00:00
#SBATCH -o /home/serge_ad_aihub/paiec/logs/%x-%j.out
# usage: sbatch fetch.sbatch <list-file> [tables]
source /home/serge_ad_aihub/paiec/orchestration/jobs/env.sh
cd $ORCH && $PY fetch.py --list "$1" --tables "${2:-subjects,responses,benchmarks}" --workers 8
