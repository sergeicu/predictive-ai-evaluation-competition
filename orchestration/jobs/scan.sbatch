#!/bin/bash
#SBATCH -A sundai_aihub -p cpu -J paiec-scan
#SBATCH -c 4 --mem=8G -t 00:30:00
#SBATCH -o /home/serge_ad_aihub/paiec/logs/%x-%j.out
source /home/serge_ad_aihub/paiec/orchestration/jobs/env.sh
cd $ORCH && $PY scan_catalog.py
