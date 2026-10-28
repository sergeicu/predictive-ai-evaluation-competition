# sourced by every job
export PAIEC_ROOT=/home/serge_ad_aihub/paiec
export PAIEC_SCRATCH=/scratch/serge_ad_aihub/paiec
export HF_HOME=$PAIEC_SCRATCH/hf_cache
export HF_TOKEN="$(cat "$HOME/.secrets/hf_token")"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
PY=$PAIEC_ROOT/.venv/bin/python
ORCH=$PAIEC_ROOT/orchestration
mkdir -p $PAIEC_SCRATCH $PAIEC_ROOT/runs $PAIEC_ROOT/logs
