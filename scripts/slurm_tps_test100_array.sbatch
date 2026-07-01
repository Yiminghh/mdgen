#!/bin/bash
#SBATCH --job-name=mdgen-tps-test100
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --array=0-9%2
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=logs/tps_test100_%A_%a.out
#SBATCH --error=logs/tps_test100_%A_%a.err

set -euxo pipefail

cd /home/yh1924/projects/mdgen
ENV_PREFIX=/home/yh1924/projects/mdgen/envs/mdgen-py310
PYTHON=$ENV_PREFIX/bin/python

CHUNK_IDX=${SLURM_ARRAY_TASK_ID:-${MDGEN_CHUNK_IDX:-0}}
N_CHUNKS=${MDGEN_N_CHUNKS:-10}
NUM_BATCHES=${MDGEN_NUM_BATCHES:-100}
BATCH_SIZE=${MDGEN_BATCH_SIZE:-10}
OUT_DIR=${MDGEN_OUT_DIR:-outputs/tps_test100}

echo "start $(date -Is) host=$(hostname)"
echo "chunk_idx=${CHUNK_IDX} n_chunks=${N_CHUNKS} num_batches=${NUM_BATCHES} batch_size=${BATCH_SIZE} out_dir=${OUT_DIR}"

$PYTHON - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

$PYTHON tps_inference.py \
  --sim_ckpt checkpoints/interpolation.ckpt \
  --data_dir data/4AA_data \
  --split splits/4AA_test.csv \
  --num_frames 100 \
  --num_batches "${NUM_BATCHES}" \
  --batch_size "${BATCH_SIZE}" \
  --suffix _i100 \
  --data_stride 100 \
  --mddir data/4AA_sims \
  --out_dir "${OUT_DIR}" \
  --chunk_idx "${CHUNK_IDX}" \
  --n_chunks "${N_CHUNKS}"

echo "done $(date -Is)"
