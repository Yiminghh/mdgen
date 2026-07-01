#!/bin/bash
#SBATCH --job-name=mdgen-forward-test100
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/forward_test100_%j.out
#SBATCH --error=logs/forward_test100_%j.err

set -euxo pipefail

cd /home/yh1924/projects/mdgen
ENV_PREFIX=/home/yh1924/projects/mdgen/envs/mdgen-py310
PYTHON=$ENV_PREFIX/bin/python

NUM_ROLLOUTS=${MDGEN_NUM_ROLLOUTS:-10}
NUM_FRAMES=${MDGEN_NUM_FRAMES:-1000}
OUT_DIR=${MDGEN_OUT_DIR:-outputs/forward_test100}

echo "start $(date -Is) host=$(hostname)"
echo "num_rollouts=${NUM_ROLLOUTS} num_frames=${NUM_FRAMES} out_dir=${OUT_DIR}"

$PYTHON - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

$PYTHON sim_inference.py \
  --sim_ckpt checkpoints/forward_sim.ckpt \
  --data_dir data/4AA_data \
  --split splits/4AA_test.csv \
  --num_rollouts "${NUM_ROLLOUTS}" \
  --num_frames "${NUM_FRAMES}" \
  --suffix _i100 \
  --xtc \
  --out_dir "${OUT_DIR}"

echo "done $(date -Is)"
