#!/bin/bash
#SBATCH --job-name=mdgen-forward-smoke
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/forward_smoke_%j.out
#SBATCH --error=logs/forward_smoke_%j.err

set -euxo pipefail

cd /home/yh1924/projects/mdgen
ENV_PREFIX=/home/yh1924/projects/mdgen/envs/mdgen-py310
PYTHON=$ENV_PREFIX/bin/python

echo "start $(date -Is) host=$(hostname)"

$PYTHON - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

$PYTHON sim_inference.py \
  --sim_ckpt checkpoints/forward_sim.ckpt \
  --data_dir data/4AA_data \
  --split splits/smoke_FLRH.csv \
  --pdb_id FLRH \
  --num_rollouts 1 \
  --num_frames 1000 \
  --suffix _i100 \
  --xtc \
  --out_dir outputs/forward_smoke

echo "done $(date -Is)"
