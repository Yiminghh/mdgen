#!/bin/bash
#SBATCH --job-name=mdgen-upsampling-smoke
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/upsampling_smoke_%j.out
#SBATCH --error=logs/upsampling_smoke_%j.err

set -euxo pipefail

cd /home/yh1924/projects/mdgen
ENV_PREFIX=/home/yh1924/projects/mdgen/envs/mdgen-py310
PYTHON=$ENV_PREFIX/bin/python

SMOKE_NAME=${MDGEN_SMOKE_NAME:-EHFR}
SMOKE_FRAMES=${MDGEN_SMOKE_FRAMES:-20}
SMOKE_DATA_DIR=${MDGEN_SMOKE_DATA_DIR:-data/4AA_data_implicit_smoke}
OUT_DIR=${MDGEN_OUT_DIR:-outputs/upsampling_smoke}

echo "start $(date -Is) host=$(hostname)"
echo "name=${SMOKE_NAME} frames=${SMOKE_FRAMES} data_dir=${SMOKE_DATA_DIR} out_dir=${OUT_DIR}"

$PYTHON - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

$PYTHON - <<PY
import os
import numpy as np

name = "${SMOKE_NAME}"
frames = int("${SMOKE_FRAMES}")
out_dir = "${SMOKE_DATA_DIR}"
src = f"data/4AA_data_implicit/{name}_i100.npy"
dst = f"{out_dir}/{name}_i100.npy"

os.makedirs(out_dir, exist_ok=True)
arr = np.lib.format.open_memmap(src, "r")
np.save(dst, np.asarray(arr[:frames]))
print("smoke_npy", dst, np.load(dst, mmap_mode="r").shape)
PY

$PYTHON upsampling_inference.py \
  --ckpt checkpoints/upsampling.ckpt \
  --data_dir "${SMOKE_DATA_DIR}" \
  --split splits/4AA_implicit_test.csv \
  --pdb_id "${SMOKE_NAME}" \
  --batch_size 2 \
  --cond_interval 100 \
  --suffix _i100 \
  --out_dir "${OUT_DIR}"

echo "done $(date -Is)"
