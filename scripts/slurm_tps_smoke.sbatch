#!/bin/bash
#SBATCH --job-name=mdgen-tps-smoke
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/tps_smoke_%j.out
#SBATCH --error=logs/tps_smoke_%j.err

set -euxo pipefail

cd /home/yh1924/projects/mdgen
ENV_PREFIX=/home/yh1924/projects/mdgen/envs/mdgen-py310
PYTHON=$ENV_PREFIX/bin/python

SMOKE_NAME=${MDGEN_SMOKE_NAME:-IPGD}
OUT_DIR=${MDGEN_OUT_DIR:-outputs/tps_smoke}

echo "start $(date -Is) host=$(hostname)"
echo "name=${SMOKE_NAME} out_dir=${OUT_DIR}"

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
  --pdb_id "${SMOKE_NAME}" \
  --num_frames 100 \
  --num_batches 1 \
  --batch_size 2 \
  --suffix _i100 \
  --data_stride 100 \
  --mddir data/4AA_sims \
  --out_dir "${OUT_DIR}"

echo "done $(date -Is)"
