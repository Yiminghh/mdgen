#!/bin/bash
#SBATCH --job-name=mdgen-prep-4aa-test-full
#SBATCH --partition=cpu
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --output=logs/prep_4aa_test_full_%j.out
#SBATCH --error=logs/prep_4aa_test_full_%j.err

set -euxo pipefail

cd /home/yh1924/projects/mdgen
ENV_PREFIX=/home/yh1924/projects/mdgen/envs/mdgen-py310
PYTHON=$ENV_PREFIX/bin/python

SPLIT=${MDGEN_SPLIT:-splits/4AA_test.csv}
SIM_DIR=${MDGEN_SIM_DIR:-data/4AA_sims}
OUT_DIR=${MDGEN_OUT_DIR:-data/4AA_data}
NUM_WORKERS=${MDGEN_NUM_WORKERS:-4}

echo "start $(date -Is) host=$(hostname)"
echo "split=${SPLIT} sim_dir=${SIM_DIR} out_dir=${OUT_DIR} num_workers=${NUM_WORKERS}"

$PYTHON -m scripts.prep_sims \
  --split "${SPLIT}" \
  --sim_dir "${SIM_DIR}" \
  --outdir "${OUT_DIR}" \
  --num_workers "${NUM_WORKERS}"

$PYTHON - <<'PY'
from pathlib import Path

out_dir = Path("data/4AA_data")
full = sorted(p for p in out_dir.glob("*.npy") if not p.name.endswith("_i100.npy"))
stride = sorted(out_dir.glob("*_i100.npy"))
print("full_no_suffix", len(full))
print("stride_i100", len(stride))
print("full_bytes", sum(p.stat().st_size for p in full))
PY

echo "done $(date -Is)"
