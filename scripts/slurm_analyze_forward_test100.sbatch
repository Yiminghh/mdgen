#!/bin/bash
#SBATCH --job-name=mdgen-analyze-forward-test100
#SBATCH --partition=cpu
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --output=logs/analyze_forward_test100_%j.out
#SBATCH --error=logs/analyze_forward_test100_%j.err

set -euxo pipefail

cd /home/yh1924/projects/mdgen
ENV_PREFIX=/home/yh1924/projects/mdgen/envs/mdgen-py310
PYTHON=$ENV_PREFIX/bin/python

export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PDB_DIR=${MDGEN_PDB_DIR:-outputs/forward_test100}
SAVE_NAME=${MDGEN_SAVE_NAME:-out.pkl}
NUM_WORKERS=${MDGEN_ANALYSIS_WORKERS:-4}
PDB_IDS=${MDGEN_PDB_IDS:-}
PDB_ID_ARGS=()
if [[ -n "${PDB_IDS}" ]]; then
  read -r -a PDB_ID_LIST <<< "${PDB_IDS}"
  PDB_ID_ARGS=(--pdb_id "${PDB_ID_LIST[@]}")
fi

echo "start $(date -Is) host=$(hostname)"
echo "pdb_dir=${PDB_DIR} save_name=${SAVE_NAME} num_workers=${NUM_WORKERS} pdb_ids=${PDB_IDS:-ALL}"

$PYTHON -m scripts.analyze_peptide_sim \
  --mddir data/4AA_sims \
  --pdbdir "${PDB_DIR}" \
  --plot \
  --save \
  --save_name "${SAVE_NAME}" \
  "${PDB_ID_ARGS[@]}" \
  --num_workers "${NUM_WORKERS}"

echo "done $(date -Is)"
