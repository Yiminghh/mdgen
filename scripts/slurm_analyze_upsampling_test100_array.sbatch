#!/bin/bash
#SBATCH --job-name=mdgen-analyze-upsampling-test100
#SBATCH --partition=cpu
#SBATCH --array=0-9%4
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=56G
#SBATCH --output=logs/analyze_upsampling_test100_%A_%a.out
#SBATCH --error=logs/analyze_upsampling_test100_%A_%a.err

set -euxo pipefail

cd /home/yh1924/projects/mdgen
ENV_PREFIX=/home/yh1924/projects/mdgen/envs/mdgen-py310
PYTHON=$ENV_PREFIX/bin/python

CHUNK_IDX=${SLURM_ARRAY_TASK_ID:-${MDGEN_CHUNK_IDX:-0}}
N_CHUNKS=${MDGEN_N_CHUNKS:-10}
PDB_DIR=${MDGEN_PDB_DIR:-outputs/upsampling_test100}

export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "start $(date -Is) host=$(hostname)"
echo "chunk_idx=${CHUNK_IDX} n_chunks=${N_CHUNKS} pdb_dir=${PDB_DIR}"

$PYTHON -m scripts.analyze_upsampling \
  --mddir data/4AA_sims_implicit \
  --pdbdir "${PDB_DIR}" \
  --chunk_idx "${CHUNK_IDX}" \
  --n_chunks "${N_CHUNKS}"

echo "done $(date -Is)"
