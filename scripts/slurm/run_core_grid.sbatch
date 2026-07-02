#!/usr/bin/env bash
#SBATCH --job-name=bmvm-grid
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

source .venv/bin/activate

CONFIG="${CONFIG:-configs/experiments/core_grid.json}"
MANIFEST_DIR="${MANIFEST_DIR:-data/manifests}"
OUTPUT_DIR="${OUTPUT_DIR:-results/grid}"
VLM_CACHE_DIR="${VLM_CACHE_DIR:-data/vlm_cache}"
SIM_FALLBACK="${SIM_FALLBACK:-0}"
SOURCE_COMMIT="${SOURCE_COMMIT:-}"
STREAM_COUNTS="${STREAM_COUNTS:-}"
LIMIT="${LIMIT:-}"

mkdir -p "${OUTPUT_DIR}"

cmd=(
  python3 scripts/run_grid.py
  --config "${CONFIG}"
  --manifest-dir "${MANIFEST_DIR}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${STREAM_COUNTS}" ]]; then
  cmd+=(--stream-counts "${STREAM_COUNTS}")
fi

if [[ -n "${SOURCE_COMMIT}" ]]; then
  cmd+=(--source-commit "${SOURCE_COMMIT}")
fi

if [[ -n "${LIMIT}" ]]; then
  cmd+=(--limit "${LIMIT}")
fi

if [[ "${SIM_FALLBACK}" == "1" ]]; then
  echo "Running with simulated VLM fallback enabled. Do not use this for paper-quality real experiments."
else
  cmd+=(--vlm-cache-dir "${VLM_CACHE_DIR}" --no-simulated-vlm-fallback)
fi

printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
