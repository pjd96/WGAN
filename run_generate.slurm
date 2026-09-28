#!/bin/bash
#SBATCH --job-name=wgan-generate
#SBATCH --output=wgan-generate-%j.out
#SBATCH --error=wgan-generate-%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Usage: sbatch run_generate.slurm [run_directory] [epoch] [image_count]
if (( $# > 3 )); then
  echo "Usage: sbatch run_generate.slurm [run_directory] [epoch] [image_count]" >&2
  exit 1
fi
RUN_DIR="${1:-runs/cifar10-dcgan}"
EPOCH="${2:-24}"
NIMAGES="${3:-64}"
CONFIG="$RUN_DIR/generator_config.json"
WEIGHTS="$RUN_DIR/netG_epoch_${EPOCH}.pth"
# Keep outputs from separate generation jobs in separate directories.
OUTPUT_DIR="$RUN_DIR/generated-${SLURM_JOB_ID}"

for path in "$CONFIG" "$WEIGHTS"; do
  if [[ ! -f "$path" ]]; then
    echo "Required file not found: $path" >&2
    exit 1
  fi
done

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate wgan
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
nvidia-smi

# generate.py checks CUDA availability before loading the model.
python -u generate.py --config "$CONFIG" \
  --weights "$WEIGHTS" --output_dir "$OUTPUT_DIR" \
  --nimages "$NIMAGES" --batchSize 64 --manualSeed 1 --cuda
