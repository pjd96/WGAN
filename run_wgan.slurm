#!/bin/bash
#SBATCH --job-name=wgan
#SBATCH --output=wgan-%j.out
#SBATCH --error=wgan-%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate wgan
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
nvidia-smi
python -u - <<'PYTHON'
import os
import sys
import torch

print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
if torch.version.cuda is None:
    raise RuntimeError("CPU-only PyTorch installed; install a CUDA build in the wgan environment")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA build installed but GPU unavailable; check Slurm allocation and driver")
print("Allocated GPU:", torch.cuda.get_device_name(0))
# Verify execution on the allocated GPU before loading the dataset.
x = torch.ones(4, device="cuda")
assert (x + x).sum().item() == 8
print("CUDA computation passed", flush=True)
PYTHON
python download_data.py --dataroot data/cifar10 --check-only
python -u main.py --dataset cifar10 --dataroot data/cifar10 \
  --imageSize 32 --batchSize 64 --workers 4 --niter 25 \
  --cuda --manualSeed 123 --experiment "runs/cifar10-${SLURM_JOB_ID}"
