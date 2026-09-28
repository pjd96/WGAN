# Image generation / Wasserstein GAN

**Project:** Train and evaluate a Wasserstein generative adversarial network.

**Paper:** Martin Arjovsky, Soumith Chintala and Léon Bottou,
“Wasserstein Generative Adversarial Networks,” ICML, 2017.
https://arxiv.org/abs/1701.07875

**Original code:** https://github.com/martinarjovsky/WassersteinGAN

This teaching copy updates the original code for Python 3.12 and PyTorch 2.7.1.
It implements the original **weight-clipped WGAN**, not WGAN-GP. The default
optimizer is RMSprop. See [VALIDATION.md](VALIDATION.md) for what was tested and
what still requires a GPU experiment. Short successful runs check the software;
they do not establish convergence or reproduce the paper's image quality.

**University server rule:** run all GPU work through a Slurm script submitted
with `sbatch`, including GPU checks, training, and inference. The GPU commands
below belong inside a Slurm job; do not run them directly on the login node.
A login node may have no visible GPU even when compute nodes have GPUs.

## 1. Get the teaching copy and create an environment

If working remotely, first log in:

```bash
ssh -i <path_to_key_file> <username>@<server>
```

Use the copy supplied by your supervisor (including this guide and
`download_data.py`). **Cloning the original upstream repository alone does not
include these compatibility fixes.** Put the supplied directory in your projects
folder, then run all commands below from its root:

```bash
cd ~/projects/WassersteinGAN
conda create -n wgan python=3.12 pip -y
conda activate wgan
```

If conda is unavailable, install Miniconda using the
[official installation guide](https://www.anaconda.com/docs/getting-started/miniconda/install).
Alternatively, with Python 3.12 installed, use a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Choose **one** PyTorch installation below. The pinned versions are a tested
baseline, not a claim to be the latest release. CPU testing is supported; use an
NVIDIA GPU for substantial training.

**Linux CPU:**

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

**Linux NVIDIA GPU:** install the CUDA-enabled packages below. Run `nvidia-smi`
and CUDA availability checks inside the submitted Slurm job. For a driver
compatible with CUDA 11.8:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

Use the [official PyTorch version table](https://pytorch.org/get-started/previous-versions/)
for other supported CUDA builds of this version pair. The wheel supplies the CUDA
runtime; the host still needs a compatible NVIDIA driver. If switching an existing
CPU environment to GPU, create a fresh environment or reinstall the two PyTorch
packages from the GPU index with `--force-reinstall`.

Verify the active environment:

```bash
python -m pip check
python -c "import torch, torchvision; print('torch:', torch.__version__, 'torchvision:', torchvision.__version__); print('CUDA build:', torch.version.cuda)"
```

`CUDA available: False` is expected with the CPU setup, and can also be expected
on a login node with a CUDA build installed. `torch.version.cuda` distinguishes
the builds: `None` means CPU-only. Seeing GPUs in `nvidia-smi` does not make a
CPU-only PyTorch build CUDA-capable. Do not pass `--cuda` with a CPU-only build.
For CPU tests, limiting thread counts can avoid excessive overhead:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

## 2. Prepare CIFAR-10 before training

CIFAR-10 is the recommended first dataset. It contains 50,000 training and
10,000 test RGB images at 32 × 32 pixels. WGAN ignores the class labels and trains
only on the training split. This is a convenient starting exercise, not the LSUN
experiment from the paper. Dataset source: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).

Allow approximately **163 MiB for the archive and 350 MiB total after extraction**;
reserve at least 1 GB for data and additional space for checkpoints. Run this on
a machine with internet access, before submitting a cluster job:

```bash
python download_data.py --dataroot data/cifar10
python download_data.py --dataroot data/cifar10 --check-only
```

The script downloads the official Python archive over HTTPS, checks the archive
MD5 and extracted data files using torchvision's checksums, extracts it, and checks
that images can be decoded. Repeating it reuses valid data. `--check-only` never
downloads. An interrupted download can be retried; partial downloads restart.
Training itself never downloads anything.

Expected layout:

```text
data/cifar10/
├── cifar-10-python.tar.gz
└── cifar-10-batches-py/
    ├── data_batch_1 ... data_batch_5
    ├── test_batch
    ├── batches.meta
    └── readme.html
```

If the server cannot access the dataset, download the **CIFAR-10 Python version**
from the source page on another machine. Copy `cifar-10-python.tar.gz` into
`data/cifar10/`, then rerun the preparation command **without** `--check-only`.
The script checks and extracts the existing archive without downloading it again.
Alternatively, copy the whole prepared `data/cifar10/` folder to the compute
server and run `--check-only` there. No pretrained weights are required.

For long remote downloads, use `tmux new -s wgan-data`, detach with **Ctrl-b d**,
and reconnect using `tmux attach -t wgan-data`.

## 3. Check training and generation

First run the offline regression suite (uses temporary synthetic images, not the
downloaded dataset):

```bash
python -m unittest discover -s tests -v
```

Then run a small CPU test on real CIFAR-10:

```bash
python main.py --dataset cifar10 --dataroot data/cifar10 \
  --imageSize 32 --ngf 16 --ndf 16 --batchSize 16 --workers 0 \
  --niter 1 --max_batches 101 --manualSeed 123 \
  --experiment runs/cifar10-check

python generate.py --config runs/cifar10-check/generator_config.json \
  --weights runs/cifar10-check/netG_epoch_0.pth \
  --output_dir runs/cifar10-check/generated --nimages 16 --manualSeed 123
```

Expect finite `Loss_D` and `Loss_G` values, generator and critic `.pth` checkpoints,
`generator_config.json`, and 16 PNGs in the generated directory. **These images
will not look like a trained model**: the test performs only two generator updates.
`--max_batches` limits real-data batches **per epoch**; omit it for actual training.
Use a fresh experiment directory for each run, since matching filenames overwrite
previous outputs.

The original schedule uses up to 100 critic updates per generator update for the
first 25 generator steps and every 500th step, then `--Diters` (default 5).
This explains why initial output is slow and why the batch counter jumps.

## 4. Train a model

Training command for a one-GPU Slurm script (submit the script with `sbatch`):

```bash
python -u main.py --dataset cifar10 --dataroot data/cifar10 \
  --imageSize 32 --batchSize 64 --workers 4 --niter 25 \
  --cuda --manualSeed 123 --experiment runs/cifar10-dcgan
```

This is a starting configuration, not a guarantee that 25 epochs are sufficient.
For CPU training, remove `--cuda` and reduce workers if necessary; expect it to be
slow. DCGAN image sizes must be powers of two, at least 16 (normally 32 or 64).
Keep `--nc 3` for the provided RGB loaders.

Optional model experiments: add `--mlp_G --ngf 512` for an MLP generator,
`--mlp_D --ndf 512` for an MLP critic, or `--noBN` for the DCGAN generator without
batch normalization. As in the original training script, `--noBN` affects the
**generator only**. Do not combine it with `--mlp_G`.

### Slurm example

The supplied [run_wgan.slurm](run_wgan.slurm) includes GPU diagnostics and a CUDA
computation check. Submit it with `sbatch run_wgan.slurm`. The following is a
minimal equivalent template. Adapt partition,
resources, time limit, and conda installation path to your university cluster.
Download the data before submitting; compute nodes may have no internet access.

```bash
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
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"
python download_data.py --dataroot data/cifar10 --check-only
python -u main.py --dataset cifar10 --dataroot data/cifar10 \
  --imageSize 32 --batchSize 64 --workers 4 --niter 25 \
  --cuda --manualSeed 123 --experiment "runs/cifar10-${SLURM_JOB_ID}"
```

Submit from the project root and monitor:

```bash
sbatch run_wgan.slurm
squeue -u "$USER"
tail -f wgan-<job-id>.out
```

Checkpoints are written after every completed epoch, named `netG_epoch_0.pth`,
`netD_epoch_0.pth`, etc. Sample grids are written every 500 generator updates;
short runs may produce no grids. Use `generate.py` on any completed checkpoint.
To continue from weights, supply `--netG <generator.pth> --netD <critic.pth>` with
matching architecture options and a **new** experiment directory. This is a warm
start: optimizer state, epoch count, random state, and critic schedule are not
restored by the original checkpoint format.

## 5. Generate images and collect results

Submit the supplied [run_generate.slurm](run_generate.slurm) from the project
root. It uses the same GPU partition and conda environment as `run_wgan.slurm`,
with a 15-minute time limit. Adapt its resources and conda path for your cluster.
For a completed 25-epoch training job numbered 99 (epochs are numbered 0–24):

```bash
sbatch run_generate.slurm runs/cifar10-99
```

Replace `99` with your **training** job ID. The optional second and third arguments
select the checkpoint epoch and image count (defaults: `24` and `64`):

```bash
sbatch run_generate.slurm runs/cifar10-99 24 128
```

Images are saved in `runs/cifar10-99/generated-<generation-job-id>/`; logs are
`wgan-generate-<generation-job-id>.out` and `.err` in the project root.
With no arguments, the script uses `runs/cifar10-dcgan`. Generation runs this
command inside the Slurm job, with the selected paths and image count:

```bash
python generate.py --config runs/cifar10-dcgan/generator_config.json \
  --weights runs/cifar10-dcgan/netG_epoch_24.pth \
  --output_dir runs/cifar10-dcgan/generated --nimages 64 --cuda
```

Use your actual experiment path and a checkpoint that exists. Omit `--cuda` for
CPU inference, even if the checkpoint was trained on a GPU. Generation needs only
the generator configuration and weights; **it does not need a dataset or input
images**. Images are sampled from random latent noise. Use `--manualSeed` to repeat
a run on the same setup, and `--batchSize` to limit generation memory usage.

From your laptop, copy results back with:

```bash
scp -r -i <path_to_key_file> <username>@<server>:<project_path>/runs/cifar10-dcgan .
```

Retain the configuration, checkpoints, training log, seed, and package versions
(`python -m pip freeze > environment-used.txt`) with experimental results. The
original critic sign convention means the README plots **`-Loss_D`**. A plausible
loss alone does not establish sample quality; compare generated images and measure
quality/diversity across runs. GPU operations are not guaranteed bitwise repeatable.

## 6. Optional: LSUN bedrooms or your own images

LSUN bedrooms is the original README's training example. It is much larger than
CIFAR-10. The [official LSUN repository](https://github.com/fyu/lsun) provides the
download script and dataset details. Its legacy download service may be unavailable;
real LSUN downloading and large-scale training were **not validated here**.
For this reason, CIFAR-10 is the fully tested preparation path.

If the official service is reachable, download just bedrooms (the upstream script
fetches both training and validation archives):

```bash
git clone https://github.com/fyu/lsun.git data/lsun-tools
mkdir -p data/lsun
python data/lsun-tools/download.py -c bedroom -o data/lsun
unzip -t data/lsun/bedroom_train_lmdb.zip
unzip data/lsun/bedroom_train_lmdb.zip -d data/lsun
```

If it fails, obtain `bedroom_train_lmdb.zip` from your supervisor or an approved
dataset source, then use the two unzip commands. Check archive size and available
storage before proceeding; allow space for both archive and extracted database.
Do not rename image files or convert them: the loader needs an **LMDB database**:

```text
data/lsun/bedroom_train_lmdb/data.mdb
data/lsun/bedroom_train_lmdb/lock.mdb
```

Pass the **parent** directory, not the database directory:

```bash
python main.py --dataset lsun --dataroot data/lsun --imageSize 64 \
  --cuda --experiment runs/lsun-dcgan
```

The first LSUN load builds an index cache and can take a long time. The LMDB Python
package is included in `requirements.txt`. Only `bedroom_train` is used.

For your own JPEG/PNG images, put them in at least one subdirectory:

```text
data/my_images/all/image001.jpg
data/my_images/all/image002.jpg
```

Then run:

```bash
python main.py --dataset folder --dataroot data/my_images --imageSize 64 \
  --cuda --experiment runs/custom-dcgan
```

The loader resizes and center-crops images, converts them to RGB, and normalizes
them to [-1, 1]. Subdirectory labels are ignored. The `imagenet` and `lfw` options
are aliases for this folder loader; they do not download those datasets.
