#!/bin/bash
# Experiment on FRIDA (see slurm/README.md). Submit from the repo root:
#   sbatch slurm/frida.sbatch [config]    (default configs/llms_bias_bench.yaml)
# sbatch options override the lines below, e.g. --time=1:00:00 for the trial run.
#SBATCH --job-name=fact
#SBATCH --partition=frida
#SBATCH --gres=gpu:B200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=results/%x_%j.out

# The container runs as root with HOME=/root: env HOME= points it back to the real home, where the Hugging Face
# cache and token are.
srun --container-image="$HOME/containers/fact.sqfs" --container-mounts="$HOME:$HOME" --container-workdir="$PWD" \
    env HOME="$HOME" python -m src.main run --config "${1:-configs/llms_bias_bench.yaml}"
