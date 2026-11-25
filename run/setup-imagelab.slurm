#!/usr/bin/env bash
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH -J modes-setup
#SBATCH --output=/scratch/aharun/slurm/metu__imagelab-denge__kalkan-akbas/scratchbook-symmetry/MoDES/setup-%j.out
# Venv on top of the FastMMoE image (torch 2.8, transformers 4.57.1, flash-attn 2.8.3, qwen-vl-utils 0.0.14) with lmms-eval at the last
# commit before the MoDES release (2026-02-19), then the Qwen3-VL-30B-A3B-Instruct weights and the GQA testdev parquets.
set -euo pipefail
P=metu__imagelab-denge__kalkan-akbas/scratchbook-symmetry
DATA=/scratch/aharun/data/$P
SIF=$DATA/containers/fastmmoe-a0c6d94.sif
VENV=$DATA/envs/modes
LMMS=ee31b5a65ea79c8a2a4c9d6fcab736751d39513d
STORE=$DATA/modes-storage
export HF_HOME=$DATA/hf-home
mkdir -p "$STORE/models" "$STORE/datasets/GQA" "$HF_HOME" "$DATA/envs"
X=(singularity exec --cleanenv --bind /scratch/aharun --env HF_HOME="$HF_HOME" --env HF_HUB_ENABLE_HF_TRANSFER=0 "$SIF")
"${X[@]}" python -m pip freeze | grep -iE '^(torch|torchvision|transformers|accelerate|flash.attn|numpy)[= @]' > "$DATA/envs/modes-constraints.txt"
cat "$DATA/envs/modes-constraints.txt"
[ -x "$VENV/bin/python" ] || "${X[@]}" python -m venv --system-site-packages "$VENV"
"${X[@]}" "$VENV/bin/python" -m pip install -q -c "$DATA/envs/modes-constraints.txt" \
    "lmms_eval @ git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git@$LMMS" loguru tiktoken blobfile ipdb
"${X[@]}" "$VENV/bin/python" -c 'import torch, transformers, lmms_eval, flash_attn; from transformers import Qwen3VLMoeForConditionalGeneration; print("import ok", torch.__version__, transformers.__version__, flash_attn.__version__)'
"${X[@]}" "$VENV/bin/python" -c 'import sys; from huggingface_hub import snapshot_download as s
s("Qwen/Qwen3-VL-30B-A3B-Instruct", local_dir=sys.argv[1] + "/models/Qwen3-VL-30B-A3B-Instruct")
s("lmms-lab/GQA", repo_type="dataset", local_dir=sys.argv[1] + "/datasets/GQA", allow_patterns=[
  "testdev_balanced_images/testdev-00000-of-00001.parquet", "testdev_balanced_instructions/testdev-00000-of-00001.parquet"])' "$STORE"
"${X[@]}" "$VENV/bin/python" -c 'from datasets import load_dataset; d = load_dataset("lmms-lab/ChartQA", split="test"); print("chartqa", len(d))'
du -sh "$STORE"/models/* "$STORE"/datasets/*
echo "DONE $(date +%T)"
