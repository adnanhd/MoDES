#!/usr/bin/env bash
#SBATCH -p debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --mem=180G
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --time=3-00:00:00
#SBATCH --no-requeue
#SBATCH -J modes-gate
#SBATCH --output=/scratch/aharun/slurm/metu__imagelab-denge__kalkan-akbas/scratchbook-symmetry/MoDES/gate-%j.out
# MoDES gate on one RTX PRO 6000: README steps 1-3 for Qwen3-VL-30B-A3B-Instruct with one process instead of eight:
# calibration (GQA, KL, 1024 samples, batch 8), frontier search (rho 0.85, grid 100, exp), ChartQA at k=8 and at MoDES.
set -euo pipefail
P=metu__imagelab-denge__kalkan-akbas/scratchbook-symmetry
DATA=/scratch/aharun/data/$P
SIF=$DATA/containers/fastmmoe-a0c6d94.sif
VENV=$DATA/envs/modes
CODE=/scratch/aharun/scratch/$P/MoDES-${SHA:?}
OUT=/scratch/aharun/results/$P/MoDES/gate
RHO=${RHO:-0.85}
test -z "$(git -C "$CODE/MoDES" status --porcelain --untracked-files=no)" || { echo "STOP: tracked changes in $CODE/MoDES"; exit 1; }
mkdir -p "$OUT"
ln -sfn "$DATA/modes-storage/models" "$OUT/models"; ln -sfn "$DATA/modes-storage/datasets" "$OUT/datasets"
cd "$CODE/MoDES"; ln -sfn "$OUT" storage
M=storage/models/Qwen3-VL-30B-A3B-Instruct
X=(singularity exec --nv --cleanenv --bind /scratch/aharun --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?}"
   --env HF_HOME="$DATA/hf-home" --env PYTHONPATH="$CODE:$CODE/MoDES" --env HF_ALLOW_CODE_EVAL=1
   --env HF_DATASETS_TRUST_REMOTE_CODE=true --env PYTHONDONTWRITEBYTECODE=1 "$SIF")
A=("$VENV/bin/accelerate" launch --num_processes 1 --main_process_port 5678)
echo "code $(git rev-parse --short HEAD); out $OUT; rho $RHO"; hostname; nvidia-smi -L
IMP=storage/importance/gqa/Qwen3-VL-30B-A3B-Instruct/kl_0_1024.pkl
[ -e "$IMP" ] || "${X[@]}" "${A[@]}" get_layer_importance_ddp.py --name_or_path $M --save_dir storage/importance \
    --dataset gqa --loss_type kl --batch_size 8 --num_samples 1024
ls storage/importance/gqa/Qwen3-VL-30B-A3B-Instruct/
TAU=$(ls storage/search/gqa/Qwen3-VL-30B-A3B-Instruct/kl_0_1024_${RHO}_*grid100.pkl 2>/dev/null | head -1 || true)
if [ -z "$TAU" ]; then
    "${X[@]}" "${A[@]}" grid_search_tau_ddp.py --name_or_path $M --save_dir storage/search/ --dataset gqa --loss_type kl \
        --batch_size 8 --num_samples 1024 --layer_importance_path "$IMP" --target_skip_proportion "$RHO" \
        --grid_num 100 --grid_map exp
    TAU=$(ls storage/search/gqa/Qwen3-VL-30B-A3B-Instruct/kl_0_1024_${RHO}_*grid100.pkl | head -1)
fi
echo "tau file $TAU"
for ARM in modes default; do
    MA="pretrained=$M"; [ $ARM = default ] || MA="$MA,tau_skip_path=$TAU,layer_importance_path=$IMP"
    [ -e "storage/eval/$ARM/.done" ] || { "${X[@]}" "${A[@]}" eval/qwen3.py --model qwen3_vl --model_args "$MA" \
        --tasks chartqa --batch_size 1 --output_path "storage/eval/$ARM/" && touch "storage/eval/$ARM/.done"; }
done
grep -rh '"relaxed_overall' storage/eval/*/ | head
echo "DONE $(date +%T)"
