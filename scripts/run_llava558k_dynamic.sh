#!/usr/bin/env bash
set -euo pipefail

# Full LLaVA-Pretrain image training without a feature cache.  The caller may
# set SPLITOCULO_PHASE=gan after warmup has produced warmup_best.pth.
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${repo_dir}/.venv-temporal/bin/python"
data_dir="${repo_dir}/data/llava_pretrain_558k"
output_dir="${repo_dir}/checkpoints/llava558k_32b_49x64_dynamic"
phase="${SPLITOCULO_PHASE:-warmup}"
dynamic_batch_size="${SPLITOCULO_DYNAMIC_BATCH_SIZE:-16}"

if [[ "${phase}" != "warmup" && "${phase}" != "gan" ]]; then
    echo "SPLITOCULO_PHASE must be warmup or gan" >&2
    exit 2
fi

common_args=(
    --dynamic
    --data_dir "${data_dir}"
    --qwen_model Qwen/Qwen2.5-VL-32B-Instruct
    --qwen_layer 4
    --qwen_local_files_only
    --qwen_visual_only
    --transmission_tokens 49
    --bottleneck_dim 64
    --dynamic_batch_size "${dynamic_batch_size}"
    --phase "${phase}"
    --device cuda
    --output_dir "${output_dir}"
)

if [[ "${phase}" == "gan" ]]; then
    common_args+=(
        --warmup_checkpoint "${output_dir}/warmup_best.pth"
    )
fi

exec "${python_bin}" "${repo_dir}/scripts/train_gan.py" "${common_args[@]}"
