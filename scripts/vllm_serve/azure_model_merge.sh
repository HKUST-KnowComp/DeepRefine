#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_PATH="/path_to_your_checkpoints/checkpoints"

shopt -s nullglob
steps=("$CHECKPOINT_PATH"/global_step_*)

if [ ${#steps[@]} -eq 0 ]; then
    echo "No global_step_* directories found under: $CHECKPOINT_PATH"
    exit 1
fi

for step_dir in "${steps[@]}"; do
    actor_dir="$step_dir/actor"
    hf_dir="$actor_dir/huggingface"

    if [ ! -d "$actor_dir" ]; then
        echo "Skip (missing actor dir): $step_dir"
        continue
    fi

    echo "==== Processing: $step_dir ===="

    python3 -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "$actor_dir" \
        --target_dir "$hf_dir"

    # Keep only huggingface artifacts after merge.
    find "$actor_dir" -mindepth 1 -maxdepth 1 ! -name "huggingface" -exec rm -rf {} +
    echo "Cleaned: $actor_dir (kept only huggingface/)"
done

echo "All done."