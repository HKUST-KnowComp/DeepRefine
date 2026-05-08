#!/usr/bin/env bash
VALLE_DIR="/path_to_your_models/models"

export HF_HOME=${VALLE_DIR}
export TRANSFORMERS_CACHE=${VALLE_DIR}

python3 - <<'PY'
from transformers import AutoModelForCausalLM

AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
)
print("Done.")
PY

echo "Done: $VALLE_DIR"