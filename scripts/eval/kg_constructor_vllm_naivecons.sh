# Adjust CHECKPOINT_PATH and STEP_NUM as needed

CUDA_VISIBLE_DEVICES=7 vllm serve models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1 \
    --host 0.0.0.0 \
    --port 8111 \
    --gpu-memory-utilization 0.65 \
    --tensor-parallel-size 1 \
    --max-model-len 16384