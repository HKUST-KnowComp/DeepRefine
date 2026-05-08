export HF_HOME=/path_to_your_models/models
export VLLM_CACHE_DIR=/path_to_your_cache/cache
export NCCL_P2P_DISABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

CUDA_VISIBLE_DEVICES=5 vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --host 0.0.0.0 \
  --port 8132 \
  --gpu-memory-utilization 0.65 \
  --tensor-parallel-size 1 \
  --max-model-len 32768