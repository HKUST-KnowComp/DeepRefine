export HF_HOME=/path_to_your_models/models
export VLLM_CACHE_DIR=/path_to_your_cache/cache
export NCCL_P2P_DISABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8129 \
  --gpu-memory-utilization 0.95 \
  --tensor-parallel-size 1 \
  --max-model-len 32768