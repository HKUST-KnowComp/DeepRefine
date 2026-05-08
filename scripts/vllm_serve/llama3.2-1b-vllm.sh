export HF_HOME=/path_to_your_models/models
export VLLM_CACHE_DIR=/path_to_your_cache/cache
export NCCL_P2P_DISABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

CUDA_VISIBLE_DEVICES=6 vllm serve unsloth/Llama-3.2-1B-Instruct \
  --host 0.0.0.0 \
  --port 8130 \
  --gpu-memory-utilization 0.4 \
  --tensor-parallel-size 1 \
  --max-model-len 4096