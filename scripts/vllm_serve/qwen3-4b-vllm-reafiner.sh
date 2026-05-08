export HF_HOME=/path_to_your_models/models
export VLLM_CACHE_DIR=/path_to_your_cache/cache
export NCCL_P2P_DISABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

CUDA_VISIBLE_DEVICES=3 vllm serve Qwen3-4B-refinement-rl-gbd_f1reward-hotpotqa-32-6-16-1-5e-7/global_step_50/actor/huggingface  \
  --host 0.0.0.0 \
  --port 8133 \
  --gpu-memory-utilization 0.85 \
  --tensor-parallel-size 1 \
  --max-model-len 16384