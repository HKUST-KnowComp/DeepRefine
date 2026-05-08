export HF_HOME=/data/haoyuhuang/data/model
export VLLM_CACHE_DIR=/home/haoyuhuang/.cache/vllm
export NCCL_P2P_DISABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-3B-Instruct \
  --host 0.0.0.0 \
  --port 8129 \
  --gpu-memory-utilization 0.78 \
  --tensor-parallel-size 1 \
  --max-model-len 16384