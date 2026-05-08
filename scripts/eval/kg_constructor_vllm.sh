# Adjust CHECKPOINT_PATH and STEP_NUM as needed
CHECKPOINT_PATH="3b-medium-mix-data-batchsize-64-textlinking-false-deducable-true_p"
STEP_NUM="350"

CUDA_VISIBLE_DEVICES=0 vllm serve $CHECKPOINT_PATH \
    --host 0.0.0.0 \
    --port 8111 \
    --gpu-memory-utilization 0.8 \
    --tensor-parallel-size 1 \
    --max-model-len 16384