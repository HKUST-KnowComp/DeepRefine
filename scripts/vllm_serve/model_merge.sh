CHECKPOINT_PATH="/path_to_your_checkpoints/checkpoints"
STEP_NUM="30"

python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $CHECKPOINT_PATH/global_step_$STEP_NUM/actor \
    --target_dir $CHECKPOINT_PATH/global_step_$STEP_NUM/actor/huggingface