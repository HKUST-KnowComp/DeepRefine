# run on 2xA100
# make sure your current working directory is the root of the project
# export CUDA devices
#!/bin/bash
export CUDA_VISIBLE_DEVICES=3,4
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HYDRA_FULL_ERROR=1
export HF_HUB_OFFLINE=0
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray}"
mkdir -p "$RAY_TMPDIR"

WANDB_API_KEY=""
GEN_ACC_JUDGE_BASE_URL="http://127.0.0.1:8129/v1"
GEN_ACC_JUDGE_API_KEY="EMPTY"
GEN_ACC_JUDGE_MODEL="Qwen/Qwen2.5-7B-Instruct"
# Print the values (for debugging)
export WANDB_API_KEY
set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/config"

# parameters
DOC_SIZE=15 # available: 8,12,15
TEXT_LINKING="False" # available: True, False
ITERATIVE="True"
REFINEMTN="True"    # False for AR1
BATCH_SIZE=64
MICRO_BATCH_SIZE=1
MINI_BATCH_SIZE=16
GROUP_SIZE=16
LR=1e-6

TRAIN_DATA="$PROJECT_DIR/data/hotpopqa_dev_all.parquet"
VAL_DATA="$PROJECT_DIR/data/hotpotqa_valid_refinement_gbd_smallkg_chunk_genacc_5000.parquet"
MAX_ASSISTANT_TURN=6
MAX_USER_TURN=6

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

reward_fn_file_path="verl/third_party/autograph_r1/gbd_reward.py"
reward_function="gbd_reward"

EXPERIMENT_NAME="exp-Qwen3-1.7B-deeprefine-gbd_reward-cr-hotpotqa-${BATCH_SIZE}-${GROUP_SIZE}-${MINI_BATCH_SIZE}-${MICRO_BATCH_SIZE}-${LR}"
CHECKPOINT_DIR="${CHECKPOINT_ROOT:-$PROJECT_DIR/checkpoints}/${EXPERIMENT_NAME}"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='autograph_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=True \
    data.train_batch_size=$BATCH_SIZE \
    data.val_batch_size=$BATCH_SIZE \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.shuffle=True \
    data.truncation='middle' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.02 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=1e-4 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=1e-3 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.autograph_mode='autorefine' \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$GROUP_SIZE \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$MAX_ASSISTANT_TURN \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$MAX_USER_TURN \
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=disable \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path='config/interaction_config/refinement_interaction_config.yaml' \
    actor_rollout_ref.rollout.multi_turn.use_inference_chat_template=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.rollout.gen_acc_judge_base_url="$GEN_ACC_JUDGE_BASE_URL" \
    actor_rollout_ref.rollout.gen_acc_judge_api_key="$GEN_ACC_JUDGE_API_KEY" \
    actor_rollout_ref.rollout.gen_acc_judge_model="$GEN_ACC_JUDGE_MODEL" \
    critic.strategy=fsdp2 \
    reward_model.strategy=fsdp2 \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger=['console','wandb'] \
    trainer.project_name='deeprefine' \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.total_training_steps=200 \
    trainer.save_freq=10 \
    trainer.test_freq=-1 \
    trainer.ray_wait_register_center_timeout=36000 \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA"  \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    custom_reward_function.path="$reward_fn_file_path" \
    actor_rollout_ref.rollout._target_=verl.third_party.autograph_r1.autograph_config.AutoGraphActorConfig \
    actor_rollout_ref.rollout.use_api=True \
    actor_rollout_ref.rollout.rag_method='re_edge' \
    actor_rollout_ref.rollout.text_linking=$TEXT_LINKING \
    actor_rollout_ref.rollout.freeze_answer_api=True \
    actor_rollout_ref.rollout.iterative=$ITERATIVE \
    actor_rollout_ref.rollout.tight=False \
    actor_rollout_ref.rollout.reward_function=$reward_function \
    actor_rollout_ref.rollout.set_llm_judge_model=True \
    actor_rollout_ref.rollout.reranker_model_name='Qwen/Qwen3-Embedding-0.6B-batch' \
    actor_rollout_ref.rollout.llm_judge_model_name='Qwen/Qwen2.5-7B-Instruct' \
    actor_rollout_ref.rollout.skip_tokenizer_init=False \
    custom_reward_function.reward_kwargs.triple_repetition_penalty=0.0 \
    custom_reward_function.reward_kwargs.q00_gain=0.0 \
    custom_reward_function.reward_kwargs.q01_gain=1.0 \
    custom_reward_function.reward_kwargs.q10_gain=-1.0 \
    custom_reward_function.reward_kwargs.q11_gain=0.0