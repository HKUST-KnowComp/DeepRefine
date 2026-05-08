export GEN_ACC_JUDGE_BASE_URL="***"
export GEN_ACC_JUDGE_MODEL="deepseek-v3"
export GEN_ACC_JUDGE_API_KEY="***"

python ./autorefiner/scripts/prepare_rl_data_smallkg.py \
    --input-file your_path_to/hotpot_train_v1.1.json \
    --kgs-base your_path_to/KGs \
    --dataset hotpotqa \
    --gen-acc-threshold 1.0 \
    --force-reprocess \
    --max-samples 5000