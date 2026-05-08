# reafiner ckpt path
CHECKPOINT_PATH="Qwen3-8B-refinement-rl-gbd_f1reward-hotpotqa-64-6-16-1-1e-7"
STEP_NUM="30"

python benchmark/autograph/benchmarking_graph_refiner.py \
    --reafiner_model_name $CHECKPOINT_PATH/global_step_$STEP_NUM/actor/huggingface \
    --refine_max_subset_size 1000 \
    --refine_subset_target_coverage 0.8 \
    --refine_subset_topk 10 \
    --refine_subset_1hop_sample_size 500 \
    --refine_subset_workers 16 \
    --refine \
    --kg_type naive