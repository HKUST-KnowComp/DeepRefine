# reafiner ckpt path
CHECKPOINT_PATH="Qwen3-4B-refinement-rl-gbd_f1reward-hotpotqa-32-6-16-1-5e-7"

python benchmark/autograph/benchmarking_text_refiner.py  --reafiner_model_name $CHECKPOINT_PATH --refine --kg_type naive