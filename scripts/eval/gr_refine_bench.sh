# reafiner ckpt path
CHECKPOINT_PATH="Qwen/Qwen2.5-7B-Instruct"

python benchmark/autograph/benchmarking_graph_refiner.py  --reafiner_model_name $CHECKPOINT_PATH --refine --kg_type naive