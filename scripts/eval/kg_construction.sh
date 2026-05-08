CHECKPOINT_PATH="3b-medium-mix-data-batchsize-64-textlinking-false-deducable-true_p"
STEP_NUM="350"

# Adjust the API url in the python script as needed
python benchmark/autograph/custom_kg_extraction.py --model_name $CHECKPOINT_PATH