## Evaluation for General Benchmark

We evaluation the general benchmark MMLU under the help of lm-evaluation-harness. 

We change the prompt and response filter method in ./lm-evaluation-harness/lm_eval/tasks/mmlu/generative/_default_template_yaml file to make the assessment more accurate. 

We also provide the subject_score.py in ./evaluation_general/lm-evaluation-harness/results/ to help compute the final score of MMLU based on the subject classification provided in our paper.

To set up the environment and run the evaluation experiments on MMLU, please refer to the README.md in ./lm-evaluation-harness.

For ease of testing, we wrapped our model in an api interface to call. For example, if we deploy the model on port 10086, then the test command running in ./lm-evaluation-harnes should be:
```
lm_eval --model local-chat-completions --model_args model=meta-llama/Llama-3.1-8B-Instruct,base_url=http://0.0.0.0:10086/v1/chat/completions,num_concurrent=1 --tasks mmlu_generative  --apply_chat_template --output_path OUTPUT_DIRECTORY --log_samples
```
