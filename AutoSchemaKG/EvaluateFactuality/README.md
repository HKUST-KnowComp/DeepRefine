# Evaluation for Factuality

## Setup

Our evaluation for factuality is based on the [FELM](https://hkust-nlp.github.io/felm/) benchmark. For installation and dataset preparation, please refer to the [FELM repository](https://github.com/hkust-nlp/felm).

## Usage

**Deploy**

To run the evaluation, please first deploy your model as an OpenAI API compatible service. We use the `openai` package to interact with the model.

**File Replacement**

Then, replace the following file with the provided `eval.py`:

* Replace `felm/eval/eval.py` with the provided `eval.py`.

**Run Evaluation**

Finally, to run the evaluation, execute the following command:

```bash
cd felm/eval
python eval.py --path <path_to_the_felm_dataset> --model <model_name> --method < raw or content> --url <url_to_your_model>
```

## Acknowledgements

This evaluation mainly builds upon the [FELM](https://hkust-nlp.github.io/felm/) benchmark.