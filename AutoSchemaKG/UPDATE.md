# UPDATE

## Custom prompt
In order to use custom prompt for KG construction, you need to define:
1. custom prompt

example:

{
    "en":{
        "system": "You are a helpful assistant",
        "triple_extraction": "You are an expert knowledge graph constructor. Your task is to extract list of knowledge graph triples.\nEach triple should be a JSON object with three keys:\n1.  `subject`,  `relation`, `object`. Do not include any text other than the list of JSON output."
    }
}

2. schema json (for verifing the generate json match your requirement or not)

{
    "triple_extraction": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "subject": { "type": "string" },
                "relation": { "type": "string" },
                "object": { "type": "string" }
            },
            "required": ["subject", "relation", "object"]
        }
    }
}

## Benchmarking with custom / atlas KG
In order to use vllm, please pip install vllm for faster inferencing
Run the scripts in example_scripts/custom_extraction/vllm_openai_server_hosting_script to host both llm and embedding.
If you are using Qwen3 model, you can set reasoning_effort in LLM generator to enable thinking token or not. Default should be reasoning_effort = None.

(Note: set the working_directory for index to the directory you stored the graph, in order for it to find the graphml file. The created embeddings and index will also be stored there)
The other follows the original pipeline in ATLAS benchmark, you can refer to example_scripts/custom_extraction/benchmarking.py for reference.