from verl.workers.config import RolloutConfig
from dataclasses import dataclass

@dataclass
class AutoGraphActorConfig(RolloutConfig):
    """Add AutoGraph-specific configuration."""
    use_api:bool =True # always set to true for AutoGraph, otherwise using original VeRL multi-turn training
    rag_method:str = 'subgraph' # available: 'subgraph', 'hipporag2', 'edge', 'tog', 'hipporag' (Use subgraph for Graph Retriever, hipporag2 for Graph-based Text Retriever)
    text_linking:bool= False # Indicate whether it is Graph-based Text Retriever or not
    freeze_answer_api:bool = False # Whether use frozen answer API or use local frozen LLM to generate answers (Set true for 3B model, false for 7B model)
    set_llm_judge_model:bool = False # Whether set the llm_judge_model_name or not
    llm_judge_model_name:str = 'Qwen/Qwen2.5-7B-Instruct' # LLM model name for judging the generated answer (default is 7B)
    # Embedding model used by the Reranker in RAG. This is read in sglang_rollout.py via
    # self.config.get("reranker_model_name", ...), but also needs to be part of the dataclass
    # to avoid unexpected keyword errors during Hydra instantiation.
    reranker_model_name:str = 'Qwen/Qwen3-Embedding-0.6B-batch'
    iterative:bool = True # Whether iteratively construct KG triples for each document or not
    tight:bool = True # Whether use tight retrieval or not (tight retrieval means the number of hops / the number of retrieved documents is equal to the number of hops in the question / number of supporting contexts)
    reward_function:str = 'deducible_reward' # available: f1_reward, deducible_reward, recall_reward
    filter_repetition_rollout:bool = True # Whether filter out the excessive repetition in rollout stage
    filter_repetition_threshold:float = 0.9 # Threshold for filtering excessive repetition in rollout stage
    gen_acc_judge_base_url:str = 'https://yunwu.ai/v1'
    gen_acc_judge_api_key:str = '***'
    gen_acc_judge_model:str = 'deepseek-v3'