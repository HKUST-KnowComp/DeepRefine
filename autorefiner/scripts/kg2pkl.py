"""
Transform the constructed AutoSchemaKG to pkl file
"""
import pickle
import os
import sys
from pathlib import Path
from atlas_rag.vectorstore.create_graph_index import create_embeddings_and_index
from atlas_rag.vectorstore.embedding_model import Qwen3Emb
from atlas_rag.llm_generator import LLMGenerator
from openai import OpenAI

def main():
    checkpoint_path = "/data/haoyuhuang/data/AtlasTune/checkpoints/20251211_234743_qwen2.5-3B-autograph-easy-docsize15-textlinkingFalse-loose/global_step_350/actor/huggingface"
    kg_name = "hotpotqa"
    if checkpoint_path == "Qwen/Qwen2.5-3B-Instruct" or checkpoint_path == "Qwen/Qwen2.5-7B-Instruct" or checkpoint_path == 'meta-llama/Llama-3.2-3B-Instruct' or checkpoint_path == 'meta-llama/Llama-3.2-1B-Instruct':
        output_directory = f'/data/haoyuhuang/data/AtlasTune/checkpoints/{checkpoint_path.split("/")[-1]}/constructed_kg/{kg_name}_output'
    else:
        output_directory = f'{checkpoint_path}/constructed_kg/{kg_name}_output'
    output_directory = Path(output_directory)  # Convert to Path object
    output_pkl_path = output_directory / 'original_kg.pkl'
    
    encoder_model_name = "Qwen/Qwen3-Embedding-0.6B"
    sentence_model = OpenAI(
        base_url="http://0.0.0.0:8128/v1",
        api_key="EMPTY KEY",
    )
    sentence_encoder = Qwen3Emb(sentence_model)

    data = create_embeddings_and_index(
                sentence_encoder=sentence_encoder,
                model_name=encoder_model_name,
                working_directory=output_directory,
                keyword=kg_name,
                include_concept=False,
                include_events=False,
                normalize_embeddings=False,
                text_batch_size=512,
                node_and_edge_batch_size=512,
                use_flat_index=True
            )
    # save to pkl file
    with open(output_pkl_path, "wb") as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    main()