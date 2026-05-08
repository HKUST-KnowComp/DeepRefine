from configparser import ConfigParser
from openai import OpenAI
from atlas_rag.retriever import HippoRAG2Retriever, TogRetriever, HippoRAGRetriever
from atlas_rag.vectorstore.embedding_model import Qwen3Emb
from atlas_rag.vectorstore.create_graph_index import create_embeddings_and_index
from atlas_rag.logging import setup_logger
from atlas_rag.llm_generator import LLMGenerator
from atlas_rag.evaluation import BenchMarkConfig, RAGBenchmark
from transformers import AutoModel
from sentence_transformers import SentenceTransformer
import torch
import argparse
import time

argparser = argparse.ArgumentParser(description="Run Atlas Multi-hop QA Benchmark")
argparser.add_argument('--keyword', type=str, default='2wikimultihopqa', help='Keyword for the dataset')
argparser.add_argument('--graph_type', type=str, default='base', choices=['base', 'rl'], help='Type of graph to use')
args = argparser.parse_args()

def main():
    num_gpus = torch.cuda.device_count()
    print("Number of GPUs available:", num_gpus)
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Load SentenceTransformer model
    encoder_model_name = "Qwen/Qwen3-Embedding-0.6B"
    sentence_model = OpenAI(
        base_url="http://0.0.0.0:8128/v1",
        api_key="EMPTY KEY",
    )
    sentence_encoder = Qwen3Emb(sentence_model)

    reader_model_name = "Qwen/Qwen2.5-7B-Instruct"
    client = OpenAI(
        base_url="http://0.0.0.0:8129/v1",
        api_key="EMPTY KEY",
    )
    llm_generator = LLMGenerator(client=client, model_name=reader_model_name)

    # Create embeddings and index
    working_directory = f'/data/autograph/{args.graph_type}'
    data = create_embeddings_and_index(
        sentence_encoder=sentence_encoder,
        model_name=encoder_model_name,
        working_directory=working_directory,
        keyword=args.keyword,
        include_concept=False,
        include_events=False,
        normalize_embeddings=False,
        text_batch_size=512,
        node_and_edge_batch_size=512,
    )

    # Configure benchmarking
    benchmark_config = BenchMarkConfig(
        dataset_name=args.keyword,
        question_file=f"custom_benchmark/qa_data/{args.keyword}.json",
        result_dir="custom_result/",
        include_concept=False,
        include_events=False,
        reader_model_name=reader_model_name,
        encoder_model_name=encoder_model_name,
        number_of_samples=-1,  # -1 for all samples
    )
    # Set up logger
    logger = setup_logger(benchmark_config, log_path = f"./log/{args.keyword}_{args.graph_type}_{time.time()}_benchmark.log")

    # Initialize HippoRAG2Retriever
    hipporag2_retriever = HippoRAG2Retriever(
        llm_generator=llm_generator,
        sentence_encoder=sentence_encoder,
        data=data,
        logger=logger
    )
    tog_retriever = TogRetriever(
        llm_generator=llm_generator,
        sentence_encoder=sentence_encoder,
        data=data
        )
    hipporag_retriever = HippoRAGRetriever(
        llm_generator=llm_generator,
        sentence_encoder=sentence_encoder,
        data=data,
        logger=logger
    )

    # Start benchmarking
    benchmark = RAGBenchmark(config=benchmark_config, logger=logger)
    benchmark.run([hipporag2_retriever, hipporag_retriever, tog_retriever], llm_generator=llm_generator)

if __name__ == "__main__":
    main()