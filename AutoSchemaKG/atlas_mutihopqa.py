from configparser import ConfigParser
from openai import OpenAI
from atlas_rag.retriever import HippoRAG2Retriever, TogRetriever, HippoRAGRetriever
from atlas_rag.vectorstore.embedding_model import NvEmbed
from atlas_rag.vectorstore.create_graph_index import create_embeddings_and_index
from atlas_rag.logging import setup_logger
from atlas_rag.llm_generator import LLMGenerator
from atlas_rag.evaluation import BenchMarkConfig, RAGBenchmark
from transformers import AutoModel
from sentence_transformers import SentenceTransformer
import torch
import argparse

argparser = argparse.ArgumentParser(description="Run Atlas Multi-hop QA Benchmark")
argparser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use')
argparser.add_argument('--keyword', type=str, default='2wikimultihopqa', help='Keyword for the dataset')
argparser.add_argument('--graph_type', type=str, default='oie', choices=['atlas', 'oie', 'stanford_oie'], help='Type of graph to use for the retriever')

args = argparser.parse_args()

def main():
    num_gpus = torch.cuda.device_count()
    print("Number of GPUs available:", num_gpus)
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Load SentenceTransformer model
    encoder_model_name = "nvidia/NV-Embed-v2"
    # sentence_model = AutoModel.from_pretrained(encoder_model_name, trust_remote_code=True, device_map="auto")
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    sentence_model = SentenceTransformer('nvidia/NV-Embed-v2', trust_remote_code=True, device=device)
    sentence_model.max_seq_length = 32768
    sentence_model.tokenizer.padding_side="right"
    sentence_encoder = NvEmbed(sentence_model)

    # Load OpenRouter API key and initialize LLMGenerator
    config = ConfigParser()
    config.read('config.ini')
    reader_model_name = "meta-llama/Llama-3.3-70B-Instruct"
    client = OpenAI(
        base_url="https://api.deepinfra.com/v1/openai",
        api_key=config['settings']['DEEPINFRA_API_KEY'],
    )
    llm_generator = LLMGenerator(client=client, model_name=reader_model_name)

    # Create embeddings and index
    working_directory = f'/data/httsangaj/atlas/{args.graph_type}'
    data = create_embeddings_and_index(
        sentence_encoder=sentence_encoder,
        model_name='nvidia/NV-Embed-v2',
        working_directory=working_directory,
        keyword=args.keyword,
        include_concept=True,
        include_events=True,
        normalize_embeddings=True,
        text_batch_size=1,
        node_and_edge_batch_size=1,
    )

    # Configure benchmarking
    benchmark_config = BenchMarkConfig(
        dataset_name=args.keyword,
        question_file=f"benchmark_data/{args.keyword}.json",
        include_concept=True,
        include_events=True,
        reader_model_name=reader_model_name,
        encoder_model_name=encoder_model_name,
        number_of_samples=-1,  # -1 for all samples    )
    )
    # Set up logger
    logger = setup_logger(benchmark_config, log_path = f"./log/{args.keyword}_{args.graph_type}_benchmark.log")

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
    benchmark.run([hipporag2_retriever, hipporag_retriever], llm_generator=llm_generator)

if __name__ == "__main__":
    main()