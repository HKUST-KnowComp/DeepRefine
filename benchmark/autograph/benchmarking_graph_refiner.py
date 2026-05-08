from configparser import ConfigParser
from openai import OpenAI
from atlas_rag.retriever import *
from atlas_rag.vectorstore.embedding_model import Qwen3Emb
from atlas_rag.vectorstore.create_graph_index import create_embeddings_and_index
from atlas_rag.logging import setup_logger
from atlas_rag.llm_generator import LLMGenerator
from atlas_rag.llm_generator.generation_config import GenerationConfig
from atlas_rag.evaluation import BenchMarkConfig, RAGBenchmark
from transformers import AutoModel
from sentence_transformers import SentenceTransformer
from atlas_rag.retriever.inference_config import InferenceConfig
import torch
import argparse
import time
import os
import sys
import json
import pickle
import asyncio
from dataclasses import asdict
from tqdm import tqdm
sys.path.append('../../')
from autorefiner.src.reafiner import Reafiner, RetrievalStepResult


def _refinement_result_to_jsonable(
    sample: dict,
    final_answer,
    refinement_result,
) -> dict:
    """Serialize RefinementResult for JSONL (drop non-JSON-safe callables)."""
    base = {
        "sample": sample,
        "final_answer": final_answer,
    }
    if refinement_result is None:
        base["refinement_result"] = None
        return base

    hist = []
    for step in refinement_result.interaction_history:
        if isinstance(step, RetrievalStepResult):
            hist.append(
                {
                    "num_hops": step.num_hops,
                    "base_top_k": step.base_top_k,
                    "query": step.query,
                    "retrieved_subgraph": step.retrieved_subgraph,
                    "raw_response": step.raw_response,
                    "answerable": step.answerable,
                    "answer": step.answer,
                }
            )
        else:
            hist.append(str(step))

    base["refinement_result"] = {
        "query": refinement_result.query,
        "history_horizon_size": refinement_result.history_horizon_size,
        "interaction_history": hist,
        "error_abduction_reason": refinement_result.error_abduction_reason,
        "original_subgraph": refinement_result.original_subgraph,
        "refined_subgraph": refinement_result.refined_subgraph,
        "refinement_action_raw": refinement_result.refinement_action_raw,
        "refinement_action_count": len(refinement_result.refinement_action_list),
    }
    return base


argparser = argparse.ArgumentParser(description="Run Atlas Multi-hop QA Benchmark")
argparser.add_argument("--refine", action="store_true", help="Refine the KG")
argparser.add_argument("--port", type=int, default=8110, help="Port number for LLM server")
# set store true if using upperbound retrieval
argparser.add_argument("--use_upperbound", action="store_true", help="Use upperbound retrieval")
# set store true if using dense retrieval only
argparser.add_argument("--use_dense_only", action="store_true", help="Use dense retrieval only")
argparser.add_argument("--reader_model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Reader model name")
argparser.add_argument("--reafiner_model_name", type=str, default="qwen3-8b", help="Reader model name")
argparser.add_argument("--kg_type", type=str, default="naive", help="KG type: naive, ar1, graphify")
argparser.add_argument("--refine_max_subset_size", type=int, default=200, help="Max number of queries to refine KG (coverage selection).")
argparser.add_argument("--refine_subset_target_coverage", type=float, default=0.8, help="Stop selecting when coverage reached.")
argparser.add_argument("--refine_subset_topk", type=int, default=50, help="TopK triples per query used for coverage selection.")
argparser.add_argument("--refine_subset_1hop_sample_size", type=int, default=100, help="Number of 1-hop sampled triples per query used for coverage.")
argparser.add_argument("--refine_subset_workers", type=int, default=8, help="Parallel workers for subset selection stage.")
argparser.add_argument(
    "--llm_enable_thinking",
    action="store_true",
    help=(
        "If set, pass enable_thinking=True for the server's tokenizer.apply_chat_template "
        "(HTTP extra_body: chat_template_kwargs, and top-level enable_thinking when needed by the gateway). "
        "Default: False. Unsupported template kwargs are only retried without them on HTTP 400, never on 429."
    ),
)
args = argparser.parse_args()


def _llm_default_config() -> GenerationConfig:
    """GenerationConfig defaults: template kwargs for remote apply_chat_template, not sampling params."""
    cfg = GenerationConfig()
    cfg.chat_template_kwargs = {"enable_thinking": args.llm_enable_thinking}
    return cfg

kg_names = ["2wikimultihopqa", "musique", "hotpotqa", "2021wiki", "locomo"]

async def main():
    for kg_name in kg_names:
        # Load SentenceTransformer model
        encoder_model_name = "Qwen/Qwen3-Embedding-0.6B"
        sentence_model = OpenAI(
            base_url="http://0.0.0.0:8128/v1",
            api_key="EMPTY KEY",
        )
        sentence_encoder = Qwen3Emb(sentence_model)

        if "checkpoints" not in args.reafiner_model_name:
            # freeze model vllm
            reafiner_model_name = args.reafiner_model_name
            reafiner_client = OpenAI(
                base_url="http://0.0.0.0:8132/v1",
                api_key="EMPTY KEY",
            )
            reafiner_llm_generator = LLMGenerator(
                client=reafiner_client,
                model_name=reafiner_model_name,
                default_config=_llm_default_config(),
            )
        else:
            # RL trained model vllm
            reafiner_model_name = args.reafiner_model_name
            reafiner_client = OpenAI(
                base_url="http://0.0.0.0:8132/v1",
                api_key="EMPTY KEY",
            )
            reafiner_llm_generator = LLMGenerator(
                client=reafiner_client,
                model_name=reafiner_model_name,
                default_config=_llm_default_config(),
            )

        reader_model_name = args.reader_model_name
        reader_client = OpenAI(
            base_url="http://0.0.0.0:8129/v1",
            api_key="EMPTY KEY",
        )
        reader_llm_generator = LLMGenerator(
            client=reader_client,
            model_name=reader_model_name,
            default_config=_llm_default_config(),
        )
        
        # save under the reafiner model name dict
        if "checkpoints" not in args.reafiner_model_name:
            if reafiner_model_name == "qwen3-8b":
                if args.kg_type == "naive":
                    output_directory = f'/data/haoyuhuang/model/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/constructed_kg_naive/{kg_name}_output'
                elif args.kg_type == "ar1":
                    output_directory = f'/data/haoyuhuang/model/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/constructed_kg_ar1/{kg_name}_output'
                elif args.kg_type == "graphify":
                    output_directory = f'/data/haoyuhuang/model/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/constructed_kg_graphify/{kg_name}_output'
        else:
            if args.kg_type == "naive":
                output_directory = f'{reafiner_model_name}/constructed_kg_naive/{kg_name}_output'
            elif args.kg_type == "ar1":
                output_directory = f'{reafiner_model_name}/constructed_kg_ar1/{kg_name}_output'
            elif args.kg_type == "graphify":
                output_directory = f'{reafiner_model_name}/constructed_kg_graphify/{kg_name}_output'

        # load graph data
        if not args.use_upperbound:
            data = create_embeddings_and_index(
                sentence_encoder=sentence_encoder,
                model_name=encoder_model_name,
                working_directory=output_directory,
                keyword=kg_name,
                include_concept=False,
                include_events=False,
                normalize_embeddings=False,
                text_batch_size=256,
                node_and_edge_batch_size=256,
                use_flat_index=True
            )
        # Configure benchmarking
        if kg_name == "2021wiki":
            qa_names = ["nq", "popqa"]
        else:
            qa_names = [kg_name]
        for qa_name in qa_names:
            # refine the KG
            if args.refine:
                if os.path.exists(f"{output_directory}/refined_kg_{reafiner_model_name.replace('/', '_')}.pkl"):
                    print(f"\033[94m Found refined KG in {output_directory}/refined_kg_{reafiner_model_name.replace('/', '_')}.pkl \033[0m")
                    with open(f"{output_directory}/refined_kg_{reafiner_model_name.replace('/', '_')}.pkl", "rb") as f:
                        data = pickle.load(f)
                else:
                    reafiner = Reafiner(
                        data=data,
                        sentence_encoder=sentence_encoder,
                        llm_generator=reafiner_llm_generator,
                        base_top_k=10,
                        max_hops=4,
                        max_triple_num=70,
                        max_triple_num_by_step=[10, 30, 50, 70],
                        history_horizon_size=4,
                        if_gen_answer=False,
                    )
                    question_file=f"../{qa_name}.json"
                    with open(question_file, "r") as f:
                        query_data = json.load(f)
                        query_data = query_data[:1000]
                    # -------- Coverage-based subset selection (方案A, with 1-hop sampling) --------
                    print(f"[Selecting refine subset]")
                    selected_queries, subset_stats = reafiner.select_refine_subset(
                        query_data=query_data,
                        max_subset_size=args.refine_max_subset_size,
                        target_coverage=args.refine_subset_target_coverage,
                        retrieve_topk=args.refine_subset_topk,
                        one_hop_sample_size=args.refine_subset_1hop_sample_size,
                        selection_workers=args.refine_subset_workers,
                    )
                    print(
                        f"\033[94m Selected {len(selected_queries)} / {len(query_data)} queries "
                        f"(coverage={subset_stats['covered']}/{subset_stats['universe']}={subset_stats['coverage_ratio']:.3f}) \033[0m"
                    )

                    refinement_log_path = os.path.join(
                        output_directory,
                        f"refinement_results_{reafiner_model_name.replace('/', '_')}_{qa_name}_{int(time.time())}.jsonl",
                    )
                    print(f"\033[94m Refinement trace log: {refinement_log_path} \033[0m")

                    # -------- Refine KG only using selected subset --------
                    with open(refinement_log_path, "w", encoding="utf-8") as refinement_log_f:
                        for sample in tqdm(selected_queries, desc="Refining KG (selected subset)"):
                            query = sample["question"]
                            final_answer, refined_kg_data, refinement_result = reafiner.refine(query=query)
                            record = _refinement_result_to_jsonable(
                                sample, final_answer, refinement_result
                            )
                            refinement_log_f.write(
                                json.dumps(record, ensure_ascii=False) + "\n"
                            )
                            refinement_log_f.flush()
                            print(f"Refined KG: {reafiner.kg}")
                            n_steps = (
                                len(refinement_result.interaction_history)
                                if refinement_result is not None
                                else 0
                            )
                            print(f"\033[94m [Total Steps: {n_steps}] \033[0m")
                    data = reafiner.data
                    # TODO: add the passage node to the KG
                    text_id_list = list(reafiner.text_id_to_node_name.keys())
                    for text_id in text_id_list:
                        reafiner.kg.add_node(
                            text_id,
                            file_id=text_id,
                            id=reafiner._safe_sanitize(reafiner.text_id_to_node_name[text_id]),
                            type="passage"
                        )
                    for node_id in list(reafiner.node_list):
                        if reafiner.node_id_to_file_id[node_id] is not None:
                            reafiner.kg.add_edge(
                                node_id,
                                reafiner.node_id_to_file_id[node_id],
                                relation="mention in",
                                type="Source"
                            )
                    print(f"Refined KG (w/ passage nodes): {reafiner.kg}")
                    data['KG'] = reafiner.kg
                # save the data file for repeatedly using
                # Use pickle to save complex objects (NetworkX graph, FAISS indices, numpy arrays)
                if args.refine:
                    with open(f"{output_directory}/refined_kg_{reafiner_model_name.replace('/', '_')}.pkl", "wb") as f:
                        pickle.dump(data, f)
                    print(f"Refined KG data saved to {output_directory}/refined_kg_{reafiner_model_name.replace('/', '_')}.pkl")

            inference_config = InferenceConfig(keyword=qa_name)
            # get the parent directory of output_directory
            base_dir = '/'.join(output_directory.split('/')[:-2])
            if args.use_upperbound:
                base_dir = base_dir + "_upperbound"
            if args.use_dense_only:
                base_dir = base_dir + "_dense"
            benchmark_config = BenchMarkConfig(
                dataset_name=qa_name,
                question_file=f"../{qa_name}.json",
                result_dir=f"{base_dir}/benchmark/graph_retrieval",
                include_concept=False,
                include_events=False,
                reader_model_name=reader_model_name,
                encoder_model_name=encoder_model_name,
                number_of_samples=1000,  # -1 for all samples
                upper_bound_mode=args.use_upperbound,
                topN=10
            )
            # Set up logger
            logger = setup_logger(benchmark_config, 
                                  log_path = f"{base_dir}/benchmark/graph_retrieval/{qa_name}_{time.time()}_benchmark.log")
            logger.info(f"INFERENCE CONFIG: {inference_config}")
            logger.info(f"BENCHMARK CONFIG: {benchmark_config}")
            if args.use_upperbound:
                from atlas_rag.retriever.upper_bound_retriever import UpperBoundRetriever
                upperbound_retriever = UpperBoundRetriever()
                benchmark = RAGBenchmark(config=benchmark_config, logger=logger)
                await benchmark.run_async([upperbound_retriever], llm_generator=reader_llm_generator)
            if args.use_dense_only:
                # Initialize DenseRetriever
                dense_retriever = SimpleTextRetriever(
                    passage_dict=data["text_dict"],
                    sentence_encoder=sentence_encoder,
                    data=data,
                    inference_config=inference_config,
                )
                benchmark = RAGBenchmark(config=benchmark_config, logger=logger)
                await benchmark.run_async([dense_retriever], llm_generator=reader_llm_generator)
            elif not args.use_upperbound and not args.use_dense_only:
                tog_retriever = TogV3Retriever(
                    llm_generator=reader_llm_generator,
                    sentence_encoder=sentence_encoder,
                    data=data,
                    inference_config=inference_config,
                    )
                graph_retriever = SimpleGraphRetriever(
                    llm_generator=reader_llm_generator,
                    sentence_encoder=sentence_encoder,
                    data=data,
                )
                
                subgraph_retriever = SubgraphRetriever(
                    llm_generator=reader_llm_generator,
                    sentence_encoder=sentence_encoder,
                    data=data,
                )
                # Start benchmarking (retrievers 顺序，query 并发)
                benchmark = RAGBenchmark(config=benchmark_config, logger=logger)
                await benchmark.run_async([tog_retriever, subgraph_retriever],
                                          llm_generator=reader_llm_generator, max_concurrency=32)

if __name__ == "__main__":
    asyncio.run(main())