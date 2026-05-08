"""
SFT Data Preparation Script for AutoGraph-R1

This script prepares supervised fine-tuning data by:
1. Loading dataset from HuggingFace (gzone0111/musique_hotpotqa_graph_retriever)
2. Extracting knowledge graph triples using atlas/Qwen3-235B-A22B-Instruct-2507
3. Using SubgraphRetriever with deducible judge to evaluate triple quality
4. Filtering and saving only high-quality triples for SFT

This addresses reviewer's question (a): 
"What's the performance of SFT using the reward function (e.g. deducible Judge) 
to select positive data and train the model."
"""

import os
import json
import json_repair
import asyncio
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import networkx as nx
from networkx import DiGraph

from openai import AsyncOpenAI
from autograph.rag_server.subgraph_retriever import SubgraphRetriever
from autograph.rag_server.llm_api import LLMGenerator
from autograph.rag_server.reranker_api import Reranker
from autograph.rag_server.base_retriever import RetrieverConfig


class SFTDataPreparation:
    """Prepare SFT data using deducible judge reward function"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
        base_url: str = "http://0.0.0.0:8129/v1",
        api_key: str = "***",
        num_hop: int = 3,
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ):
        """Initialize the SFT data preparation pipeline"""
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.num_hop = num_hop
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.emb_client = AsyncOpenAI(base_url="http://0.0.0.0:8128/v1", api_key="sk-embedding-key")
        # Initialize LLM generator and reranker
        self.llm_generator = LLMGenerator(
            client=self.client,
            model_name=model_name,
            backend="openai"
        )
        self.reranker = Reranker(
            emb_client=self.emb_client,
            model_name="Qwen/Qwen3-Embedding-0.6B"
        )
        
        # Initialize retriever config
        self.retriever_config = RetrieverConfig(
            name="subgraph",
            num_hop=num_hop,
            use_full_kg=False,
            temperature_reasoning=temperature
        )
        
        # Initialize subgraph retriever
        self.retriever = SubgraphRetriever(
            config=self.retriever_config,
            llm_generator=self.llm_generator,
            reranker=self.reranker
        )
    
    async def extract_triples_from_text(
        self, 
        text: str, 
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Extract knowledge graph triples from text using the LLM
        
        Args:
            text: The input text to extract triples from
            system_prompt: Optional custom system prompt
            
        Returns:
            List of triples as dicts with 'subject', 'relation', 'object'
        """
        if system_prompt is None:
            system_prompt = """You are an expert knowledge graph constructor.  
Your task is to extract factual information from the provided text and represent it strictly as a JSON array of knowledge graph triples.  

### Output Format
- The output must be a **JSON array**.
- Each element in the array must be a **JSON object** with exactly three non-empty keys:
  - "subject": the main entity, concept, event, or attribute.  
  - "relation": a concise, descriptive phrase or verb that describes the relationship (e.g., "founded by", "started on", "is a", "has circulation of").  
  - "object": the entity, concept, value, event, or attribute that the subject has a relationship with.  

### Constraints
- **Do not include any text other than the JSON output.**
- Do not add explanations, comments, or formatting outside of the JSON array.  
- Extract **all possible and relevant triples**.  
- All keys must exist and all values must be non-empty strings.  
- The "subject" and "object" can be specific entities (e.g., "Radio City", "Football in Albania", "Echosmith") or specific values (e.g., "3 July 2001", "1,310,696").  
- If no triples can be extracted, return exactly: `[]`."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            result = response.choices[0].message.content.strip()
            # Check if result is empty or just brackets
            if not result or result == "[]":
                print(f"WARNING: LLM returned empty result for text: {text[:100]}...")
                return []
            
            # Parse JSON response
            triples = json_repair.loads(result)
            # Validate triple structure
            if not isinstance(triples, list):
                return []
            
            valid_triples = []
            for triple in triples:
                if (isinstance(triple, dict) and 
                    "subject" in triple and 
                    "relation" in triple and 
                    "object" in triple and
                    triple["subject"] and 
                    triple["relation"] and 
                    triple["object"]):
                    valid_triples.append(triple)
            
            return valid_triples
            
        except Exception as e:
            print(f"Error extracting triples: {e}")
            return []
    
    def triples_to_graph(self, triples: List[Dict[str, str]]) -> DiGraph:
        """Convert list of triples to NetworkX DiGraph"""
        kg = DiGraph()
        for triple in triples:
            kg.add_edge(
                triple["subject"],
                triple["object"],
                relation=triple["relation"]
            )
        return kg
    
    async def evaluate_triples_deducibility(
        self,
        question: str,
        answer: str,
        triples: List[Dict[str, str]],
        sub_queries: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate whether extracted triples can deduce the answer
        using the deducible judge reward function
        
        Args:
            question: The question to answer
            answer: Ground truth answer
            triples: List of extracted KG triples
            sub_queries: Optional list of sub-queries for semantic reward
            
        Returns:
            Dict with evaluation results including:
            - deducible: 'yes' or 'no'
            - edge_coverage: fraction of edges used
            - semantic_reward: semantic fidelity score
        """
        if not triples:
            return {
                "deducible": "no",
                "edge_coverage": 0.0,
                "semantic_reward": 0.0,
                "subgraph_size": 0
            }
        
        # Convert triples to graph
        kg = self.triples_to_graph(triples)
        
        # Prepare sampling params
        sampling_params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        # Run retriever with deduce_reward
        result_json = await self.retriever.retrieve(
            question=question,
            kg=kg,
            sampling_params=sampling_params,
            sub_queries=[],
            answer=answer,
            reward_function="deduce_reward"
        )
        result = json_repair.loads(result_json)
        
        return {
            "deducible": result.get("answer", "no"),
            "edge_coverage": result.get("edge_coverage", 0.0),
            "semantic_reward": result.get("semantic_reward", 0.0),
            "subgraph_size": len(kg.edges)
        }
    
    async def process_single_example(
        self, 
        example: Dict[str, Any],
        save_all_data: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single example from the dataset
        
        Args:
            example: Single data example
            save_all_data: If True, save all examples. If False, only save deducible ones.
            
        Returns:
            Processed example with evaluation results, or None if filtered out
        """
        try:
            # Extract relevant information
            extra_info = example.get("extra_info", {})
            extra_info = extra_info.get("interaction_kwargs")
            question = extra_info.get("question", "")
            answer_list = extra_info.get("ground_truth", [])
            answer = answer_list[0] if answer_list else ""
            
            # Get supporting context
            full_context = extra_info.get("full_context", [])
            if hasattr(full_context, 'tolist'):
                full_context = full_context.tolist()
            elif not isinstance(full_context, list):
                full_context = list(full_context) if full_context else []
            context_text = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(full_context)])
            prompt_text = f"Extract from the documents:\n\n{context_text}"
            
            
            # Get sub-queries if available
            sub_queries = extra_info.get("sub_queries", [])
            triples = await self.extract_triples_from_text(prompt_text)
            # Extract triples from context
            # Use the same format as in the original dataset prompt
            if not triples:
                return None
            # Evaluate deducibility
            eval_result = await self.evaluate_triples_deducibility(
                question=question,
                answer=answer,
                triples=triples,
                sub_queries=sub_queries
            )
            # Prepare output data
            output_data = {
                "question": question,
                "answer": answer,
                "full_context": full_context,
                "extracted_triples": triples,
                "deducible": eval_result["deducible"],
                "edge_coverage": eval_result["edge_coverage"],
                "semantic_reward": eval_result["semantic_reward"],
                "subgraph_size": eval_result["subgraph_size"],
                "sub_queries": sub_queries,
                "data_source": example.get("data_source", ""),
                "ability": example.get("ability", "")
            }
            
            # Filter based on deducibility if not saving all
            if not save_all_data and eval_result["deducible"] != "yes":
                return None
            
            return output_data
            
        except Exception as e:
            print(f"Error processing example: {e}")
            return None
    
    async def process_dataset(
        self,
        input_path: str,
        output_path: str,
        max_examples: Optional[int] = None,
        save_all_data: bool = False,
        batch_size: int = 10
    ):
        """
        Process entire dataset and save filtered results
        
        Args:
            input_path: Path to input parquet file
            output_path: Path to save output parquet file
            max_examples: Maximum number of examples to process (None for all)
            save_all_data: If True, save all examples. If False, only deducible ones.
            batch_size: Number of examples to process concurrently
        """
        print(f"Loading dataset from {input_path}...")
        df = pd.read_parquet(input_path)
        # shuffle df
        df = df.sample(frac=1).reset_index(drop=True)
        if max_examples:
            df = df.head(max_examples)
        
        print(f"Processing {len(df)} examples...")
        
        processed_data = []
        total_deducible = 0
        total_processed = 0
        
        # Process in batches
        for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
            batch = df.iloc[i:i+batch_size]
            
            # Process batch concurrently
            tasks = [
                self.process_single_example(row.to_dict(), save_all_data)
                for _, row in batch.iterrows()
            ]
            results = await asyncio.gather(*tasks)
            # Collect valid results
            for result in results:
                if result is not None:
                    processed_data.append(result)
                    total_processed += 1
                    if result["deducible"] == "yes":
                        total_deducible += 1
        
        # Save results
        print(f"\nProcessed {total_processed} examples")
        print(f"Deducible examples: {total_deducible} ({100*total_deducible/total_processed:.2f}%)")
        
        if processed_data:
            output_df = pd.DataFrame(processed_data)
            output_df.to_parquet(output_path, index=False)
            print(f"Saved {len(output_df)} examples to {output_path}")
            
            # Save statistics
            stats = {
                "total_processed": total_processed,
                "total_deducible": total_deducible,
                "deducible_rate": total_deducible / total_processed if total_processed > 0 else 0,
                "avg_edge_coverage": output_df["edge_coverage"].mean(),
                "avg_subgraph_size": output_df["subgraph_size"].mean(),
            }
            
            stats_path = output_path.replace(".parquet", "_stats.json")
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"Saved statistics to {stats_path}")
        else:
            print("No valid examples to save!")


async def main():
    parser = argparse.ArgumentParser(
        description="Prepare SFT data using deducible judge reward function"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="./data/raw_dataset/train.parquet",
        help="Path to input parquet file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./data/sft_dataset/train_deducible.parquet",
        help="Path to save output parquet file"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of examples to process"
    )
    parser.add_argument(
        "--save_all",
        action="store_true",
        help="Save all examples (not just deducible ones)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Batch size for concurrent processing"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
        help="Model name for annotation"
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://0.0.0.0:4000",
        help="Base URL for API"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="dummy_api_key",
        help="API key"
    )
    parser.add_argument(
        "--num_hop",
        type=int,
        default=3,
        help="Number of hops for subgraph retrieval"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize pipeline
    pipeline = SFTDataPreparation(
        model_name=args.model_name,
        base_url=args.base_url,
        api_key=args.api_key,
        num_hop=args.num_hop
    )
    
    # Process dataset
    await pipeline.process_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        max_examples=args.max_examples,
        save_all_data=args.save_all,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    asyncio.run(main())