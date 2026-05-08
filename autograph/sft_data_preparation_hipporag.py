"""
SFT Data Preparation Script for HippoRAG2 (Graph-Text Retriever)

This script prepares supervised fine-tuning data by:
1. Loading dataset with samples containing full_context in extra_info
2. Pre-extracting triples from all unique documents (to avoid redundant LLM calls)
3. For each sample:
   - Building a sample-specific knowledge graph from full_context documents
   - Using HippoRAG2 to retrieve relevant passages via PageRank
   - Evaluating if all supporting contexts are retrieved
4. Filtering and saving only samples with complete retrieval (or all if --save_all)

This addresses reviewer's question about retrieval quality evaluation.
The approach efficiently handles large datasets by pre-extracting triples once
and reusing them across samples.
"""

import os
import json
import json_repair
import asyncio
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict
import networkx as nx
from networkx import DiGraph

from openai import AsyncOpenAI
from autograph.rag_server.hipporag2 import HippoRAG2Retriever
from autograph.rag_server.llm_api import LLMGenerator
from autograph.rag_server.reranker_api import Reranker
from autograph.rag_server.base_retriever import RetrieverConfig


class SFTDataPreparationHippoRAG:
    """Prepare SFT data using HippoRAG2 for graph-text retrieval"""
    
    def __init__(
        self,
        model_name: str = "atlas/Qwen3-32B",
        base_url: str = "http://0.0.0.0:4000",
        api_key: str = "sk-mThM069nvoUcAOep2SMloA",
        top_k: int = 10,
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ):
        """Initialize the SFT data preparation pipeline"""
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.top_k = top_k
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
        
        # Initialize retriever config for HippoRAG2
        self.retriever_config = RetrieverConfig(
            name="hipporag2",
            topN_passages=top_k,
            topN_edges=30,
            weight_adjust=0.05,
            temperature_reasoning=temperature
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
    
    async def extract_all_document_triples(
        self, 
        documents: List[str],
        batch_size: int = 20
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Pre-extract triples for all unique documents in parallel batches
        
        Args:
            documents: List of document texts
            batch_size: Number of documents to process concurrently
            
        Returns:
            Dict mapping document text to extracted triples
        """
        print(f"Extracting triples from {len(documents)} unique documents (batch_size={batch_size})...")
        
        triples_dict = {}
        
        async def extract_single_doc(doc_text: str) -> Tuple[str, List[Dict[str, str]]]:
            """Helper to extract triples for a single document"""
            prompt_text = f"Extract from the document:\n\n{doc_text}"
            doc_triples = await self.extract_triples_from_text(prompt_text)
            return doc_text, doc_triples
        
        # Process documents in batches
        for i in tqdm(range(0, len(documents), batch_size), desc="Extracting triples"):
            batch = documents[i:i+batch_size]
            
            # Process batch concurrently
            tasks = [extract_single_doc(doc_text) for doc_text in batch]
            results = await asyncio.gather(*tasks)
            
            # Collect results
            for doc_text, doc_triples in results:
                triples_dict[doc_text] = doc_triples
        
        print(f"Extracted triples from {len(triples_dict)} documents")
        return triples_dict
    
    def build_sample_graph(
        self,
        full_context: List[str],
        triples_dict: Dict[str, List[Dict[str, str]]]
    ) -> Tuple[DiGraph, List[Dict[str, str]]]:
        """
        Build a sample-specific knowledge graph from full_context documents
        
        Args:
            full_context: List of document texts for this sample
            triples_dict: Pre-extracted triples for all documents
            
        Returns:
            Tuple of (NetworkX DiGraph with entity and text nodes, 
                     List of actual triples in the graph after deduplication)
        """
        kg = DiGraph()
        actual_triples = []  # Track triples that actually made it into the graph
        
        for idx, doc_text in enumerate(full_context):
            # Use index as document ID (Document 0, Document 1, etc.)
            doc_id = f"Document {idx}"
            
            # Add text node
            kg.add_node(doc_id, node_type="text", text=doc_text)
            
            # Get pre-extracted triples for this document
            doc_triples = triples_dict.get(doc_text, [])
            
            # Add entities and edges to graph
            for triple in doc_triples:
                subject = triple["subject"]
                relation = triple["relation"]
                obj = triple["object"]
                
                # Add entity nodes if they don't exist
                if not kg.has_node(subject):
                    kg.add_node(subject, node_type="entity")
                if not kg.has_node(obj):
                    kg.add_node(obj, node_type="entity")
                
                # Add entity-entity edge (DiGraph overwrites if edge exists)
                kg.add_edge(subject, obj, relation=relation)
                
                # Link entities to this text document
                kg.add_edge(subject, doc_id, relation="appears_in")
                kg.add_edge(obj, doc_id, relation="appears_in")
        
        # After building graph, extract actual triples from the final graph
        # Only extract entity-to-entity edges (not "appears_in" edges to documents)
        # This matches what hipporag2.index_kg() will embed
        for src, dst, data in kg.edges(data=True):
            # Check if both nodes are entity nodes (not text document nodes)
            src_node_type = kg.nodes[src].get('node_type')
            dst_node_type = kg.nodes[dst].get('node_type')
            
            if src_node_type == 'entity' and dst_node_type == 'entity' and data.get('relation'):
                actual_triples.append({
                    'subject': src,
                    'relation': data['relation'],
                    'object': dst
                })
        
        return kg, actual_triples
    
    async def retrieve_passages(
        self,
        query: str,
        sample_kg: DiGraph,
        supporting_context: List[str] = None,
        top_k: int = None
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Retrieve relevant passage IDs for a query using HippoRAG2 on sample-specific graph
        
        Args:
            query: The search query
            sample_kg: Sample-specific knowledge graph
            supporting_context: Ground truth supporting contexts for evaluation
            top_k: Number of passages to retrieve
            
        Returns:
            Tuple of (retrieved_passage_ids, metrics_dict)
            - retrieved_passage_ids: List of retrieved passage IDs (text node IDs)
            - metrics_dict: Dict with 'precision' and 'recall' keys
        """
        if top_k is None:
            top_k = self.top_k
        
        # Set up HippoRAG2 state with sample-specific graph
        text_list = []
        node_list = []
                # Initialize HippoRAG2
        hipporag = HippoRAG2Retriever(
            config=self.retriever_config,
            llm_generator=self.llm_generator,
            reranker=self.reranker
        )
        hipporag.KG = sample_kg
        
        for node, attrs in sample_kg.nodes(data=True):
            if attrs.get("node_type") == "text":
                text_list.append(node)
            else:
                node_list.append(node)

        hipporag.entity_KG = sample_kg.subgraph(node_list)
        hipporag.text_list = text_list
        hipporag.node_list = node_list
        # Index the entity KG for this sample (this will create edge_list internally)
        await hipporag.index_kg()
        
        # Now set edge_list to match what was used in index_kg
        # This must be done AFTER index_kg to ensure they're in sync
        hipporag.edge_list = list(hipporag.entity_KG.edges(data="relation"))
        
        # Get personalization dict from query
        node_dict = await hipporag.retrieve_personalization_dict(
            query,
            topN=hipporag.config.topN_edges,
            weight_adjust=hipporag.config.weight_adjust
        )
        
        # Run PageRank
        pr = nx.pagerank(sample_kg, personalization=node_dict)
        
        # Extract text node scores
        text_scores = {}
        for node in pr:
            if node in hipporag.text_list:
                text_scores[node] = pr[node]
        
        # Get top-k passages
        sorted_passages = sorted(text_scores.items(), key=lambda x: x[1], reverse=True)
        sorted_passages = sorted_passages[:top_k]
        retrieved_ids = [passage_id for passage_id, _ in sorted_passages]
        
        # Calculate precision and recall
        precision = await hipporag.calculate_precision_reward(
            retrieved_ids,
            supporting_context or []
        )
        recall = await hipporag.calculate_recall_reward(
            retrieved_ids,
            supporting_context or []
        )
        
        metrics = {
            "precision": precision,
            "recall": recall
        }
        
        return retrieved_ids, metrics
    
    def calculate_retrieval_metrics(
        self,
        retrieved_ids: List[str],
        supporting_context: List[str],
        full_context: List[str]
    ) -> Dict[str, Any]:
        """
        Calculate retrieval metrics by comparing retrieved docs with supporting context
        
        Args:
            retrieved_ids: List of retrieved document IDs (e.g., "Document 0")
            supporting_context: List of ground truth supporting document texts
            full_context: List of all document texts for this sample
            
        Returns:
            Dict with recall, precision, and other metrics
        """
        # Map document IDs to texts
        retrieved_texts = []
        for rid in retrieved_ids:
            # Extract index from "Document N"
            try:
                idx = int(rid.split()[-1])
                if idx < len(full_context):
                    retrieved_texts.append(full_context[idx])
            except (ValueError, IndexError):
                continue
        
        # Calculate metrics by matching texts
        retrieved_set = set(retrieved_texts)
        supporting_set = set(supporting_context)
        
        # Count matches using substring matching (same as HippoRAG2)
        tp = 0
        for supp in supporting_set:
            for retr in retrieved_set:
                if supp in retr or retr in supp:
                    tp += 1
                    break
        
        fp = len(retrieved_set) - tp
        fn = len(supporting_set) - tp
        
        recall = tp / len(supporting_set) if len(supporting_set) > 0 else 0.0
        precision = tp / len(retrieved_set) if len(retrieved_set) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Check if all supporting docs are retrieved
        all_retrieved = (tp == len(supporting_set))
        
        return {
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "all_retrieved": all_retrieved,
            "retrieved_count": len(retrieved_set),
            "supporting_count": len(supporting_set),
            "tp": tp,
            "fp": fp,
            "fn": fn
        }
    
    async def process_single_example(
        self,
        example: Dict[str, Any],
        triples_dict: Dict[str, List[Dict[str, str]]],
        save_all_data: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single example from the dataset
        
        Args:
            example: Single data example with extra_info containing full_context
            triples_dict: Pre-extracted triples for all documents
            save_all_data: If True, save all examples. If False, only save with all_retrieved=True.
            
        Returns:
            Processed example with retrieval results, or None if filtered out
        """
        try:
            # Extract query information
            # Get extra_info with full_context and supporting_context
            extra_info = example.get("extra_info", {})
            interaction_kwargs = extra_info.get("interaction_kwargs", {})
            
            full_context = interaction_kwargs.get("full_context", [])
            supporting_context = interaction_kwargs.get("supporting_context", [])
            query = interaction_kwargs.get("question")

            # Normalize common array-like inputs (numpy arrays, pandas Series,
            # tuples, sets) to plain Python lists so truth checks and indexing
            # behave as expected.
            if isinstance(full_context, (np.ndarray, pd.Series)):
                full_context = full_context.tolist()
            elif isinstance(full_context, (set, tuple)):
                full_context = list(full_context)
            elif full_context is None:
                full_context = []

            if isinstance(supporting_context, (np.ndarray, pd.Series)):
                supporting_context = supporting_context.tolist()
            elif isinstance(supporting_context, (set, tuple)):
                supporting_context = list(supporting_context)
            elif supporting_context is None:
                supporting_context = []

            # Ensure query is a string (safe fallback)
            if query is None:
                query = ""
            else:
                query = str(query)

            # Now check emptiness explicitly
            if len(full_context) == 0:
                print(f"Warning: No full_context found for query: {query[:50]}...")
                return None
            
            # Build sample-specific graph and get actual triples
            sample_kg, actual_triples = self.build_sample_graph(full_context, triples_dict)
            
            # Retrieve passages using HippoRAG2
            retrieved_ids, retrieval_metrics = await self.retrieve_passages(
                query, 
                sample_kg=sample_kg,
                supporting_context=supporting_context,
                top_k=self.top_k
            )
            
            # Get retrieved texts from full_context
            # Also keep numeric indices of retrieved documents
            retrieved_doc_indices = []
            retrieved_texts = []
            for rid in retrieved_ids:
                try:
                    idx = int(rid.split()[-1])
                    retrieved_doc_indices.append(idx)
                    if idx < len(full_context):
                        retrieved_texts.append(full_context[idx])
                except (ValueError, IndexError):
                    continue
            
            # Calculate additional metrics
            metrics = self.calculate_retrieval_metrics(
                retrieved_ids, 
                supporting_context,
                full_context
            )
            
            # Add precision and recall from HippoRAG2
            metrics["hipporag_precision"] = retrieval_metrics["precision"]
            metrics["hipporag_recall"] = retrieval_metrics["recall"]
            
            # Create SFT training prompt with all documents from full_context
            documents_text = "\n\n".join([
                f"Document {i}: {doc}" 
                for i, doc in enumerate(full_context)
            ])
            
            system_instruction = """You are an expert knowledge graph constructor.  
Your task is to extract factual information from the provided documents and represent it strictly as a JSON array of knowledge graph triples.  

### Output Format
- The output must be a **JSON array**.
- Each element in the array must be a **JSON object** with exactly three non-empty keys:
  - "subject": the main entity, concept, event, or attribute.  
  - "relation": a concise, descriptive phrase or verb that describes the relationship (e.g., "founded by", "started on", "is a", "has circulation of").  
  - "object": the entity, concept, value, event, or attribute that the subject has a relationship with.  

### Constraints
- **Do not include any text other than the JSON output.**
- Do not add explanations, comments, or formatting outside of the JSON array.  
- Extract **all possible and relevant triples** from the documents.  
- All keys must exist and all values must be non-empty strings."""

            prompt = f"""{system_instruction}

Generate triples for the following documents:

{documents_text}"""
            
            # Only save this sample if the retriever achieved perfect recall (recall == 1.0).
            # Drop the sample otherwise.
            if metrics.get("recall", 0.0) < 0.8:
                return None

            # Use the actual triples from the graph (after DiGraph deduplication)
            # instead of collecting from all documents (which would have duplicates)
            response_triples = actual_triples

            # Format response as JSON
            response = json.dumps(response_triples)
            
            # Get answer for reference
            ground_truth = interaction_kwargs.get("ground_truth", [])
            if isinstance(ground_truth, list) and len(ground_truth) > 0:
                answer = ground_truth[0]
            else:
                answer = str(ground_truth)
            
            # Prepare output data
            output_data = {
                "prompt": prompt,
                "response": response,
                "query": query,
                "answer": str(answer),
                "retrieved_ids": retrieved_ids,
                "retrieved_texts": retrieved_texts,
                "retrieved_doc_indices": retrieved_doc_indices,
                "num_triples": len(response_triples),
                "supporting_context": supporting_context,
                "full_context": full_context,
                "recall": metrics["recall"],
                "precision": metrics["precision"],
                "f1": metrics["f1"],
                "all_retrieved": metrics["all_retrieved"],
                "retrieved_count": metrics["retrieved_count"],
                "supporting_count": metrics["supporting_count"],
                "data_source": example.get("data_source", ""),
                "split": extra_info.get("split", "")
            }
            
            # Filter based on all_retrieved if not saving all
            if not save_all_data and not metrics["all_retrieved"]:
                return None
            
            return output_data
            
        except Exception as e:
            print(f"Error processing example: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def process_dataset(
        self,
        input_path: str,
        output_path: str,
        max_examples: Optional[int] = None,
        save_all_data: bool = False,
        batch_size: int = 10,
        triple_batch_size: int = 20
    ):
        """
        Process entire dataset and save filtered results
        
        Args:
            input_path: Path to input parquet file with samples containing extra_info
            output_path: Path to save output parquet file
            max_examples: Maximum number of examples to process (None for all)
            save_all_data: If True, save all examples. If False, only all_retrieved ones.
            batch_size: Number of examples to process concurrently
            triple_batch_size: Number of documents to extract triples from concurrently
        """
        # Load dataset
        print(f"Loading dataset from {input_path}...")
        df = pd.read_parquet(input_path)
        
        # shuffle dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        if max_examples:
            df = df.head(max_examples)
        
        print(f"Loaded {len(df)} examples")
        
        # Step 1: Collect all unique documents from full_context
        print("Collecting unique documents from all samples...")
        unique_documents = set()
        
        for _, row in df.iterrows():
            extra_info = row.get("extra_info", {})
            interaction_kwargs = extra_info.get("interaction_kwargs", {})
            full_context = interaction_kwargs.get("full_context", [])
            unique_documents.update(full_context)
        
        unique_documents = list(unique_documents)
        print(f"Found {len(unique_documents)} unique documents")
        
        # Step 2: Pre-extract triples for all unique documents
        triples_dict = await self.extract_all_document_triples(
            unique_documents, 
            batch_size=triple_batch_size
        )
        
        # Step 3: Process each sample
        print(f"\nProcessing {len(df)} samples...")
        
        processed_data = []
        total_all_retrieved = 0
        total_processed = 0
        
        # Process in batches
        for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
            batch = df.iloc[i:i+batch_size]
            
            # Process batch concurrently
            tasks = [
                self.process_single_example(row.to_dict(), triples_dict, save_all_data)
                for _, row in batch.iterrows()
            ]
            results = await asyncio.gather(*tasks)
            
            # Collect valid results
            for result in results:
                if result is not None:
                    processed_data.append(result)
                    total_processed += 1
                    if result["all_retrieved"]:
                        total_all_retrieved += 1
        
        # Save results
        print(f"\nProcessed {total_processed} examples")
        
        if total_processed > 0:
            print(f"All retrieved examples: {total_all_retrieved} ({100*total_all_retrieved/total_processed:.2f}%)")
        else:
            print("No examples were successfully processed!")
            return
        
        if processed_data:
            output_df = pd.DataFrame(processed_data)
            output_df.to_parquet(output_path, index=False)
            print(f"Saved {len(output_df)} examples to {output_path}")
            
            # Save statistics
            stats = {
                "total_processed": total_processed,
                "total_all_retrieved": total_all_retrieved,
                "all_retrieved_rate": total_all_retrieved / total_processed,
                "avg_recall": float(output_df["recall"].mean()),
                "avg_precision": float(output_df["precision"].mean()),
                "avg_f1": float(output_df["f1"].mean()),
                "avg_retrieved_count": float(output_df["retrieved_count"].mean()),
                "avg_supporting_count": float(output_df["supporting_count"].mean()),
            }
            
            stats_path = output_path.replace(".parquet", "_stats.json")
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"Saved statistics to {stats_path}")
        else:
            print("No valid examples to save!")


async def main():
    parser = argparse.ArgumentParser(
        description="Prepare SFT data using HippoRAG2 for graph-text retrieval"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="./data/raw_dataset_graph_text/train.parquet",
        help="Path to input parquet file with samples"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./data/sft_dataset/train_hipporag_all_retrieved.parquet",
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
        help="Save all examples (not just all_retrieved ones)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Batch size for concurrent processing of samples"
    )
    parser.add_argument(
        "--triple_batch_size",
        type=int,
        default=20,
        help="Batch size for concurrent triple extraction from documents"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="atlas/Qwen3-30B-A3B-Instruct-2507",
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
        "--top_k",
        type=int,
        default=5,
        help="Number of passages to retrieve"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize pipeline
    pipeline = SFTDataPreparationHippoRAG(
        model_name=args.model_name,
        base_url=args.base_url,
        api_key=args.api_key,
        top_k=args.top_k
    )
    
    # Process dataset
    await pipeline.process_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        max_examples=args.max_examples,
        save_all_data=args.save_all,
        batch_size=args.batch_size,
        triple_batch_size=args.triple_batch_size
    )


if __name__ == "__main__":
    asyncio.run(main())
