from neo4j import GraphDatabase
import faiss
import numpy as np
import random
from collections import defaultdict
from typing import List
import time
import logging
from atlas_rag.llm_generator.llm_generator import LLMGenerator
from atlas_rag.vectorstore.embedding_model import BaseEmbeddingModel
from atlas_rag.retriever.lkg_retriever.base import BaseLargeKGEdgeRetriever
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

class LargeKGToGRetriever(BaseLargeKGEdgeRetriever):
    def __init__(self, keyword: str, neo4j_driver: GraphDatabase, 
                 llm_generator: LLMGenerator, sentence_encoder: BaseEmbeddingModel, filter_encoder: BaseEmbeddingModel,
                 node_index: faiss.Index, 
                 topN : int = 5,
                 Dmax : int = 3,
                 Wmax : int = 3,
                 prune_size: int = 10,
                 logger: logging.Logger = None):
        """
        Initialize the LargeKGToGRetriever for billion-level KG retrieval using Neo4j.

        Args:
            keyword (str): Identifier for the KG dataset (e.g., 'cc_en').
            neo4j_driver (GraphDatabase): Neo4j driver for database access.
            llm_generator (LLMGenerator): LLM for NER, rating, and reasoning.
            sentence_encoder (BaseEmbeddingModel): Encoder for generating embeddings.
            node_index (faiss.Index): FAISS index for node embeddings.
            logger (Logger, optional): Logger for verbose output.
        """
        self.keyword = keyword
        self.neo4j_driver = neo4j_driver
        self.llm_generator = llm_generator
        self.sentence_encoder = sentence_encoder
        self.filter_encoder = filter_encoder
        self.node_faiss_index = node_index
        self.verbose = logger is not None
        self.logger = logger 
        self.topN = topN
        self.Dmax = Dmax
        self.Wmax = Wmax
        self.prune_size = prune_size

    def ner(self, text: str) -> List[str]:
        """
        Extract named entities from the query text using the LLM.

        Args:
            text (str): The query text.

        Returns:
            List[str]: List of extracted entities.
        """
        entities = self.llm_generator.large_kg_tog_ner(text)
        if self.verbose:
            self.logger.info(f"Extracted entities: {entities}")
        return entities

    def retrieve_topk_nodes(self, query: str, top_k_nodes: int = 5) -> List[str]:
        """
        Retrieve top-k nodes similar to entities in the query.

        Args:
            query (str): The user query.
            top_k_nodes (int): Number of nodes to retrieve per entity.

        Returns:
            List[str]: List of node numeric_ids.
        """
        start_time = time.time()
        entities = self.ner(query)
        if self.verbose:
            ner_time = time.time() - start_time
            self.logger.info(f"NER took {ner_time:.2f} seconds, entities: {entities}")
        if not entities:
            entities = [query]

        initial_nodes = []
        for entity in entities:
            entity_embedding = self.sentence_encoder.encode([entity])
            D, I = self.node_faiss_index.search(entity_embedding, 3)
            if self.verbose:
                self.logger.info(f"Entity: {entity}, FAISS Distances: {D}, Indices: {I}")
            if len(I[0]) > 0:  # Check if results exist
                initial_nodes.extend([str(i) for i in I[0]])
        # no need filtering as ToG pruning will handle it.
        topk_nodes_ids = list(set(initial_nodes))

        with self.neo4j_driver.session() as session:
            start_time = time.time()
            query = """
            MATCH (n:Node)
            WHERE n.numeric_id IN $topk_nodes_ids
            RETURN n.numeric_id AS id, n.name AS name
            """
            result = session.run(query, topk_nodes_ids=topk_nodes_ids)
            topk_nodes_dict = {}
            for record in result:
                topk_nodes_dict[record["id"]] = record["name"]
            if self.verbose:
                neo4j_time = time.time() - start_time
                self.logger.info(f"Neo4j query took {neo4j_time:.2f} seconds, count: {len(topk_nodes_ids)}")
        
        if self.verbose:
            self.logger.info(f"Top-k nodes: {topk_nodes_dict}")
        return list(topk_nodes_dict.keys()), list(topk_nodes_dict.values()) # numeric_id of nodes returned

    def expand_paths(self, P: List[List[str]], PIDS: List[List[str]], PTYPES: List[List[str]], width: int, query: str) -> List[List[str]]:
        """
        Expand each path by adding neighbors of the last node.

        Args:
            P (List[List[str]]): Current list of paths, where each path is a list of alternating node_ids and relation types.

        Returns:
            List[List[str]]: List of expanded paths.
        """
        last_nodes = []
        last_node_ids = []
        last_node_types = []
        
        paths_end_with_text = []
        paths_end_with_text_id = []
        paths_end_with_text_type = []

        paths_end_with_node = []
        paths_end_with_node_id = []
        paths_end_with_node_type = []
        if self.verbose:
            self.logger.info(f"Expanding paths, current paths: {P}")
        for p, pid, ptype in zip(P, PIDS, PTYPES):
            if not p or not pid or not ptype:  # Skip empty paths
                continue
            t = ptype[-1]
            if t == "Text":
                paths_end_with_text.append(p)
                paths_end_with_text_id.append(pid)
                paths_end_with_text_type.append(ptype)
                continue
            last_node = p[-1]  # Last node in the path
            last_node_id = pid[-1]  # Last node numeric_id

            last_nodes.append(last_node) 
            last_node_ids.append(last_node_id)
            last_node_types.append(t)

            paths_end_with_node.append(p)
            paths_end_with_node_id.append(pid)
            paths_end_with_node_type.append(ptype)
        
        assert len(last_nodes) == len(last_node_ids) == len(last_node_types), "Mismatch in last nodes, ids, and types lengths"
        
        if not last_node_ids:
            return paths_end_with_text, paths_end_with_text_id, paths_end_with_text_type

        with self.neo4j_driver.session() as session:
            # Query Node relationships
            start_time = time.time()
            outgoing_query = """
            CALL apoc.cypher.runTimeboxed(
            "MATCH (n:Node)-[r:Relation]-(m:Node) WHERE n.numeric_id IN $last_node_ids 
            WITH n, r, m ORDER BY rand() LIMIT 60000
            RETURN n.numeric_id AS source, n.name AS source_name, r.relation AS rel_type, m.numeric_id AS target, m.name AS target_name, 'Node' AS target_type",
            {last_node_ids: $last_node_ids},
            60000
            )
            YIELD value
            RETURN value.source AS source, value.source_name AS source_name, value.rel_type AS rel_type, value.target AS target, value.target_name AS target_name, value.target_type AS target_type
            """
            outgoing_result = session.run(outgoing_query, last_node_ids=last_node_ids)
            outgoing = [(record["source"], record['source_name'], record["rel_type"], record["target"], record["target_name"], record["target_type"]) 
                        for record in outgoing_result]
            if self.verbose:
                outgoing_time = time.time() - start_time
                self.logger.info(f"Outgoing relationships query took {outgoing_time:.2f} seconds, count: {len(outgoing)}")

            # # Query outgoing Node -> Text relationships
            # start_time = time.time()
            # outgoing_text_query = """
            # MATCH (n:Node)-[r:Source]->(t:Text)
            # WHERE n.numeric_id IN $last_node_ids
            # RETURN n.numeric_id AS source, n.name AS source_name, 'from Source' AS rel_type, t.numeric_id as target, t.original_text AS target_name, 'Text' AS target_type
            # """
            # outgoing_text_result = session.run(outgoing_text_query, last_node_ids=last_node_ids)
            # outgoing_text = [(record["source"], record["source_name"], record["rel_type"], record["target"], record["target_name"], record["target_type"])
            #                 for record in outgoing_text_result]
            # if self.verbose:
            #     outgoing_text_time = time.time() - start_time
            #     self.logger.info(f"Outgoing Node->Text relationships query took {outgoing_text_time:.2f} seconds, count: {len(outgoing_text)}")

        last_node_to_new_paths = defaultdict(list)
        last_node_to_new_paths_ids = defaultdict(list)
        last_node_to_new_paths_types = defaultdict(list)
    

        for p, pid, ptype in zip(P, PIDS, PTYPES):
            last_node = p[-1]  
            last_node_id = pid[-1] 
            # Outgoing Node -> Node
            for source, source_name, rel_type, target, target_name, target_type in outgoing:
                if source == last_node_id and target_name not in p:
                    new_path = p + [rel_type, target_name]
                    if target_name.lower() in stopwords.words('english'):
                        continue
                    last_node_to_new_paths[last_node].append(new_path)
                    last_node_to_new_paths_ids[last_node].append(pid + [target])
                    last_node_to_new_paths_types[last_node].append(ptype + [target_type])

            # # Outgoing Node -> Text
            # for source, source_name, rel_type, target, target_name, target_type in outgoing_text:
            #     if source == last_node_id and target_name not in p:
            #         new_path = p + [rel_type, target_name]

            #         last_node_to_new_paths_text[last_node].append(new_path)
            #         last_node_to_new_paths_text_ids[last_node].append(pid + [target])
            #         last_node_to_new_paths_text_types[last_node].append(ptype + [target_type])

            # # Incoming Node -> Node
            # for source, rel_type, target, source_name, source_type in incoming:
            #     if target == last_node_id and source not in p:
            #         new_path = p + [rel_type, source_name]
                    

        num_paths = 0
        for last_node, new_paths in last_node_to_new_paths.items():
            num_paths += len(new_paths)
        # for last_node, new_paths in last_node_to_new_paths_text.items():
        #     num_paths += len(new_paths)
        new_paths = []
        new_pids = []
        new_ptypes = []
        if self.verbose:
            self.logger.info(f"Number of new paths before filtering: {num_paths}")
            self.logger.info(f"last nodes: {last_node_to_new_paths.keys()}")
        if num_paths > len(last_node_ids) * width:
            # Apply filtering when total paths exceed threshold
            for last_node, new_ps in last_node_to_new_paths.items():
                if len(new_ps) > width:
                    path_embeddings = self.filter_encoder.encode(new_ps)
                    query_embeddings = self.filter_encoder.encode([query])
                    scores = np.dot(path_embeddings, query_embeddings.T).flatten()
                    top_indices = np.argsort(scores)[-width:]
                    new_paths.extend([new_ps[i] for i in top_indices])
                    new_pids.extend([last_node_to_new_paths_ids[last_node][i] for i in top_indices])
                    new_ptypes.extend([last_node_to_new_paths_types[last_node][i] for i in top_indices])
                else:
                    new_paths.extend(new_ps)
                    new_pids.extend(last_node_to_new_paths_ids[last_node])
                    new_ptypes.extend(last_node_to_new_paths_types[last_node])
        else:
            # Collect all paths without filtering when total is at or below threshold
            for last_node, new_ps in last_node_to_new_paths.items():
                new_paths.extend(new_ps)
                new_pids.extend(last_node_to_new_paths_ids[last_node])
                new_ptypes.extend(last_node_to_new_paths_types[last_node])

        if self.verbose:
            self.logger.info(f"Expanded paths count: {len(new_paths)}")
            self.logger.info(f"Expanded paths: {new_paths}")
        return new_paths, new_pids, new_ptypes

    def path_to_string(self, path: List[str]) -> str:
        """
        Convert a path to a human-readable string for LLM rating.

        Args:
            path (List[str]): Path as a list of node_ids and relation types.

        Returns:
            str: String representation of the path.
        """
        if len(path) < 1:
            return ""
        path_str = []
        with self.neo4j_driver.session() as session:
            for i in range(0, len(path), 2):
                node_id = path[i]
                result = session.run("MATCH (n:Node {numeric_id: $node_id}) RETURN n.name", node_id=node_id)
                node_name = result.single()["n.name"] if result.single() else node_id
                if i + 1 < len(path):
                    rel_type = path[i + 1]
                    path_str.append(f"{node_name} ---> {rel_type} --->")
                else:
                    path_str.append(node_name)
        return " ".join(path_str).strip()

    def prune(self, query: str, P: List[List[str]], PIDS: List[List[str]], PTYPES: List[List[str]], topN: int = 5) -> List[List[str]]:
        """
        Prune paths to keep the top N based on LLM relevance ratings.

        Args:
            query (str): The user query.
            P (List[List[str]]): List of paths to prune.
            topN (int): Number of paths to retain.

        Returns:
            List[List[str]]: Top N paths.
        """
        ratings = []
        path_strings = P
        
        # Process paths in chunks of 10
        for i in range(0, len(path_strings), self.prune_size):
            chunk = path_strings[i:i + self.prune_size]
            
            # Construct user prompt with the current chunk of paths listed
            user_prompt = f"Please rate the following paths based on how well they help answer the query (1-5, 0 if not relevant).\n\nQuery: {query}\n\nPaths:\n"
            for j, path_str in enumerate(chunk, 1):
                user_prompt += f"{j + i}. {path_str}\n"
            user_prompt += "\nProvide a list of integers, each corresponding to the rating of the path's ability to help answer the query."

            # Define system prompt to expect a list of integers
            system_prompt = "You are a rating machine that only provides a list of comma-separated integers (0-5) as a response, each rating how well the corresponding path helps answer the query."
            
            # Send the prompt to the language model
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.llm_generator.generate_response(messages, max_new_tokens=1024, temperature=0.0)
            if self.verbose:
                self.logger.info(f"LLM response for chunk {i // self.prune_size + 1}: {response}")
            
            # Parse the response into a list of ratings
            rating_str = response.strip()
            chunk_ratings = [int(r) for r in rating_str.split(',') if r.strip().isdigit()]
            if len(chunk_ratings) > len(chunk):
                chunk_ratings = chunk_ratings[:len(chunk)]
                if self.verbose:
                    self.logger.warning(f"Received more ratings ({len(chunk_ratings)}) than paths in chunk ({len(chunk)}). Trimming ratings.")
            ratings.extend(chunk_ratings)  # Concatenate ratings
        
        # Ensure ratings length matches number of paths, padding with 0s if necessary
        if len(ratings) < len(path_strings):
            # self.logger.warning(f"Number of ratings ({len(ratings)}) does not match number of paths ({len(path_strings)}). Padding with 0s.")
            # ratings += [0] * (len(path_strings) - len(ratings))
            # fall back to use filter encoder to get topN
            self.logger.warning(f"Number of ratings ({len(ratings)}) does not match number of paths ({len(path_strings)}). Using filter encoder to get topN paths.")
            path_embeddings = self.filter_encoder.encode(path_strings)
            query_embedding = self.filter_encoder.encode([query])
            scores = np.dot(path_embeddings, query_embedding.T).flatten()
            top_indices = np.argsort(scores)[-topN:]
            top_paths = [path_strings[i] for i in top_indices]
            return top_paths, [PIDS[i] for i in top_indices], [PTYPES[i] for i in top_indices]
        elif len(ratings) > len(path_strings):
            self.logger.warning(f"Number of ratings ({len(ratings)}) exceeds number of paths ({len(path_strings)}). Trimming ratings.")
            ratings = ratings[:len(path_strings)]
        
        # Sort indices based on ratings in descending order
        sorted_indices = sorted(range(len(ratings)), key=lambda i: ratings[i], reverse=True)

        # Filter out indices where the rating is 0
        filtered_indices = [i for i in sorted_indices if ratings[i] > 0]

        # Take the top N indices from the filtered list
        top_indices = filtered_indices[:topN]

        # Use the filtered indices to get the top paths, PIDS, and PTYPES
        if self.verbose:
            self.logger.info(f"Top indices after pruning: {top_indices}")
            self.logger.info(f"length of path_strings: {len(path_strings)}")
        top_paths = [path_strings[i] for i in top_indices]
        top_pids = [PIDS[i] for i in top_indices]
        top_ptypes = [PTYPES[i] for i in top_indices]
        
        
        # Log top paths if verbose mode is enabled
        if self.verbose:
            self.logger.info(f"Pruned to top {topN} paths: {top_paths}")
        
        return top_paths, top_pids, top_ptypes

    def reasoning(self, query: str, P: List[List[str]]) -> bool:
        """
        Check if the current paths are sufficient to answer the query.

        Args:
            query (str): The user query.
            P (List[List[str]]): Current list of paths.

        Returns:
            bool: True if sufficient, False otherwise.
        """
        triples = []
        with self.neo4j_driver.session() as session:
            for path in P:
                if len(path) < 3:
                    continue
                for i in range(0, len(path) - 2, 2):
                    node1_name = path[i]
                    rel = path[i + 1]
                    node2_name = path[i + 2]
                    triples.append(f"({node1_name}, {rel}, {node2_name})")
        triples_str = ". ".join(triples)
        prompt = f"Are these triples, along with your knowledge, sufficient to answer the query?\nQuery: {query}\nTriples: {triples_str}"
        messages = [
            {"role": "system", "content": "Answer Yes or No only."},
            {"role": "user", "content": prompt}
        ]
        response = self.llm_generator.generate_response(messages,max_new_tokens=512)
        if self.verbose:
            self.logger.info(f"Reasoning result: {response}")
        return "yes" in response.lower()

    def retrieve_passages(self, query: str) -> List[str]:
        """
        Retrieve the top N paths to answer the query.

        Args:
            query (str): The user query.
            topN (int): Number of paths to return.
            Dmax (int): Maximum depth of path expansion.
            Wmax (int): Maximum width of path expansion.

        Returns:
            List[str]: List of triples as strings.
        """
        topN = self.topN
        Dmax = self.Dmax
        Wmax = self.Wmax
        if self.verbose:
            self.logger.info(f"Retrieving paths for query: {query}")
        
        initial_nodes_ids, initial_nodes = self.retrieve_topk_nodes(query, top_k_nodes=topN)
        if not initial_nodes:
            if self.verbose:
                self.logger.info("No initial nodes found.")
            return []

        P = [[node] for node in initial_nodes]
        PIDS = [[node_id] for node_id in initial_nodes_ids]
        PTYPES = [["Node"] for _ in initial_nodes_ids]  # Assuming all initial nodes are of type 'Node'
        for D in range(Dmax + 1):
            if self.verbose:
                self.logger.info(f"Depth {D}, Current paths: {len(P)}")
            P, PIDS, PTYPES = self.expand_paths(P, PIDS, PTYPES, Wmax, query)
            if not P:
                if self.verbose:
                    self.logger.info("No paths to expand.")
                break
            P, PIDS, PTYPES = self.prune(query, P, PIDS, PTYPES, topN)
            if D == Dmax:
                if self.verbose:
                    self.logger.info(f"Reached maximum depth {Dmax}, stopping expansion.")
                break
            if self.reasoning(query, P):
                if self.verbose:
                    self.logger.info("Paths sufficient, stopping expansion.")
                break

        # Extract final triples
        triples = []
        with self.neo4j_driver.session() as session:
            for path in P:
                for i in range(0, len(path) - 2, 2):
                    node1_name = path[i]
                    rel = path[i + 1]
                    node2_name = path[i + 2]
                    triples.append(f"({node1_name}, {rel}, {node2_name})")
        
        if self.verbose:
            self.logger.info(f"Final triples: {triples}")
        return triples, 'N/A'  # 'N/A' for passages_score as this retriever does not return passages
