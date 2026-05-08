import numpy as np
from atlas_rag.vectorstore.embedding_model import BaseEmbeddingModel
from atlas_rag.llm_generator.llm_generator import (
    LLMGenerator,
    _count_tokens_api_safe,
    _truncate_string_to_token_budget_api_safe,
)
from typing import Optional
from atlas_rag.retriever.base import BaseEdgeRetriever
from atlas_rag.retriever.inference_config import InferenceConfig
import json_repair
import networkx as nx

class TogV3Retriever(BaseEdgeRetriever):
    def __init__(self, llm_generator, sentence_encoder, data, inference_config: Optional[InferenceConfig] = None):
        self.KG = data["KG"]

        self.node_list = data["node_list"]
        self.KG:nx.DiGraph = self.KG.subgraph(self.node_list)

        self.edge_list = list(self.KG.edges)
        self.edge_list_with_relation = [(edge[0], self.KG.edges[edge]["relation"], edge[1])  for edge in self.edge_list]
        self.edge_list_string = [f"{edge[0]}  {self.KG.edges[edge]['relation']}  {edge[1]}" for edge in self.edge_list]
        
        self.llm_generator:LLMGenerator = llm_generator
        self.sentence_encoder:BaseEmbeddingModel = sentence_encoder        

        self.node_embeddings = data["node_embeddings"]
        self.edge_embeddings = data["edge_embeddings"]

        self.inference_config = inference_config if inference_config is not None else InferenceConfig()

    def ner(self, text):
        """Extract topic entities from the query using LLM."""
        messages = [
            {
                "role": "system",
                "content": "Extract the named entities from the provided question and output them as a JSON object in the format: {\"entities\": [\"entity1\", \"entity2\", ...]}"
            },
            {
                "role": "user",
                "content": f"Extract all the named entities from: {text}"
            }
        ]
        response = self.llm_generator.generate_response(messages)
        try:
            entities_json = json_repair.loads(response)
        except Exception as e:
            return {}
        if "entities" not in entities_json or not isinstance(entities_json["entities"], list):
            return {}
        return entities_json
    


    def retrieve_topk_nodes(self, query, topN=5, **kwargs):
        # extract entities from the query
        entities = self.ner(query)
        entities = entities.get("entities", [])

        if len(entities) == 0:
            # If the NER cannot extract any entities, we 
            # use the query as the entity to do approximate search
            entities = [query]
            query_embeddings = self.sentence_encoder.encode([query], query_type='entity')
            scores = self.node_embeddings @ query_embeddings[0].T
            top_indices = np.argsort(scores)[-topN:][::-1]
            return [self.node_list[i] for i in top_indices]
        # evenly distribute the topk for each entity
        topk_for_each_entity = topN//len(entities)
    
        # retrieve the top k nodes
        topk_nodes = []

        for entity_index, entity in enumerate(entities):
            if entity in self.node_list:
                topk_nodes.append(entity)
    
        for entity_index, entity in enumerate(entities): 
            topk_for_this_entity = topk_for_each_entity + 1
            
            entity_embedding = self.sentence_encoder.encode([entity])
            # Calculate similarity scores using dot product
            scores = self.node_embeddings @ entity_embedding[0].T
            # Get top-k indices
            top_indices = np.argsort(scores)[-topk_for_this_entity:][::-1]
            topk_nodes += [self.node_list[i] for i in top_indices]
            
        topk_nodes = list(set(topk_nodes))

        if len(topk_nodes) > 2*topN:
            topk_nodes = topk_nodes[:2*topN]
        return topk_nodes

    def retrieve(self, query, topN=5, **kwargs):
        """ 
        Retrieve the top N paths that connect the entities in the query.
        Dmax is the maximum depth of the search.
        """
        Dmax = self.inference_config.Dmax
        # in the first step, we retrieve the top k nodes
        initial_nodes = self.retrieve_topk_nodes(query, topN=topN)
        E = initial_nodes
        P = [ [e] for e in E]
        D = 0

        while D <= Dmax:
            P = self.search(query, P)
            P = self.prune(query, P, topN)
            if self.reasoning(query, P):
                generated_text = self.generate(query, P)
                break
            
            D += 1
        
        if D > Dmax:    
            generated_text = self.generate(query, P)
        
        # print(generated_text)
        return generated_text

    def search(self, query, P):
        new_paths = []
        for path in P:
            tail_entity = path[-1]
            sucessors = list(self.KG.successors(tail_entity))
            predecessors = list(self.KG.predecessors(tail_entity))

            # print(f"tail_entity: {tail_entity}")
            # print(f"sucessors: {sucessors}")
            # print(f"predecessors: {predecessors}")

            # # print the attributes of the tail_entity
            # print(f"attributes of the tail_entity: {self.KG.nodes[tail_entity]}")
           
            # remove the entity that is already in the path
            sucessors = [neighbour for neighbour in sucessors if neighbour not in path]
            # predecessors = [neighbour for neighbour in predecessors if neighbour not in path]

            if len(sucessors) == 0:
                new_paths.append(path)
                continue
            for neighbour in sucessors:
                relation = self.KG.edges[(tail_entity, neighbour)]["relation"]
                new_path = path + [relation, neighbour]
                new_paths.append(new_path)
            
            # for neighbour in predecessors:
            #     relation = self.KG.edges[(neighbour, tail_entity)]["relation"]
            #     new_path = path + [relation, neighbour]
            #     new_paths.append(new_path)
        
        return new_paths
    
    def prune(self, query, P, topN=3):
        # Create path strings for embedding comparison
        path_strings = []
        
        for path in P:
            # Format path for embedding model
            formatted_nodes = []
            for i, node_or_relation in enumerate(path):
                if i % 2 == 0:
                    formatted_nodes.append(self.KG.nodes[node_or_relation]["id"])
                else:
                    formatted_nodes.append(node_or_relation)
            
            # Join the formatted path elements
            path_string = " ".join(formatted_nodes)
            path_strings.append(path_string)
        
        # Encode query and paths
        # Pass query_type='edge' for appropriate prefixing in embedding models that support it
        query_embedding = self.sentence_encoder.encode([query], query_type='edge')[0]
        path_embeddings = self.sentence_encoder.encode(path_strings)
        
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        path_embeddings = path_embeddings / np.linalg.norm(path_embeddings, axis=1, keepdims=True)

        # Compute similarity scores
        scores = path_embeddings @ query_embedding
        
        # Sort paths by similarity scores (higher is better)
        sorted_indices = np.argsort(scores)[::-1]  # Descending order
        sorted_paths = [P[i] for i in sorted_indices[:topN]]
        
        return sorted_paths

    def reasoning(self, query, P):
        triples = []
        for path in P:
            for i in range(0, len(path)-2, 2):

                # triples.append((path[i], path[i+1], path[i+2]))
                triples.append((self.KG.nodes[path[i]]["id"], path[i+1], self.KG.nodes[path[i+2]]["id"]))
        
        triples_string = [f"({triple[0]}, {triple[1]}, {triple[2]})" for triple in triples]
        triples_blob = ". ".join(triples_string)

        prefix = (
            "Given a question and the associated retrieved knowledge graph triples (entity, relation, entity), "
            "you are asked to answer whether it's sufficient for you to answer the question with these triples "
            f"and your knowledge (Yes or No). Query: {query} \n Knowledge triples: "
        )
        system_content = "Answer the question following the prompt."
        max_in = getattr(self.llm_generator.config, "max_prompt_tokens", 30720)
        fixed = _count_tokens_api_safe(system_content) + _count_tokens_api_safe(prefix) + 96
        budget = max_in - fixed
        if budget > 0 and _count_tokens_api_safe(triples_blob) > budget:
            triples_blob = _truncate_string_to_token_budget_api_safe(triples_blob, budget)

        prompt = prefix + triples_blob

        messages = [{"role": "system", "content": system_content},
        {"role": "user", "content": f"{prompt}"}]

        response = self.llm_generator.generate_response(messages)
        return "yes" in response.lower()

    def generate(self, query, P):
        triples = []
        for path in P:
            for i in range(0, len(path)-2, 2):
                # triples.append((path[i], path[i+1], path[i+2]))
                triples.append((self.KG.nodes[path[i]]["id"], path[i+1], self.KG.nodes[path[i+2]]["id"]))
        
        triples_string = [f"({triple[0]}, {triple[1]}, {triple[2]})" for triple in triples]
        
        # response = self.llm_generator.generate_with_context_kg(query, triples_string)
        return triples_string, ["N/A" for _ in range(len(triples_string))]