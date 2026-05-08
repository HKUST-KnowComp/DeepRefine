from tqdm import tqdm
import networkx as nx
import numpy as np
from atlas_rag.vectorstore.embedding_model import BaseEmbeddingModel
from atlas_rag.llm_generator.llm_generator import LLMGenerator
from logging import Logger
from typing import Optional
from atlas_rag.retriever.base import BasePassageRetriever
from atlas_rag.retriever.inference_config import InferenceConfig
import json_repair
import json
def min_max_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val
    
    # Handle the case where all values are the same (range is zero)
    if range_val == 0:
        return np.ones_like(x)  # Return an array of ones with the same shape as x
    
    return (x - min_val) / range_val

class HippoRAGRetriever(BasePassageRetriever):
    def __init__(self, llm_generator:LLMGenerator, sentence_encoder:BaseEmbeddingModel, 
                 data:dict,  inference_config: Optional[InferenceConfig] = None, logger = None, **kwargs):
        self.inference_config = inference_config if inference_config is not None else InferenceConfig()  
        self.passage_dict = data["text_dict"]
        self.llm_generator = llm_generator
        self.sentence_encoder = sentence_encoder
        self.node_embeddings = data["node_embeddings"]
        if isinstance(self.node_embeddings, list):
            self.node_embeddings = np.array(self.node_embeddings)
        self.node_list = data["node_list"]
        self.edge_list = data["edge_list"]
        self.node_embeddings = data["node_embeddings"]
        self.edge_embeddings = data["edge_embeddings"]
        self.edge_embeddings = self.edge_embeddings / np.linalg.norm(self.edge_embeddings, axis=1, keepdims=True)
        self.text_embeddings = data["text_embeddings"]
        file_id_to_node_id = {}
        self.KG = data["KG"]
        for node_id in tqdm(list(self.KG.nodes)):
            if self.KG.nodes[node_id]['type'] == "passage":
                file_ids = self.KG.nodes[node_id]['file_id'].split(',')
                for file_id in file_ids:
                    if file_id not in file_id_to_node_id:
                        file_id_to_node_id[file_id] = []
                    file_id_to_node_id[file_id].append(node_id)
        # further filter any file_id that is not passage type
        self.file_id_to_node_id = file_id_to_node_id

        node_id_to_file_id = {}
        text_id_to_node_name = {}
        for node_id in list(self.KG.nodes):
            if self.inference_config.keyword == "musique" and self.KG.nodes[node_id]['type']=="passage":
                text_id_to_node_name[node_id] = self.KG.nodes[node_id]["id"]
            elif self.KG.nodes[node_id]['type']=="passage":
                text_id_to_node_name[node_id] = self.KG.nodes[node_id]["id"]
            else:
                node_id_to_file_id[node_id] = self.KG.nodes[node_id]["file_id"]
        self.node_id_to_file_id = node_id_to_file_id
        self.text_id_to_node_name = text_id_to_node_name
        
        self.node_id_to_attr_id = {self.KG.nodes[n]['id']: n for n in self.KG.nodes}
        self.KG:nx.DiGraph = self.KG.subgraph(self.node_list)  # Ensure KG only contains nodes in node_list
        
        # Get ID mapping tables if available (for incremental updates)
        self.node_faiss_id_to_list_idx = data.get("node_faiss_id_to_list_idx", {})
        self.edge_faiss_id_to_list_idx = data.get("edge_faiss_id_to_list_idx", {})
        
        self.node_name_list = [self.KG.nodes[node]["id"] for node in self.node_list]
        
        if self.inference_config.hipporag_mode == "query2edge":
            self.q2kg_fn = self.query2edge
        else:
            self.q2kg_fn = self.retrieve_personalization_dict
        
        self.logger :Logger = logger
        if self.logger is None:
            self.logging = False
        else:
            self.logging = True

    def query2edge(self, query, topN = 10):
        query_emb = self.sentence_encoder.encode([query], query_type="edge")
        # normalize the embeddings
        query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
        # compute the scores
        scores = min_max_normalize(self.edge_embeddings@query_emb[0].T)
        index_matrix = np.argsort(scores)[-topN:][::-1]
        log_edge_list = []
        for index in index_matrix:
            edge = self.edge_list[index]
            if edge[0] not in self.KG.nodes or edge[1] not in self.KG.nodes or edge not in self.KG.edges:
                continue
            edge_str = [self.KG.nodes[edge[0]]['id'], self.KG.edges[edge]['relation'], self.KG.nodes[edge[1]]['id']]
            log_edge_list.append(edge_str)

        similarity_matrix = [scores[i] for i in index_matrix]
        # construct the edge list
        before_filter_edge_json = {}
        before_filter_edge_json['fact'] = []
        for index, sim_score in zip(index_matrix, similarity_matrix):
            edge = self.edge_list[index]
            if edge[0] not in self.KG.nodes or edge[1] not in self.KG.nodes or edge not in self.KG.edges:
                continue
            edge_str = [self.KG.nodes[edge[0]]['id'], self.KG.edges[edge]['relation'], self.KG.nodes[edge[1]]['id']]
            before_filter_edge_json['fact'].append(edge_str)
        # if self.logging:
        #     self.logger.info(f"HippoRAG2 Before Filter Edge: {before_filter_edge_json['fact']}")
        if not self.inference_config.is_filter_edges:
            node_score_dict = {}
            for index, sim_score in zip(index_matrix, similarity_matrix):
                edge = self.edge_list[index]
                if edge[0] not in self.KG.nodes or edge[1] not in self.KG.nodes or edge not in self.KG.edges:
                    continue
                head, tail = edge[0], edge[1]
                if head not in node_score_dict:
                    node_score_dict[head] = [sim_score]
                else:
                    node_score_dict[head].append(sim_score)
                if tail not in node_score_dict:
                    node_score_dict[tail] = [sim_score]
                else:
                    node_score_dict[tail].append(sim_score)
            # average the scores
            for node in node_score_dict:
                node_score_dict[node] = sum(node_score_dict[node]) / len(node_score_dict[node])
            
            # get the topN nodes
            if len(node_score_dict) > self.inference_config.topk_nodes:
                sorted_nodes = sorted(node_score_dict.items(), key=lambda x: x[1], reverse=True)
                sorted_nodes = sorted_nodes[:self.inference_config.topk_nodes]
                node_score_dict = dict(sorted_nodes)

            # if self.logging:
            #     self.logger.info(f"HippoRAG2: Unfiltered node: {node_score_dict}")
            
            return node_score_dict
            
        filtered_facts = self.llm_generator.filter_triples_with_entity_event(query, json.dumps(before_filter_edge_json, ensure_ascii=False))
        try:
            filtered_facts = json_repair.loads(filtered_facts)['fact']
        except Exception as e:
            filtered_facts = before_filter_edge_json['fact']
        if len(filtered_facts) == 0:
            return {}
        # use filtered facts to get the edge id and check if it exists in the original candidate list.
        node_score_dict = {}
        log_edge_list = []
        for edge in filtered_facts:
            edge_str = f'{edge[0]} {edge[1]} {edge[2]}'
            search_emb = self.sentence_encoder.encode([edge_str], query_type="search")
            D, I = self.edge_faiss_index.search(search_emb, 1)
            filtered_index = I[0][0]
            # get the edge and the original score
            edge = self.edge_list[filtered_index]
            log_edge_list.append([self.KG.nodes[edge[0]]['id'], self.KG.edges[edge]['relation'], self.KG.nodes[edge[1]]['id']])
            head, tail = edge[0], edge[1]
            # check if head/tails is concept, sim_score = sim_score / # edge of that node 
            sim_score = scores[filtered_index]
            
            if head not in node_score_dict:
                node_score_dict[head] = [sim_score]
            else:
                node_score_dict[head].append(sim_score)
            if tail not in node_score_dict:
                node_score_dict[tail] = [sim_score]
            else:
                node_score_dict[tail].append(sim_score)
        # average the scores
        # if self.logging:
        #     self.logger.info(f"HippoRAG: Filtered edges: {log_edge_list}")
        
        # take average of the scores
        for node in node_score_dict:
            node_score_dict[node] = sum(node_score_dict[node]) / len(node_score_dict[node])
        
        # get the topN nodes
        if len(node_score_dict) > self.inference_config.topk_nodes:
            sorted_nodes = sorted(node_score_dict.items(), key=lambda x: x[1], reverse=True)
            sorted_nodes = sorted_nodes[:self.inference_config.topk_nodes]
            node_score_dict = dict(sorted_nodes)

        # if self.logging:
        #     self.logger.info(f"HippoRAG: Filtered node: {node_score_dict}")
        
        
        return node_score_dict
        
    def retrieve_personalization_dict(self, query, topN=10):

        # extract entities from the query
        entities = self.llm_generator.ner(query)
        entities = entities.split(", ")
        # if self.logging:
        #     self.logger.info(f"HippoRAG NER Entities: {entities}")
        # print("Entities:", entities)

        if len(entities) == 0:
            # If the NER cannot extract any entities, we 
            # use the query as the entity to do approximate search
            entities = [query]
    
        # evenly distribute the topk for each entity
        topk_for_each_entity = topN//len(entities)
    
        # retrieve the top k nodes
        topk_nodes = []

        for entity_index, entity in enumerate(entities):
            if entity in self.node_name_list:
                # get the index of the entity in the node list
                index = self.node_name_list.index(entity)
                topk_nodes.append(self.node_list[index])
            else:
                topk_for_this_entity = 1
                
                # print("Topk for this entity:", topk_for_this_entity)
                
                entity_embedding = self.sentence_encoder.encode([entity], query_type="search")
                scores = self.node_embeddings@entity_embedding[0].T
                index_matrix = np.argsort(scores)[-topk_for_this_entity:][::-1]
               
                topk_nodes += [self.node_list[i] for i in index_matrix]
        
        # if self.logging:
        #     self.logger.info(f"HippoRAG Topk Nodes: {[self.KG.nodes[node]['id'] for node in topk_nodes]}")
        
        topk_nodes = list(set(topk_nodes))

        # assert len(topk_nodes) <= topN
        if len(topk_nodes) > topN:
            topk_nodes = topk_nodes[:topN]

        
        # print("Topk nodes:", topk_nodes)
        # find the number of docs that one work appears in
        freq_dict_for_nodes = {}
        for node in topk_nodes:
            node_data = self.KG.nodes[node]
            # print(node_data)
            file_ids = node_data["file_id"]
            file_ids_list = file_ids.split(",")
            #uniq this list
            file_ids_list = list(set(file_ids_list))
            freq_dict_for_nodes[node] = len(file_ids_list)

        personalization_dict = {node: 1 / freq_dict_for_nodes[node]  for node in topk_nodes}

        # print("personalization dict: ")
        return personalization_dict

    def retrieve(self, query, topN=5, **kwargs):
        topN_nodes = self.inference_config.topk_nodes
        if not self.inference_config.is_filter_edges:
            personalization_dict = self.q2kg_fn(query, topN=self.inference_config.topk_edges)
        else:
            personalization_dict = self.q2kg_fn(query, topN=topN_nodes)

        # retrieve the top N passages
        pr = nx.pagerank(self.KG, personalization=personalization_dict, alpha=self.inference_config.ppr_alpha, 
                         max_iter=self.inference_config.ppr_max_iter, tol=self.inference_config.ppr_tol)

        for node in pr:
            pr[node] = round(pr[node], 4)
            if pr[node] < 0.001:
                pr[node] = 0
        
        passage_probabilities_sum = {}
        for node in pr:
            node_data = self.KG.nodes[node]
            file_ids = node_data["file_id"]
            # for each file id check through each text_id
            if file_ids is not None:
                file_ids_list = file_ids.split(",")
                #uniq this list
                file_ids_list = list(set(file_ids_list))
                # file id to node id
                
                for file_id in file_ids_list:
                    if file_id == 'concept_file':
                        continue
                    for node_id in self.file_id_to_node_id[file_id]:
                        if node_id not in passage_probabilities_sum:
                            passage_probabilities_sum[node_id] = 0
                        passage_probabilities_sum[node_id] += pr[node]
        
        sorted_passages = sorted(passage_probabilities_sum.items(), key=lambda x: x[1], reverse=True)
        top_passages = sorted_passages[:topN]
        top_passages, scores = zip(*top_passages)

        passage_contents = [self.passage_dict[passage_id] for passage_id in top_passages]
        top_passages = [self.text_id_to_node_name[passage_id] for passage_id in top_passages]
        return passage_contents, top_passages