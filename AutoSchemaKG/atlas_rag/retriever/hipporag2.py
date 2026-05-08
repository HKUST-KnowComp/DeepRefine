import networkx as nx
import json
from tqdm import tqdm
import json
from tqdm import tqdm
from typing import Dict, List, Tuple
import networkx as nx
import numpy as np
import json_repair
from atlas_rag.vectorstore.embedding_model import BaseEmbeddingModel
from atlas_rag.llm_generator.llm_generator import LLMGenerator
from logging import Logger
from dataclasses import dataclass
from typing import Optional
from atlas_rag.retriever.base import BasePassageRetriever
from atlas_rag.retriever.inference_config import InferenceConfig

def min_max_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val
    
    # Handle the case where all values are the same (range is zero)
    if range_val == 0:
        return np.ones_like(x)  # Return an array of ones with the same shape as x
    
    return (x - min_val) / range_val


class HippoRAG2Retriever(BasePassageRetriever):
    def __init__(self, llm_generator:LLMGenerator, 
                 sentence_encoder:BaseEmbeddingModel, 
                 data : dict, 
                 inference_config: Optional[InferenceConfig] = None,
                 logger = None,
                 **kwargs):
        self.llm_generator = llm_generator
        self.sentence_encoder = sentence_encoder
        self.inference_config = inference_config if inference_config is not None else InferenceConfig()

        self.node_list = data["node_list"]
        self.edge_list = data["edge_list"]
        self.node_embeddings = data["node_embeddings"]
        self.edge_embeddings = data["edge_embeddings"]
        self.edge_embeddings = self.edge_embeddings / np.linalg.norm(self.edge_embeddings, axis=1, keepdims=True)
        self.text_embeddings = data["text_embeddings"]
        if isinstance(self.node_embeddings, list):
            self.node_embeddings = np.array(self.node_embeddings)
        if isinstance(self.edge_embeddings, list):
            self.edge_embeddings = np.array(self.edge_embeddings)
        if isinstance(self.text_embeddings, list):
            self.text_embeddings = np.array(self.text_embeddings)
        self.edge_faiss_index = data["edge_faiss_index"]
        self.passage_dict = data["text_dict"]
        self.text_id_list = list(self.passage_dict.keys())
        self.KG = data["KG"]
        
        self.node_id_to_attr_id = {self.KG.nodes[n]['id']: n for n in self.KG.nodes}
        self.KG = self.KG.subgraph(self.node_list + self.text_id_list)  # Ensure KG only contains nodes in node_list
        
        # Get ID mapping tables if available (for incremental updates)
        self.node_faiss_id_to_list_idx = data.get("node_faiss_id_to_list_idx", {})
        self.edge_faiss_id_to_list_idx = data.get("edge_faiss_id_to_list_idx", {})
        
        
        self.logger = logger
        if self.logger is None:
            self.logging = False
        else:
            self.logging = True

        hipporag2mode = self.inference_config.hipporag_mode if inference_config is not None else "query2edge"
        if hipporag2mode == "query2edge":
            self.retrieve_node_fn = self.query2edge
        elif hipporag2mode == "query2node":
            self.retrieve_node_fn = self.query2node
        elif hipporag2mode == "ner2node":
            self.retrieve_node_fn = self.ner2node
        else:
            raise ValueError(f"Invalid mode: {hipporag2mode}. Choose from 'query2edge', 'query2node', or 'query2passage'.")

        self.inference_config = inference_config if inference_config is not None else InferenceConfig()
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

    def ner(self, text):
        return self.llm_generator.ner(text)
    
    def ner2node(self, query, topN = 10):
        entities = self.ner(query)
        entities = entities.split(", ")

        if len(entities) == 0:
            entities = [query]
        # retrieve the top k nodes
        topk_nodes = []
        node_score_dict = {}
        for entity_index, entity in enumerate(entities):
            topk_for_this_entity = 1
            entity_embedding = self.sentence_encoder.encode([entity], query_type="search")
            scores = min_max_normalize(self.node_embeddings@entity_embedding[0].T)
            index_matrix = np.argsort(scores)[-topk_for_this_entity:][::-1]
            similarity_matrix = [scores[i] for i in index_matrix]
            for index, sim_score in zip(index_matrix, similarity_matrix):
                node = self.node_list[index]
                if node not in topk_nodes:
                    topk_nodes.append(node)
                    node_score_dict[node] = sim_score
                    
        topk_nodes = list(set(topk_nodes))
        result_node_score_dict = {}
        if len(topk_nodes) > 2*topN:
            topk_nodes = topk_nodes[:2*topN]
            for node in topk_nodes:
                if node in node_score_dict:
                    result_node_score_dict[node] = node_score_dict[node]
        return result_node_score_dict
    
    def query2node(self, query, topN = 10):
        query_emb = self.sentence_encoder.encode([query], query_type="entity")
        scores = min_max_normalize(self.node_embeddings@query_emb[0].T)
        index_matrix = np.argsort(scores)[-topN:][::-1]
        similarity_matrix = [scores[i] for i in index_matrix]
        result_node_score_dict = {}
        for index, sim_score in zip(index_matrix, similarity_matrix):
            node = self.node_list[index]
            result_node_score_dict[node] = sim_score

        return result_node_score_dict
    
    def query2edge(self, query, topN = 10):
        query_emb = self.sentence_encoder.encode([query], query_type="edge")
        # normalize the embeddings
        query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
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
    
    def query2passage(self, query, weight_adjust = 0.05):
        query_emb = self.sentence_encoder.encode([query], query_type="passage")
        sim_scores = self.text_embeddings @ query_emb[0].T
        sim_scores = min_max_normalize(sim_scores)*weight_adjust # converted to probability
        # create dict of passage id and score
        return dict(zip(self.text_id_list, sim_scores))
    
    def retrieve_personalization_dict(self, query, topN=30, weight_adjust=0.05):
        node_dict = self.retrieve_node_fn(query, topN=topN)
        text_dict = self.query2passage(query, weight_adjust=weight_adjust)
  
        return node_dict, text_dict

    def retrieve(self, query, topN=5, **kwargs):
        topN_edges = self.inference_config.topk_edges
        weight_adjust = self.inference_config.weight_adjust
        
        node_dict, text_dict = self.retrieve_personalization_dict(query, topN=topN_edges, weight_adjust=weight_adjust)
          
        personalization_dict = {}
        if len(node_dict) == 0:
            # return topN text passages
            sorted_passages = sorted(text_dict.items(), key=lambda x: x[1], reverse=True)
            sorted_passages = sorted_passages[:topN]
            sorted_passages_contents = []
            sorted_scores = []
            sorted_passage_ids = []
            for passage_id, score in sorted_passages:
                sorted_passages_contents.append(self.passage_dict[passage_id])
                sorted_scores.append(float(score))
                sorted_passage_ids.append(self.text_id_to_node_name[passage_id])
            return sorted_passages_contents, sorted_passage_ids
            
        personalization_dict.update(node_dict)
        personalization_dict.update(text_dict)
        # retrieve the top N passages
        pr = nx.pagerank(self.KG, personalization=personalization_dict, 
                         alpha = self.inference_config.ppr_alpha, 
                         max_iter=self.inference_config.ppr_max_iter, 
                         tol=self.inference_config.ppr_tol)

        # get the top N passages based on the text_id list and pagerank score
        text_dict_score = {}
        for node in self.text_id_list:
            # filter out nodes that have 0 score
            if pr[node] > 0.0:
                text_dict_score[node] = pr[node]
            
        # return topN passages
        sorted_passages_ids = sorted(text_dict_score.items(), key=lambda x: x[1], reverse=True)
        sorted_passages_ids = sorted_passages_ids[:topN]
        
        sorted_passages_contents = []
        sorted_scores = []
        sorted_passage_ids = []
        for passage_id, score in sorted_passages_ids:
            sorted_passages_contents.append(self.passage_dict[passage_id])
            sorted_scores.append(score)
            sorted_passage_ids.append(self.text_id_to_node_name[passage_id])
        return sorted_passages_contents, sorted_passage_ids