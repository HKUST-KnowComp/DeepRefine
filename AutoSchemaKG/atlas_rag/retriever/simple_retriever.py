from typing import Dict, List
import numpy as np
from atlas_rag.vectorstore.embedding_model import BaseEmbeddingModel
from atlas_rag.llm_generator.llm_generator import LLMGenerator
from atlas_rag.retriever.base import BaseEdgeRetriever, BasePassageRetriever
from atlas_rag.retriever.inference_config import InferenceConfig
import json_repair
from networkx import DiGraph


def _count_text_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text, disallowed_special=()))
    except Exception:
        return max(1, len(text) // 4)


def _truncate_string_to_token_budget(s: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    lo, hi = 0, len(s)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        chunk = s[:mid]
        if _count_text_tokens(chunk) <= max_tokens:
            best = chunk
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _truncate_triple_lines_to_token_budget(lines: List[str], max_tokens: int, sep: str = "\n") -> List[str]:
    """Keep a prefix of lines so that sep.join(lines) is at most max_tokens; may trim the last line."""
    if max_tokens <= 0 or not lines:
        return lines
    kept: List[str] = []
    ctx = ""
    for line in lines:
        candidate = line if not ctx else ctx + sep + line
        if _count_text_tokens(candidate) <= max_tokens:
            ctx = candidate
            kept.append(line)
            continue
        # print(1111111)
        # Remaining budget for an extra line (or partial line).
        prefix_tokens = _count_text_tokens(ctx + sep) if ctx else 0
        remaining = max_tokens - prefix_tokens
        if remaining <= 0:
            break
        if _count_text_tokens(line) <= remaining:
            kept.append(line)
        else:
            partial = _truncate_string_to_token_budget(line, remaining)
            if partial:
                kept.append(partial)
        break
    return kept


class SimpleGraphRetriever(BaseEdgeRetriever):

    def __init__(self, llm_generator:LLMGenerator, sentence_encoder:BaseEmbeddingModel, 
                 data:dict):
        
        self.KG = data["KG"]
        self.node_list = data["node_list"]
        self.edge_list = data["edge_list"]
        
        self.llm_generator = llm_generator
        self.sentence_encoder = sentence_encoder

        self.node_faiss_index = data["node_faiss_index"]
        self.edge_faiss_index = data["edge_faiss_index"]
        self.KG = self.KG.subgraph(self.node_list)


    def retrieve(self, query, topN=5, **kwargs):
        # retrieve the top k edges
        topk_edges = []
        query_embedding = self.sentence_encoder.encode([query], query_type='edge')
        D, I = self.edge_faiss_index.search(query_embedding, topN)

        # Convert FAISS IDs to list indices using mapping table
        # If mapping table exists, use it; otherwise assume ID == list index (backward compatibility)
        if hasattr(self, 'edge_faiss_id_to_list_idx') and self.edge_faiss_id_to_list_idx:
            list_indices = [self.edge_faiss_id_to_list_idx.get(int(faiss_id), int(faiss_id)) for faiss_id in I[0]]
        else:
            list_indices = I[0]
        
        topk_edges += [self.edge_list[i] for i in list_indices if i < len(self.edge_list)]

        topk_edges_with_data = [(edge[0], self.KG.edges[edge]["relation"], edge[1]) for edge in topk_edges if edge in self.KG.edges]
        string_edge_edges = [f"{self.KG.nodes[edge[0]]['id']}  {edge[1]}  {self.KG.nodes[edge[2]]['id']}" for edge in topk_edges_with_data]

        return string_edge_edges, ["N/A" for _ in range(len(string_edge_edges))]

class SimpleTextRetriever(BasePassageRetriever):
    def __init__(self, passage_dict:Dict[str,str], sentence_encoder:BaseEmbeddingModel, data:dict, inference_config:InferenceConfig=None):  
        self.sentence_encoder = sentence_encoder
        self.passage_dict = passage_dict
        self.passage_list = list(passage_dict.values())
        self.passage_keys = list(passage_dict.keys())
        self.text_embeddings = data["text_embeddings"]
        self.KG = data["KG"]
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
        
        # Get text_faiss_id_to_list_idx mapping if available (for incremental updates)
        self.text_faiss_id_to_list_idx = data.get("text_faiss_id_to_list_idx", {})
        
    def retrieve(self, query, topN=5, **kwargs):
        query_emb = self.sentence_encoder.encode([query], query_type="passage")
        sim_scores = self.text_embeddings @ query_emb[0].T
        topk_indices = np.argsort(sim_scores)[-topN:][::-1]  # Get indices of top-k scores

        # Retrieve top-k passages
        # Note: SimpleTextRetriever uses direct similarity computation, not FAISS index,
        # so indices are already list indices, no mapping needed
        topk_passages = [self.passage_list[i] for i in topk_indices if i < len(self.passage_list)]
        topk_passages_ids = [self.passage_keys[i] for i in topk_indices if i < len(self.passage_keys)]
        topk_passages_ids = [self.text_id_to_node_name[pid] for pid in topk_passages_ids if pid in self.text_id_to_node_name]
        return topk_passages, topk_passages_ids

class SubgraphRetriever(BaseEdgeRetriever):
    def __init__(self, llm_generator: LLMGenerator, sentence_encoder: BaseEmbeddingModel, data: dict, config: InferenceConfig = None):
        self.config = config if config is not None else InferenceConfig()
        self.llm_generator = llm_generator
        self.sentence_encoder = sentence_encoder
        self.KG = data["KG"]
        self.node_list = data["node_list"]
        self.edge_list = data["edge_list"]
        self.node_faiss_index = data["node_faiss_index"]
        self.edge_faiss_index = data["edge_faiss_index"]
        self.node_embeddings = data["node_embeddings"]
        self.num_hop = self.config.num_hop

        self.node_id_to_attr_id = {self.KG.nodes[n]['id']: n for n in self.KG.nodes}
        self.KG = self.KG.subgraph(self.node_list)  # Ensure KG only contains nodes in node_list, filter out passage nodes
        
        # Get ID mapping tables if available (for incremental updates)
        self.node_faiss_id_to_list_idx = data.get("node_faiss_id_to_list_idx", {})
        self.edge_faiss_id_to_list_idx = data.get("edge_faiss_id_to_list_idx", {})

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
        entities_json = json_repair.loads(response)
        if "entities" not in entities_json or not isinstance(entities_json["entities"], list):
            return {}
        return entities_json

    def retrieve_topk_nodes(self, query, topN=5):
        """Retrieve top-k nodes relevant to the query using FAISS index."""
        entities_json = self.ner(query)
        entities = entities_json.get("entities", [])
        if not entities:
            entities = [query]
        topk_nodes = []
        entities_not_in_kg = []
        entities = list(set(str(e) for e in entities))  # Remove duplicates

        for entity in entities:
            entity = self.node_id_to_attr_id.get(entity, entity)
            if entity in self.node_list:
                topk_nodes.append(entity)
            else:
                entities_not_in_kg.append(entity)
        if entities_not_in_kg:
            query_embeddings = self.sentence_encoder.encode(entities_not_in_kg, query_type='node')
            D, I = self.node_faiss_index.search(query_embeddings, 1)  # Get top-1 node per entity
            for i in range(I.shape[0]):
                # Convert FAISS ID to list index using mapping table
                # If mapping table exists, use it; otherwise assume ID == list index (backward compatibility)
                faiss_id = int(I[i][0])
                if self.node_faiss_id_to_list_idx:
                    list_idx = self.node_faiss_id_to_list_idx.get(faiss_id, faiss_id)
                else:
                    list_idx = faiss_id
                
                if list_idx < len(self.node_list):
                    top_node = self.node_list[list_idx]
                    topk_nodes.append(top_node)
        return list(set(topk_nodes))  # Remove duplicates

    def construct_subgraph(self, initial_nodes):
        """Construct a multi-hop subgraph around initial nodes up to self.num_hop."""
        subgraph = DiGraph()
        visited = set()
        queue = [(node, 0) for node in initial_nodes if node in self.node_list]

        # Add initial nodes
        for node, _ in queue:
            subgraph.add_node(node)
            visited.add(node)

        # Breadth-first search to collect neighbors
        while queue:
            current_node, hop_count = queue.pop(0)
            if hop_count >= self.num_hop:
                continue
            # Add successors (outgoing edges)
            for neighbor in self.KG.successors(current_node):
                neighbor_id = self.KG.nodes[neighbor].get('id', None)
                if neighbor_id.isdigit():
                    # Do not further explore this neighbor
                    relation = self.KG.edges[(current_node, neighbor)]["relation"]
                    subgraph.add_edge(current_node, neighbor, relation=relation)
                    continue
                if neighbor not in visited:
                    visited.add(neighbor)
                    subgraph.add_node(neighbor)
                    queue.append((neighbor, hop_count + 1))
                relation = self.KG.edges[(current_node, neighbor)]["relation"]
                subgraph.add_edge(current_node, neighbor, relation=relation)

            # Add predecessors (incoming edges)
            for neighbor in self.KG.predecessors(current_node):
                neighbor_id = self.KG.nodes[neighbor].get('id', None)
                if neighbor_id.isdigit():
                    # Do not further explore this neighbor
                    relation = self.KG.edges[(neighbor, current_node)]["relation"]
                    subgraph.add_edge(neighbor, current_node, relation=relation)
                    continue
                if neighbor not in visited:
                    visited.add(neighbor)
                    subgraph.add_node(neighbor)
                    queue.append((neighbor, hop_count + 1))
                relation = self.KG.edges[(neighbor, current_node)]["relation"]
                subgraph.add_edge(neighbor, current_node, relation=relation)

        return subgraph

    def retrieve(self, question, **kwargs) -> str:
        """Retrieve a subgraph (or full KG) and generate an answer."""
        self.sub_queries = kwargs.get("sub_queries", [])

        initial_nodes = self.retrieve_topk_nodes(question)
        subgraph = self.construct_subgraph(initial_nodes)
        max_ctx_tokens = int(
            kwargs.get("max_subgraph_context_tokens", self.config.max_subgraph_context_tokens)
        )

        lines: List[str] = []
        for u, v, d in subgraph.edges(data=True):
            s = self.KG.nodes[u]["id"]
            r = d["relation"]
            o = self.KG.nodes[v]["id"]
            lines.append(f"({s}, {r}, {o})")

        if max_ctx_tokens > 0:
            lines = _truncate_triple_lines_to_token_budget(lines, max_ctx_tokens)

        return lines, ["N/A" for _ in range(len(lines))]
