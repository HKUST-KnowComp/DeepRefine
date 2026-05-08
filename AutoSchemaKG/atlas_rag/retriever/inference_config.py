from dataclasses import asdict, dataclass

@dataclass
class InferenceConfig:
    """
    Configuration class for inference settings.
    
    Attributes:
        topk (int): Number of top results to retrieve. Default is 5.
        Dmax (int): Maximum depth for search. Default is 4.
        weight_adjust (float): Weight adjustment factor for passage retrieval. Default is 1.0.
        topk_edges (int): Number of top edges to retrieve. Default is 50.
        topk_nodes (int): Number of top nodes to retrieve. Default is 10.
    """
    keyword: str = "musique"
    topk: int = 5
    Dmax: int = 3
    Wmax: int = 3
    weight_adjust: float = 1.0
    topk_edges: int = 50
    topk_nodes: int = 10
    ppr_alpha: float = 0.99
    ppr_max_iter: int = 2000
    ppr_tol: float = 1e-7

    # tog config
    topic_prune: bool = True
    temperature_exploration:float = 0.0
    temperature_reasoning:float = 0.0
    num_sents_for_reasoning: int = 10
    remove_unnecessary_rel: bool = True

    # subgraph retriever config
    num_hop: int = 1
    # SubgraphRetriever: if the joined triple context exceeds this many tokens, truncate (keep prefix).
    # Uses tiktoken cl100k_base when available; else falls back to len(text)//4. 0 = no limit.
    max_subgraph_context_tokens: int = 30000
    
    # hipporag 1 and 2 config
    is_filter_edges: bool = True
    hipporag_mode: str = "query2edge"  # options: query2edge, query2node

    def __str__(self):
        return "\n".join(f"{key}: {value}" for key, value in asdict(self).items())
