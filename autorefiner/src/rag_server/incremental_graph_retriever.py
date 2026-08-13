"""Graph retriever that stays consistent after incremental edge insert/delete."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from atlas_rag.retriever.simple_retriever import SimpleGraphRetriever


class IncrementalGraphRetriever(SimpleGraphRetriever):
    """SimpleGraphRetriever with faiss_id -> edge_list index mapping.

    The upstream atlas_rag package treats FAISS search IDs as ``edge_list``
    indices. After DeepRefine mutates the KG with ``add_with_ids`` /
    ``remove_ids``, those IDs diverge and ``retrieve`` raises IndexError.

    Important: ``edge_faiss_id_to_list_idx`` must be the *same dict object*
    that DeepRefine mutates (do not copy), otherwise insert/delete updates
    are invisible until an explicit reassignment.
    """

    def __init__(self, llm_generator, sentence_encoder, data: dict):
        super().__init__(llm_generator, sentence_encoder, data)
        # Share references with DeepRefine / data; do not copy.
        self.edge_faiss_id_to_list_idx: Dict[int, int] = data.setdefault(
            "edge_faiss_id_to_list_idx", {}
        )
        self.node_faiss_id_to_list_idx: Dict[int, int] = data.setdefault(
            "node_faiss_id_to_list_idx", {}
        )

    def _faiss_ids_to_list_indices(self, faiss_ids) -> List[int]:
        mapping = self.edge_faiss_id_to_list_idx
        out: List[int] = []
        seen = set()
        n_edges = len(self.edge_list)
        for faiss_id in faiss_ids:
            fid = int(faiss_id)
            if fid < 0:
                continue
            if mapping:
                if fid not in mapping:
                    # Deleted / unknown FAISS id: never fall back to using
                    # faiss_id as a list index (that is the original bug).
                    continue
                list_idx = int(mapping[fid])
            else:
                # Empty mapping is only safe as identity when FAISS still uses
                # contiguous 0..n-1 ids matching edge_list length.
                ntotal = int(getattr(self.edge_faiss_index, "ntotal", 0) or 0)
                if ntotal != n_edges:
                    continue
                list_idx = fid
            if list_idx < 0 or list_idx >= n_edges:
                continue
            if list_idx in seen:
                continue
            seen.add(list_idx)
            out.append(list_idx)
        return out

    def retrieve(self, query, topN=5, **kwargs) -> Tuple[List[str], List[str]]:
        if not self.edge_list or getattr(self.edge_faiss_index, "ntotal", 0) == 0:
            return [], []

        query_embedding = self.sentence_encoder.encode([query], query_type="edge")
        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        k = min(int(topN), len(self.edge_list), int(self.edge_faiss_index.ntotal))
        if k <= 0:
            return [], []

        _d, faiss_ids = self.edge_faiss_index.search(query_embedding, k)
        list_indices = self._faiss_ids_to_list_indices(faiss_ids[0])
        topk_edges = [self.edge_list[i] for i in list_indices]

        topk_edges_with_data = [
            (edge[0], self.KG.edges[edge]["relation"], edge[1])
            for edge in topk_edges
            if edge in self.KG.edges
        ]
        string_edge_edges = [
            f"{self.KG.nodes[edge[0]]['id']}  {edge[1]}  {self.KG.nodes[edge[2]]['id']}"
            for edge in topk_edges_with_data
            if edge[0] in self.KG.nodes and edge[2] in self.KG.nodes
        ]
        return string_edge_edges, ["N/A" for _ in range(len(string_edge_edges))]
