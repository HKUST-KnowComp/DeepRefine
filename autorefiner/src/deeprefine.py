from __future__ import annotations
import random
import re
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import networkx as nx
import numpy as np
import faiss
import json_repair
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, Iterable, Set, Callable
from atlas_rag.retriever.base import BaseEdgeRetriever, BasePassageRetriever
from atlas_rag.llm_generator import LLMGenerator
from atlas_rag.vectorstore.embedding_model import BaseEmbeddingModel
from autorefiner.src.rag_server.deeprefine_prompt import (
    REAFINER_JUDGEMENT_SYSTEM_PROMPT,
    REAFINER_JUDGEMENT_USER_PROMPT,
    REAFINER_EXPANDED_JUDGEMENT_USER_PROMPT,
    REAFINER_ERROR_ABDUCTION_SYSTEM_PROMPT,
    REAFINER_ERROR_ABDUCTION_USER_PROMPT,
    REAFINER_ABDUCTION_USER_PROMPT,
    REAFINER_KG_REFINEMENT_SYSTEM_PROMPT,
    REAFINER_KG_REFINEMENT_USER_PROMPT,
    REFINE_SUBGRAPH_SYSTEM_PROMPT,
    REFINE_SUBGRAPH_USER_PROMPT,
    REAFINER_FILTERING_SYSTEM_PROMPT,
    REAFINER_FILTERING_USER_PROMPT,
    REAFINER_KG_REFINEMENT_ACTION_SYSTEM_PROMPT,
    REAFINER_KG_REFINEMENT_ACTION_USER_PROMPT,
    REAFINER_ACTION_USER_PROMPT,
)
from atlas_rag.retriever.simple_retriever import SimpleGraphRetriever
from autorefiner.src.rag_server.incremental_graph_retriever import IncrementalGraphRetriever
from atlas_rag.retriever.hipporag import HippoRAGRetriever
from atlas_rag.evaluation.evaluation import QAJudger
from networkx import DiGraph
from tqdm import tqdm
try:
    import torch
except Exception:
    torch = None


_ILLEGAL_XML_RE = re.compile(
    "[" +
    "\x00-\x08" +
    "\x0B" +
    "\x0C" +
    "\x0E-\x1F" +
    "\uD800-\uDFFF" +   # Surrogates
    "\uFFFE\uFFFF" +    # Noncharacters
    "]"
)

@dataclass
class RetrievalStepResult:
    """
    For single step inference result, for debugging / analysis.
    """
    num_hops: int
    base_top_k: int
    query: str
    retrieved_subgraph: List[Dict[str, str]]
    raw_response: str
    answerable: bool
    answer: Optional[str] = None

@dataclass
class RefinementResult:
    """
    For single refinement result, for debugging / analysis.
    """
    query: str
    history_horizon_size: int
    interaction_history: List[RetrievalStepResult]
    error_abduction_reason: str
    original_subgraph: List[Dict[str, str]]
    refined_subgraph: List[Dict[str, str]]
    refinement_action_list: List[Callable[[], None]]
    refinement_action_raw: str


class DeepRefine:
    """
    - Minimal / K-hop Retrieve: Retrieve a subgraph from KG based on text vector index (support multi-hop iteration expansion).
    - Answerable Judgement: Judge if the current subgraph is enough to answer the query.
    - Error Abduction: If not answerable, let LLM analyze "why not answerable", summarize redundant / incomplete / incorrect information.
    - Subgraph Expanding & Refined KG Generation: Generate new triples based on the reason of the previous step, update the KG incrementally.
    - In-loop Retrieve: Continue retrieving with the updated KG until answerable or reach the step limit.
    """

    def __init__(
        self,
        data: dict,
        sentence_encoder: BaseEmbeddingModel,
        llm_generator: LLMGenerator,
        base_top_k: int = 10,
        increament_hop: int = 1,
        max_hops: int = 3,
        max_triple_num: int = 90,
        max_triple_num_by_step: Optional[List[int]] = None,
        history_horizon_size: int = 3,
        if_gen_answer: bool = True,
        seed: int = 2026,
        ground_inserts: bool = True,
        ground_mode: str = "both",
        require_query_overlap: bool = False,
        skip_generic_objects: bool = True,
        skip_conflict_inserts: bool = True,
        skip_action_if_answerable: bool = True,
        max_actions: int = 10,
        max_format_retry: int = 2,
    ) -> None:
        """
        - data:           Dictionary containing the following keys:
            - KG:             Complete KG (networkx.DiGraph, at least 'id' / 'type' in node attributes, 'relation' in edge attributes).
            - node_list:      List of node ids in the KG.
            - edge_list:      List of edge tuples in the KG.
            - node_faiss_index: Faiss index for node retrieval.
            - edge_faiss_index: Faiss index for edge retrieval.
            - text_node_dict:   Dictionary mapping node ids to text.
        - sentence_encoder: Encoder corresponding to text_faiss_index, for encoding query.
        - llm_generator:    atlas_rag.llm_generator.LLMGenerator instance.
        - base_top_k:       TopK for text vector retrieval for the 1st step.
        - max_triple_num:   Maximum number of triples for subgraph pruning (used when max_triple_num_by_step is None).
        - max_triple_num_by_step: Optional per-step caps [step1, step2, ...] for retrieval top-N (step 1) and prune budget (step>=2).
          Defaults to training schedule [10, 50, 90] when None. If shorter than max_hops, the last value is repeated.
        - if_gen_answer:    Whether to generate answer in the refinement process.
        - max_hops:         Maximum number of hops for subgraph expansion (training default: 3).
        - history_horizon_size: Size of the interaction history to be considered for error abduction.
        """
        self.data = data
        self.kg = data["KG"]
        self.node_list = data["node_list"]  # no passage nodes
        self.edge_list = data["edge_list"]  # not include passage nodes
        self.node_faiss_index = data["node_faiss_index"]    # no passage nodes
        self.edge_faiss_index = data["edge_faiss_index"]    # not include passage nodes
        self.node_embeddings = data["node_embeddings"]
        self.edge_embeddings = data["edge_embeddings"]
        self.sentence_encoder = sentence_encoder
        self.llm_generator = llm_generator
        self.base_top_k = base_top_k
        self.max_hops = max_hops
        self.increament_hop = increament_hop
        self.history_horizon_size = history_horizon_size
        self.if_gen_answer = if_gen_answer
        self.max_triple_num = max_triple_num
        # Offline / eval tricks (default on): reduce hallucinated inserts that pollute the global KG.
        self.ground_inserts = bool(ground_inserts)
        mode = (ground_mode or "both").strip().lower()
        if mode not in {"both", "either", "off"}:
            mode = "both"
        self.ground_mode = mode
        if mode == "off":
            self.ground_inserts = False
        self.require_query_overlap = bool(require_query_overlap)
        self.skip_generic_objects = bool(skip_generic_objects)
        self.skip_conflict_inserts = bool(skip_conflict_inserts)
        # When final judgement is Yes, skip abduction/action (anti-pollution for global KG merge).
        # Training still abducts after Yes on hop>1; this is an offline-only divergence.
        self.skip_action_if_answerable = bool(skip_action_if_answerable)
        self.max_actions = int(max_actions)
        self.max_format_retry = int(max_format_retry)
        self._GENERIC_OBJECTS = {
            "",
            "unknown",
            "n/a",
            "na",
            "none",
            "null",
            "?",
            "-",
            "not mentioned",
            "not specified",
            "unspecified",
        }
        # Per-thread ephemeral state so parallel propose (apply_actions=False) is safe.
        self._tls = threading.local()
        # Align with training RefinementInteraction when caller omits the schedule.
        if max_triple_num_by_step is None:
            max_triple_num_by_step = [10, 50, 90]
        self._max_triple_num_by_step: Optional[List[int]] = (
            list(max_triple_num_by_step) if max_triple_num_by_step is not None else None
        )
        self.seed = seed
        self._set_seed(seed)
        self._dim = self.data["text_faiss_index"].d
        # Ensure the order of text_node_dict keys matches the order when building text_index (pickle maintains insertion order by default).
        self._text_node_ids: List[str] = list(self.data["text_dict"].keys())

        node_id_to_file_id = {}
        text_id_to_node_name = {}
        for node_id in list(self.kg.nodes):
            if self.kg.nodes[node_id]['type']=="passage":
                text_id_to_node_name[node_id] = self.kg.nodes[node_id]["id"]
            else:
                node_id_to_file_id[node_id] = self.kg.nodes[node_id]["file_id"]
        self.node_id_to_file_id = node_id_to_file_id        # node_id -> text id
        self.text_id_to_node_name = text_id_to_node_name    # text_id -> text string
        # filter out passage nodes - create a new mutable graph instead of a frozen view
        self.kg = DiGraph(self.kg.subgraph(self.node_list))
        self.data["KG"] = self.kg
        self.node_id_to_attr_id = {self.kg.nodes[n]['id']: n for n in self.kg.nodes}
        self.qa_judge = QAJudger()
        self.entity_to_id = {}
        # Initialize ID mapping tables BEFORE building the retriever so retrieve()
        # can translate faiss_id -> edge_list index from the first call.
        # If mapping doesn't exist, create identity mapping (initial state: faiss_id == list_index)
        if "edge_faiss_id_to_list_idx" not in self.data:
            self.data["edge_faiss_id_to_list_idx"] = {i: i for i in range(len(self.edge_list))}
        if "node_faiss_id_to_list_idx" not in self.data:
            self.data["node_faiss_id_to_list_idx"] = {i: i for i in range(len(self.node_list))}
        if "text_faiss_id_to_list_idx" not in self.data and "text_dict" in self.data:
            self.data["text_faiss_id_to_list_idx"] = {i: i for i in range(len(self.data["text_dict"]))}
        self.edge_faiss_id_to_list_idx = self.data["edge_faiss_id_to_list_idx"]
        self.node_faiss_id_to_list_idx = self.data["node_faiss_id_to_list_idx"]
        if "text_faiss_id_to_list_idx" in self.data:
            self.text_faiss_id_to_list_idx = self.data["text_faiss_id_to_list_idx"]
        else:
            self.text_faiss_id_to_list_idx = {}
        # Build retriever after mapping + filtered KG are ready; share the same
        # mapping dict object so insert/delete mutations are visible immediately.
        self.retriever = IncrementalGraphRetriever(
                            llm_generator=self.llm_generator,
                            sentence_encoder=self.sentence_encoder,
                            data=self.data,
                        )
        # self.retriever = HippoRAGRetriever(
        #             llm_generator=self.llm_generator,
        #             sentence_encoder=self.sentence_encoder,
        #             data=data,
        #         )
        # Per-query edge score cache lives on self._tls (see properties below).

    def _max_triples_at_step(self, step: int) -> int:
        """step is 1-indexed (matches refine loop)."""
        if self._max_triple_num_by_step:
            i = max(0, step - 1)
            seq = self._max_triple_num_by_step
            if i >= len(seq):
                return seq[-1]
            return seq[i]
        return self.max_triple_num

    # ------------------------------------------------------------------
    # Main interface for external use
    # ------------------------------------------------------------------
    @staticmethod
    def format_original_chunk(original_chunk: Any) -> str:
        """Normalize HotpotQA / MuSiQue / plain context into action-prompt text (training-aligned)."""
        if original_chunk is None:
            return ""
        if isinstance(original_chunk, str):
            return original_chunk.strip()
        if isinstance(original_chunk, (list, tuple)):
            # HotpotQA / 2Wiki: [[title, [sent, ...]], ...]
            if (
                original_chunk
                and isinstance(original_chunk[0], (list, tuple))
                and len(original_chunk[0]) >= 2
                and isinstance(original_chunk[0][1], list)
            ):
                parts: List[str] = []
                for item in original_chunk:
                    if not isinstance(item, (list, tuple)) or len(item) < 2:
                        continue
                    paragraphs = item[1]
                    if not isinstance(paragraphs, list):
                        continue
                    for p in paragraphs:
                        if isinstance(p, str) and p.strip():
                            parts.append(p.strip())
                        elif p is not None and str(p).strip():
                            parts.append(str(p).strip())
                return "\n\n".join(parts)
            # MuSiQue: [{"paragraph_text": ...}, ...]
            if original_chunk and isinstance(original_chunk[0], dict) and "paragraph_text" in original_chunk[0]:
                parts = []
                for item in original_chunk:
                    if isinstance(item, dict):
                        text = item.get("paragraph_text")
                        if isinstance(text, str) and text.strip():
                            parts.append(text.strip())
                return "\n\n".join(parts)
            return "\n\n".join(str(x) for x in original_chunk if str(x).strip())
        return str(original_chunk).strip()

    # ---- Per-thread query-scoped state (safe for parallel propose) ----
    @property
    def _current_query(self) -> str:
        return getattr(self._tls, "current_query", "")

    @_current_query.setter
    def _current_query(self, value: str) -> None:
        self._tls.current_query = value or ""

    @property
    def _edge_score_cache_query(self) -> Optional[str]:
        return getattr(self._tls, "edge_score_cache_query", None)

    @_edge_score_cache_query.setter
    def _edge_score_cache_query(self, value: Optional[str]) -> None:
        self._tls.edge_score_cache_query = value

    @property
    def _edge_score_cache_values(self) -> Optional[np.ndarray]:
        return getattr(self._tls, "edge_score_cache_values", None)

    @_edge_score_cache_values.setter
    def _edge_score_cache_values(self, value: Optional[np.ndarray]) -> None:
        self._tls.edge_score_cache_values = value

    @property
    def _edge_score_cache_query_emb(self) -> Optional[np.ndarray]:
        return getattr(self._tls, "edge_score_cache_query_emb", None)

    @_edge_score_cache_query_emb.setter
    def _edge_score_cache_query_emb(self, value: Optional[np.ndarray]) -> None:
        self._tls.edge_score_cache_query_emb = value

    def refine(
        self,
        query: str,
        original_chunk: Any = None,
        apply_actions: bool = True,
    ) -> Tuple[str, nx.DiGraph, Optional[RefinementResult]]:
        """
        Run the entire REAfiner process for a single query.

        Aligned with training RefinementInteraction (Search-R1 continuous trajectory):
        - Judgement rounds with monotonic working-context expansion
          (caps from ``max_triple_num_by_step``, default [10, 50, 90]).
        - Later judgements only receive *newly added* triples
          (``REAFINER_EXPANDED_JUDGEMENT_USER_PROMPT``).
        - Abduction / Action continue the same message history; phase system
          prompts are injected into the user turn (same as training rollout).
        - ``original_chunk`` (training ``interaction_kwargs.original_chunk``) is preferred
          for the action prompt; falls back to passages reconstructed from the subgraph.
        - Final RAG / answer generation is optional (``if_gen_answer``); training
          keeps RAG reward-only outside the policy trajectory.
        - ``apply_actions``: if False, only propose refinement callables (read-only on KG)
          so callers can parallelize propose then apply serially.

        Returns:
        -------
        - answer:             Answer given by LLM on the final refined KG (possibly abstract natural language).
        - refined_kg:         KG after inserting new knowledge (in-place modification, also returned by reference).
        - refinement_result:  Refinement result containing interaction history, original subgraph, and refined subgraph.
        """
        interaction_history: List[RetrievalStepResult] = []
        final_answer: str = ""
        short_answer: Optional[str] = None
        base_top_k = self.base_top_k
        chunk_text = self.format_original_chunk(original_chunk)
        self._current_query = query or ""
        # Prime similarity cache once for this query; later pruning can directly reuse scores.
        self._get_or_compute_edge_scores(query)

        # Continuous conversation (Search-R1 style), matching training rollout.
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": REAFINER_JUDGEMENT_SYSTEM_PROMPT.strip()}
        ]
        sorted_context: List[str] = []
        answerable = False
        judgement_raw: Optional[str] = None

        for step in range(1, self.max_hops + 1):
            print(f"\033[94m [Step: {step}] \033[0m")
            previous_context = set(sorted_context)

            if step == 1:
                # First round: retrieve up to schedule[0] (aligned with training cap0).
                if self._max_triple_num_by_step:
                    topn1 = min(base_top_k, self._max_triples_at_step(1))
                else:
                    topn1 = base_top_k
                sorted_context, _ = self.retriever.retrieve(query, topN=topn1)
                sorted_context = self._sanitize_context_list(sorted_context)
                user_content = REAFINER_JUDGEMENT_USER_PROMPT.format(
                    question=query,
                    triples_string="\n".join(sorted_context) if sorted_context else "(no triples)",
                ).strip()
            else:
                # Monotonic expand: keep prior triples, add local then global candidates up to cap.
                cap = self._max_triples_at_step(step)
                sorted_context, added = self._expand_sorted_context_monotonic(
                    query=query,
                    sorted_context=sorted_context,
                    cap=cap,
                )
                sorted_context = self._sanitize_context_list(sorted_context)
                added = self._sanitize_context_list(added)
                if not added and len(sorted_context) <= len(previous_context):
                    # Nothing new to show; stop expanding and move to abduction/action.
                    print(
                        f"\033[93m [Step {step}] no new triples under cap={cap}; "
                        f"stop judgement expansion \033[0m"
                    )
                    break
                triples_string = "\n".join(added) if added else "(no new triples)"
                user_content = REAFINER_EXPANDED_JUDGEMENT_USER_PROMPT.format(
                    triples_string=triples_string
                ).strip()

            retrieved_subgraph = self._sorted_context_to_subgraph(sorted_context)

            answerable, judgement_raw = self._answerable_judgement_messages(
                messages, user_content
            )
            if judgement_raw is None:
                interaction_history.append(
                    RetrievalStepResult(
                        num_hops=(step - 1) * self.increament_hop,
                        base_top_k=base_top_k,
                        query=query,
                        retrieved_subgraph=retrieved_subgraph,
                        raw_response=None,
                        answerable=bool(answerable),
                        answer=None,
                    )
                )
                refinement_result = RefinementResult(
                    query=query,
                    history_horizon_size=self.history_horizon_size,
                    interaction_history=interaction_history,
                    error_abduction_reason=None,
                    original_subgraph=retrieved_subgraph,
                    refined_subgraph=None,
                    refinement_action_list=[],
                    refinement_action_raw=None,
                )
                return (None, self.data, refinement_result)

            if self.if_gen_answer:
                final_answer = self._generate_answer(
                    query, "\n".join(sorted_context)
                )
                short_answer = self.qa_judge.split_answer(final_answer)

            interaction_history.append(
                RetrievalStepResult(
                    num_hops=(step - 1) * self.increament_hop,
                    base_top_k=base_top_k,
                    query=query,
                    retrieved_subgraph=retrieved_subgraph,
                    raw_response=judgement_raw,
                    answerable=bool(answerable),
                    answer=short_answer if self.if_gen_answer else None,
                )
            )

            # Answerable: stop expanding (hop1 Yes skips action below; hop>1 still
            # abducts in training, but offline may skip via skip_action_if_answerable).
            if answerable:
                break
            # Not answerable: continue expanding while hops remain.

        # Training: skip abduction/action only when the *first* judgement is Yes.
        last_answerable = bool(interaction_history and interaction_history[-1].answerable)
        if (
            len(interaction_history) == 1
            and interaction_history[0].answerable
        ) or (self.skip_action_if_answerable and last_answerable):
            refinement_result = RefinementResult(
                query=query,
                history_horizon_size=self.history_horizon_size,
                interaction_history=interaction_history,
                error_abduction_reason=None,
                original_subgraph=interaction_history[-1].retrieved_subgraph,
                refined_subgraph=None,
                refinement_action_list=[],
                refinement_action_raw=None,
            )
            return (interaction_history[-1].answer, self.data, refinement_result)

        if not interaction_history:
            refinement_result = RefinementResult(
                query=query,
                history_horizon_size=self.history_horizon_size,
                interaction_history=interaction_history,
                error_abduction_reason=None,
                original_subgraph=None,
                refined_subgraph=None,
                refinement_action_list=[],
                refinement_action_raw=None,
            )
            return (None, self.data, refinement_result)

        # ---- Abduction (continue the same trajectory) ----
        error_abduction_reason, error_abduction_raw = self._error_abduction_messages(
            messages
        )
        if error_abduction_reason is None:
            refinement_result = RefinementResult(
                query=query,
                history_horizon_size=self.history_horizon_size,
                interaction_history=interaction_history,
                error_abduction_reason=error_abduction_reason,
                original_subgraph=interaction_history[-1].retrieved_subgraph,
                refined_subgraph=None,
                refinement_action_list=[],
                refinement_action_raw=None,
            )
            return (interaction_history[-1].answer, self.data, refinement_result)

        # ---- Action generation (continue the same trajectory) ----
        # Prefer sample context (training original_chunk); fallback to reconstructed passages.
        original_text = chunk_text or self._collect_original_text(
            interaction_history[-1].retrieved_subgraph
        )
        refinement_action_list, refinement_action_raw = self._kg_refinement_action_messages(
            messages,
            original_text=original_text,
            query=query,
            triples=interaction_history[-1].retrieved_subgraph,
            error_abduction_reason=error_abduction_reason,
        )
        if refinement_action_raw is None:
            refinement_result = RefinementResult(
                query=query,
                history_horizon_size=self.history_horizon_size,
                interaction_history=interaction_history,
                error_abduction_reason=error_abduction_reason,
                original_subgraph=interaction_history[-1].retrieved_subgraph,
                refined_subgraph=None,
                refinement_action_list=refinement_action_list
                if isinstance(refinement_action_list, list)
                else [],
                refinement_action_raw=refinement_action_raw,
            )
            return (interaction_history[-1].answer, self.data, refinement_result)

        for action in refinement_action_list:
            if apply_actions:
                action()

        refinement_result = RefinementResult(
            query=query,
            history_horizon_size=self.history_horizon_size,
            interaction_history=interaction_history,
            error_abduction_reason=error_abduction_reason,
            original_subgraph=interaction_history[-1].retrieved_subgraph,
            refined_subgraph=None,
            refinement_action_list=refinement_action_list,
            refinement_action_raw=refinement_action_raw,
        )
        return (interaction_history[-1].answer, self.data, refinement_result)

    @staticmethod
    def _sorted_context_to_subgraph(sorted_context: List[str]) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        for x in sorted_context:
            parts = x.split("  ")
            if len(parts) != 3:
                continue
            out.append(
                {"subject": parts[0], "relation": parts[1], "object": parts[2]}
            )
        return out

    def _expand_sorted_context_monotonic(
        self,
        query: str,
        sorted_context: List[str],
        cap: int,
    ) -> Tuple[List[str], List[str]]:
        """
        Training-aligned expansion: keep current triples, then add local 1-hop
        and global candidates ranked by query similarity up to ``cap``.

        Returns (new_sorted_context, newly_added_triples_only).
        """
        current: List[Tuple[str, str, str]] = []
        for triple_str in sorted_context or []:
            cleaned = self._sanitize_named_triple_str(triple_str)
            if cleaned is not None:
                current.append(cleaned)

        if not current or len(current) >= cap:
            cleaned_ctx = [f"{s}  {r}  {o}" for s, r, o in current]
            return cleaned_ctx, []

        id_to_node = {
            str(self.kg.nodes[n].get("id", "")): n
            for n in self.kg.nodes()
            if self.kg.nodes[n].get("id") is not None
            and not self._looks_like_internal_id(self.kg.nodes[n].get("id"))
        }
        # Also allow resolving by internal key when a prior bug leaked hashes into context.
        for n in self.kg.nodes():
            id_to_node.setdefault(str(n), n)

        initial_nodes = {
            id_to_node[entity]
            for s, _, o in current
            for entity in (s, o)
            if entity in id_to_node
        }

        local_triples: List[Tuple[str, str, str]] = []
        if initial_nodes:
            local_graph = self._construct_subgraph(
                list(initial_nodes), num_hop=self.increament_hop
            )
            # Always resolve display names from the full KG. Subgraph node keys are
            # internal sha256 ids; using local_graph.nodes[u].get("id", u) without
            # copied attrs used to leak hashes into the LLM prompt.
            local_triples = []
            for u, v, d in local_graph.edges(data=True):
                named = self._named_triple_from_edge(u, v, d.get("relation", ""))
                if named is not None:
                    local_triples.append(named)

        global_triples = []
        for u, v, d in self.kg.edges(data=True):
            named = self._named_triple_from_edge(u, v, d.get("relation", ""))
            if named is not None:
                global_triples.append(named)

        selected: List[Tuple[str, str, str]] = list(dict.fromkeys(current))
        selected_set = set(selected)
        for candidates in (local_triples, global_triples):
            remaining = cap - len(selected)
            if remaining <= 0:
                break
            unseen = [t for t in candidates if t not in selected_set]
            if not unseen:
                continue
            # Reuse embedding prune scorer: convert to "s  r  o" then back.
            ranked_strs = self._prune_subgraph_embd(
                unseen, query, max_triple_cap=remaining
            )
            ranked = []
            for s in ranked_strs:
                cleaned = self._sanitize_named_triple_str(s)
                if cleaned is not None:
                    ranked.append(cleaned)
            selected.extend(ranked)
            selected_set.update(ranked)

        new_context = [f"{s}  {r}  {o}" for s, r, o in selected]
        prior = set(self._sanitize_context_list(sorted_context or []))
        added = [t for t in new_context if t not in prior]
        return new_context, added

    def _answerable_judgement_messages(
        self,
        messages: List[Dict[str, str]],
        user_content: str,
    ) -> Tuple[bool, Optional[str]]:
        """Append user turn, call LLM, append assistant; mutate ``messages`` in place."""
        messages.append({"role": "user", "content": user_content})
        try:
            raw = self.llm_generator.generate_response(
                messages, temperature=0.0, max_new_tokens=256
            )
        except Exception as e:
            error_message = {"error": f"Answerable Judgement Generation Error: {e}"}
            print(error_message)
            return False, None

        print(raw)
        messages.append({"role": "assistant", "content": raw if isinstance(raw, str) else str(raw)})

        judge_match = re.search(r"<judge>(.*?)</judge>", raw, re.IGNORECASE | re.DOTALL)
        if judge_match:
            answerable = judge_match.group(1).strip().lower().startswith("yes")
        else:
            text_lower = (raw or "").lower()
            window = text_lower[:200]
            if "yes" in window and "no" not in window:
                answerable = True
            elif "no" in window and "yes" not in window:
                answerable = False
            else:
                print([{"error": f"Answerable Judgement Error Format: {raw}"}])
                return False, None
        return answerable, raw

    def _error_abduction_messages(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Continue trajectory: inject abduction phase instruction into user turn."""
        user_content = (
            f"{REAFINER_ERROR_ABDUCTION_SYSTEM_PROMPT.strip()}\n\n"
            f"{REAFINER_ABDUCTION_USER_PROMPT.strip()}"
        )
        messages.append({"role": "user", "content": user_content})
        try:
            raw = self.llm_generator.generate_response(messages, temperature=0.0)
        except Exception as e:
            print([{"error": f"Abduction Generation Error: {e}"}])
            return None, None

        print(raw)
        messages.append({"role": "assistant", "content": raw if isinstance(raw, str) else str(raw)})

        abduction_match = re.search(
            r"<abduction>(.*?)</abduction>", raw, re.IGNORECASE | re.DOTALL
        )
        if abduction_match:
            return abduction_match.group(1).strip(), raw
        # Lenient fallback (same as training generate_response_refinement_simple)
        return (raw or "").strip(), raw

    def _collect_original_text(self, triples: List[Dict[str, str]]) -> str:
        text_set = set()
        for triple in triples or []:
            if not isinstance(triple, dict):
                continue
            sub = triple.get("subject", "")
            obj = triple.get("object", "")
            sub_id = self._get_node_id(sub, self.entity_to_id)
            obj_id = self._get_node_id(obj, self.entity_to_id)
            sub_file_id = self.node_id_to_file_id.get(sub_id)
            if sub_file_id is not None and sub_file_id in self.text_id_to_node_name:
                text_set.add(self.text_id_to_node_name[sub_file_id])
            obj_file_id = self.node_id_to_file_id.get(obj_id)
            if obj_file_id is not None and obj_file_id in self.text_id_to_node_name:
                text_set.add(self.text_id_to_node_name[obj_file_id])
        return "\n\n".join(list(text_set)[:50])

    _FORMAT_RETRY_PROMPT = (
        "Your previous output could not be parsed. "
        "You MUST wrap your refinement actions inside <refinement>...</refinement> tags. "
        "Use ONLY the following functions, separated by |:\n"
        '  insert_edge("subject", "relation", "object")\n'
        '  delete_edge("subject", "object")\n'
        '  replace_node("old_entity", "new_entity")\n'
        "Example:\n"
        "<refinement>"
        'insert_edge("Albert Einstein", "born_in", "Ulm") | '
        'delete_edge("Albert Einstein", "Berlin")'
        "</refinement>\n"
        "Try again. Output ONLY the <refinement>...</refinement> block."
    )

    def _entity_grounded(self, name: str, original_text_l: str, subgraph_ents: Set[str]) -> bool:
        if not name:
            return False
        n = name.strip()
        if not n:
            return False
        n_l = n.lower()
        if original_text_l and n_l in original_text_l:
            return True
        # Soft match: entity appears as a token-ish substring in subgraph entity names.
        for e in subgraph_ents:
            e_l = e.lower()
            if n_l == e_l or (len(n_l) >= 3 and (n_l in e_l or e_l in n_l)):
                return True
        return False

    def _has_relation_conflict(self, sub: str, rel: str, obj: str) -> bool:
        """True if KG already has sub --rel--> other_obj with a different object."""
        sub_id = self._get_node_id(sub, self.entity_to_id)
        if sub_id not in self.kg:
            return False
        rel_n = self._safe_sanitize(rel).strip().lower()
        obj_n = self._safe_sanitize(obj).strip().lower()
        for _, dst, data in self.kg.out_edges(sub_id, data=True):
            existing_rel = str(data.get("relation", "")).strip().lower()
            if existing_rel != rel_n:
                continue
            existing_obj = str(self.kg.nodes[dst].get("id", dst)).strip().lower()
            if existing_obj and existing_obj != obj_n:
                return True
        return False

    def _filter_parsed_actions(
        self,
        parsed: List[Tuple[str, List[str]]],
        *,
        original_text: str,
        triples: List[Dict[str, str]],
    ) -> List[Callable[[], None]]:
        """Apply offline anti-pollution filters; return callables ready to execute."""
        original_text_l = (original_text or "").lower()
        subgraph_ents: Set[str] = set()
        for t in triples or []:
            if isinstance(t, dict):
                if t.get("subject"):
                    subgraph_ents.add(str(t["subject"]))
                if t.get("object"):
                    subgraph_ents.add(str(t["object"]))

        out: List[Callable[[], None]] = []
        for function_name, args in parsed:
            try:
                if function_name == "insert_edge":
                    if len(args) < 3:
                        continue
                    sub, rel, obj = args[0], args[1], args[2]
                    sub = self._resolve_entity_arg(sub)
                    obj = self._resolve_entity_arg(obj)
                    if sub is None or obj is None:
                        print(f"Skip insert_edge with unresolved internal id args: {args}")
                        continue
                    if self.skip_generic_objects and str(obj).strip().lower() in self._GENERIC_OBJECTS:
                        print(f"Skip insert_edge with generic object: {sub} | {rel} | {obj}")
                        continue
                    if self.ground_inserts and original_text_l:
                        # Require subject/object grounded in source text OR retrieved subgraph.
                        sub_ok = self._entity_grounded(sub, original_text_l, subgraph_ents)
                        obj_ok = self._entity_grounded(obj, original_text_l, subgraph_ents)
                        need_both = self.ground_mode != "either"
                        if need_both and not (sub_ok and obj_ok):
                            print(
                                f"Skip ungrounded insert_edge: {sub} | {rel} | {obj} "
                                f"(sub_ok={sub_ok}, obj_ok={obj_ok}, mode={self.ground_mode})"
                            )
                            continue
                        if (not need_both) and not (sub_ok or obj_ok):
                            print(
                                f"Skip ungrounded insert_edge: {sub} | {rel} | {obj} "
                                f"(sub_ok={sub_ok}, obj_ok={obj_ok}, mode={self.ground_mode})"
                            )
                            continue
                    if self.require_query_overlap and self._current_query:
                        q_l = self._current_query.lower()
                        if not (
                            self._entity_grounded(sub, q_l, set())
                            or self._entity_grounded(obj, q_l, set())
                        ):
                            print(
                                f"Skip insert_edge without query overlap: {sub} | {rel} | {obj}"
                            )
                            continue
                    if self.skip_conflict_inserts and self._has_relation_conflict(sub, rel, obj):
                        print(f"Skip conflicting insert_edge: {sub} | {rel} | {obj}")
                        continue
                    out.append(lambda s=sub, r=rel, o=obj: self._insert_edge(s, r, o))
                elif function_name == "delete_edge":
                    if len(args) < 2:
                        continue
                    sub, obj = args[0], args[-1]
                    sub = self._resolve_entity_arg(sub)
                    obj = self._resolve_entity_arg(obj)
                    if sub is None or obj is None:
                        print(f"Skip delete_edge with unresolved internal id args: {args}")
                        continue
                    out.append(lambda s=sub, o=obj: self._delete_edge(s, o))
                elif function_name == "replace_node":
                    if len(args) < 2:
                        continue
                    old_ent, new_ent = args[0], args[1]
                    old_ent = self._resolve_entity_arg(old_ent)
                    new_ent = self._resolve_entity_arg(new_ent)
                    if old_ent is None or new_ent is None:
                        print(f"Skip replace_node with unresolved internal id args: {args}")
                        continue
                    if self.ground_inserts and original_text_l:
                        if not self._entity_grounded(new_ent, original_text_l, subgraph_ents):
                            print(f"Skip ungrounded replace_node: {old_ent} -> {new_ent}")
                            continue
                    out.append(lambda old=old_ent, new=new_ent: self._replace_node(old, new))
                else:
                    print(f"Error: Unknown action format: {function_name}")
            except Exception as e:
                print([{"error": f"KG Refinement Action Filter Error: {args}, Error: {e}"}])
            if len(out) >= self.max_actions:
                break
        return out

    def _parse_refinement_actions_str(self, raw: str) -> Tuple[Optional[str], bool]:
        """Return (actions_str, parse_ok). Empty refinement is OK; unparseable junk is not."""
        if raw is None:
            return None, False
        if raw.count("delete_edge") > 20 or raw.count("insert_edge") > 30:
            print([{"error": "clip"}])
            return None, False
        refinement_match = re.search(
            r"<refinement>(.*?)</refinement>", raw, re.IGNORECASE | re.DOTALL
        )
        if refinement_match:
            return refinement_match.group(1).strip().strip("|"), True
        if re.search(r"<refinement\s*/>", raw, re.IGNORECASE):
            return "", True
        calls = re.findall(
            r"(?:insert_edge|delete_edge|replace_node)\s*\(.*?\)",
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if calls:
            return "|".join(calls), True
        # Training-tolerant: empty / prose without actions -> treat as no-op success
        if not re.search(r"(?:insert_edge|delete_edge|replace_node)\s*\(", raw, re.IGNORECASE):
            return "", True
        return None, False

    def _kg_refinement_action_messages(
        self,
        messages: List[Dict[str, str]],
        *,
        original_text: str,
        query: str,
        triples: List[Dict[str, str]],
        error_abduction_reason: str,
    ) -> Tuple[Any, Optional[str]]:
        """Continue trajectory: match training RefinementInteraction action turn.

        Training injects ACTION system prompt as environment text, then
        ``REAFINER_ACTION_USER_PROMPT`` only (KG/question/abduction already in
        the prior multi-turn history). Do **not** re-dump legacy single-shot
        fields (triples/question/error_reasons) — that breaks true multi-turn.
        """
        del query, error_abduction_reason  # available in conversation history
        continuous_user = REAFINER_ACTION_USER_PROMPT.format(
            original_text=original_text
        ).strip()
        user_content = (
            f"{REAFINER_KG_REFINEMENT_ACTION_SYSTEM_PROMPT.strip()}\n\n"
            f"{continuous_user}"
        )
        messages.append({"role": "user", "content": user_content})

        last_raw: Optional[str] = None
        for attempt in range(self.max_format_retry + 1):
            try:
                raw = self.llm_generator.generate_response(
                    messages, temperature=0.0, max_new_tokens=2048
                )
                print(raw)
            except Exception as e:
                print([{"error": f"KG Refinement Action Generation Error: {e}"}])
                return [], None

            last_raw = raw if isinstance(raw, str) else str(raw)
            messages.append({"role": "assistant", "content": last_raw})

            actions_str, parse_ok = self._parse_refinement_actions_str(last_raw)
            if not parse_ok:
                if attempt < self.max_format_retry:
                    print(
                        f"\033[93m [Format retry {attempt + 1}/{self.max_format_retry}] \033[0m"
                    )
                    messages.append({"role": "user", "content": self._FORMAT_RETRY_PROMPT})
                    continue
                print([{"error": f"KG Refinement Error Format: {last_raw}"}])
                return [], last_raw

            parsed: List[Tuple[str, List[str]]] = []
            hard_fail = False
            for action in re.split(r"[\|\n]+", actions_str or ""):
                action = action.strip()
                if not action:
                    continue
                try:
                    function_name, args = self._parse_action_string(action)
                    parsed.append((function_name, args))
                except Exception as e:
                    print(
                        [
                            {
                                "error": (
                                    f"KG Refinement Action Error Format: {action}, "
                                    f"Error: {str(e)}"
                                )
                            }
                        ]
                    )
                    hard_fail = True
                    break
            if hard_fail:
                if attempt < self.max_format_retry:
                    print(
                        f"\033[93m [Format retry {attempt + 1}/{self.max_format_retry}] \033[0m"
                    )
                    messages.append({"role": "user", "content": self._FORMAT_RETRY_PROMPT})
                    continue
                return [], last_raw

            refinement_action_list = self._filter_parsed_actions(
                parsed, original_text=original_text, triples=triples
            )
            return refinement_action_list, last_raw

        return [], last_raw

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        if torch is not None:
            try:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Retrieval related
    # ------------------------------------------------------------------
    def _encode_query(self, query: str) -> np.ndarray:
        emb = self.sentence_encoder.encode([query], normalize_embeddings=False)[0]
        emb = np.asarray(emb, dtype="float32").reshape(1, -1)
        # Some encoders may return vectors of different dimensions, here we truncate / pad to match the index dimension
        if emb.shape[1] != self._dim:
            if emb.shape[1] > self._dim:
                emb = emb[:, : self._dim]
            else:
                padded = np.zeros((1, self._dim), dtype="float32")
                padded[:, : emb.shape[1]] = emb
                emb = padded
        faiss.normalize_L2(emb)
        return emb
    
    def _prune_subgraph_llm(self, subgraph_triples: List[str], query: str) -> List[str]:
        """
        Prune the subgraph based on the LLM judgement.
        """
        system_prompt = REAFINER_FILTERING_SYSTEM_PROMPT
        user_prompt = REAFINER_FILTERING_USER_PROMPT.format(
            triples_string=subgraph_triples,
            query=query,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            raw = self.llm_generator.generate_response(
                messages, temperature=0.0, max_new_tokens=8192
            )
        except Exception as e:
            # fallback
            error_message = {"error": f"Prune Subgraph LLM Error: {e}"}
            print(error_message)
            return error_message['error'], None
        # Parse the output: extract <filtering> tag
        filtering_match = re.search(r'<filtering>(.*?)</filtering>', raw, re.IGNORECASE | re.DOTALL)
        if filtering_match:
            filtering_text = filtering_match.group(1).strip()
            try:
                filtering_triples = json.loads(filtering_text)
            except json.JSONDecodeError:
                try:
                    refined_triples = json_repair.loads(filtering_text)
                except Exception:
                    # fallback
                    error_message = {"error": f"Filtering Subgraph LLM Error Format: {raw}"}
                    print(error_message)
                    return error_message['error'], None
        else:
            if '<filtering>' in raw:
                filtering_text = raw.split('<filtering>')[1].split('</filtering>')[0].strip()
            else:
                # fallback
                error_message = {"error": f"Filtering Subgraph LLM Error Format: {raw}"}
                print(error_message)
                return error_message['error'], None
            try:
                filtering_triples = json.loads(filtering_text)
            except json.JSONDecodeError:
                # if fails, try json_repair to fix common JSON issues (like invalid escape sequences)
                try:
                    filtering_triples = json_repair.loads(filtering_text)
                except Exception:
                    # fallback
                    error_message = {"error": f"Filtering Subgraph LLM Error Format: {raw}"}
                    print(error_message)
                    return error_message['error'], None
        pruned_subgraph_triples = [f"{triple['subject']}  {triple['relation']}  {triple['object']}" for triple in filtering_triples]
        return pruned_subgraph_triples, raw
    
    def _compute_query_edge_scores(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute query-to-edge similarity for all current edges using precomputed edge embeddings.
        This avoids re-encoding subgraph edge strings during pruning.
        """
        query_embedding = self.sentence_encoder.encode([query], query_type='edge')
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()
        query_emb = np.array(query_embedding[0], dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query_emb)

        edge_embeddings = np.array(self.edge_embeddings, dtype=np.float32)
        if edge_embeddings.ndim == 1:
            edge_embeddings = edge_embeddings.reshape(1, -1)
        faiss.normalize_L2(edge_embeddings)
        similarities = edge_embeddings @ query_emb.T
        return similarities.flatten(), query_emb

    def _get_or_compute_edge_scores(self, query: str) -> np.ndarray:
        """
        Return cached edge scores for the current query when possible.
        Recompute only when query changes or edge count changed unexpectedly.
        """
        cache_valid = (
            self._edge_score_cache_query == query
            and self._edge_score_cache_values is not None
            and len(self._edge_score_cache_values) == len(self.edge_list)
        )
        if cache_valid:
            return self._edge_score_cache_values
        scores, query_emb = self._compute_query_edge_scores(query)
        self._edge_score_cache_query = query
        self._edge_score_cache_values = scores
        self._edge_score_cache_query_emb = query_emb
        return scores

    def _append_cached_edge_score(self, new_edge_embedding: np.ndarray) -> None:
        """
        Incrementally extend cached scores when a new edge is inserted during the same query.
        """
        if (
            self._edge_score_cache_values is None
            or self._edge_score_cache_query_emb is None
            or self._edge_score_cache_query is None
        ):
            return
        emb = np.array(new_edge_embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(emb)
        new_score = float((emb @ self._edge_score_cache_query_emb.T).reshape(-1)[0])
        self._edge_score_cache_values = np.append(self._edge_score_cache_values, new_score)

    def _invalidate_edge_score_cache(self) -> None:
        self._edge_score_cache_query = None
        self._edge_score_cache_values = None
        self._edge_score_cache_query_emb = None

    def _prune_subgraph_embd(
        self,
        subgraph_triples: List[str],
        query: str,
        max_triple_cap: Optional[int] = None,
    ) -> List[str]:
        """
        Prune the subgraph embedding similarity.
        """
        cap = self.max_triple_num if max_triple_cap is None else max_triple_cap
        # Score all edges once, then project subgraph triples to global edge scores.
        all_edge_scores = self._get_or_compute_edge_scores(query)
        edge_to_list_idx = {edge: idx for idx, edge in enumerate(self.edge_list)}

        scored_subgraph = []
        for s, r, o in subgraph_triples:
            s_node_id = self.node_id_to_attr_id.get(s, s)
            o_node_id = self.node_id_to_attr_id.get(o, o)
            list_idx = edge_to_list_idx.get((s_node_id, o_node_id), None)
            if list_idx is None or list_idx >= len(all_edge_scores):
                # Missing mapping is rare (typically after aggressive graph mutations); keep it as low priority.
                score = -np.inf
            else:
                score = float(all_edge_scores[list_idx])
            scored_subgraph.append((score, s, r, o))

        scored_subgraph.sort(key=lambda x: x[0], reverse=True)
        top_scored = scored_subgraph[:cap]
        pruned_subgraph_triples = [f"{s}  {r}  {o}" for _, s, r, o in top_scored]
        return pruned_subgraph_triples

    def _collect_query_candidate_triples(
        self,
        query: str,
        retrieve_topk: int = 50,
        one_hop_sample_size: int = 100,
        use_score_cache: bool = True,
    ) -> Set[str]:
        """
        Build candidate triple-string set for coverage selection:
        1) top-k retrieval triples
        2) 1-hop neighbor sampled triples around retrieved nodes
        """
        retrieved_context, _ = self.retriever.retrieve(query, topN=retrieve_topk)
        candidate = set(self._sanitize_context_list(retrieved_context))
        if one_hop_sample_size <= 0 or not candidate:
            return candidate

        node_str_list = []
        for triple_str in candidate:
            parts = triple_str.split("  ")
            if len(parts) != 3:
                continue
            s, _, o = parts
            node_str_list.extend([s, o])
        node_str_list = sorted(set(node_str_list))
        node_id_list = [self.node_id_to_attr_id.get(node_str, node_str) for node_str in node_str_list]
        subgraph = self._construct_subgraph(node_id_list, num_hop=1)
        subgraph_triples = sorted(
            {
                named
                for u, v, d in subgraph.edges(data=True)
                for named in [self._named_triple_from_edge(u, v, d.get("relation", ""))]
                if named is not None
            }
        )
        if not subgraph_triples:
            return candidate

        # Rank 1-hop triples by query relevance and keep a bounded sample.
        if use_score_cache:
            all_edge_scores = self._get_or_compute_edge_scores(query)
        else:
            all_edge_scores, _ = self._compute_query_edge_scores(query)
        edge_to_list_idx = {edge: idx for idx, edge in enumerate(self.edge_list)}
        scored_subgraph = []
        for s, r, o in subgraph_triples:
            s_node_id = self.node_id_to_attr_id.get(s, s)
            o_node_id = self.node_id_to_attr_id.get(o, o)
            list_idx = edge_to_list_idx.get((s_node_id, o_node_id), None)
            score = float(all_edge_scores[list_idx]) if list_idx is not None and list_idx < len(all_edge_scores) else -np.inf
            scored_subgraph.append((score, s, r, o))
        scored_subgraph.sort(key=lambda x: x[0], reverse=True)
        for _, s, r, o in scored_subgraph[:one_hop_sample_size]:
            candidate.add(f"{s}  {r}  {o}")
        return set(self._sanitize_context_list(list(candidate)))

    def select_refine_subset(
        self,
        query_data: List[Dict[str, Any]],
        max_subset_size: int,
        target_coverage: float,
        retrieve_topk: int = 50,
        one_hop_sample_size: int = 100,
        selection_workers: int = 1,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Coverage-based query subset selection for KG refinement.
        """
        triple_sets: List[Set[str]] = [set() for _ in range(len(query_data))]
        if selection_workers <= 1:
            for idx, sample in enumerate(tqdm(query_data, desc="Selecting refine subset (retrieve+1hop)")):
                q = sample["question"]
                triple_sets[idx] = self._collect_query_candidate_triples(
                    query=q,
                    retrieve_topk=retrieve_topk,
                    one_hop_sample_size=one_hop_sample_size,
                    use_score_cache=True,
                )
        else:
            with ThreadPoolExecutor(max_workers=selection_workers) as executor:
                futures = []
                for idx, sample in enumerate(query_data):
                    q = sample["question"]
                    futures.append(
                        executor.submit(
                            self._collect_query_candidate_triples,
                            q,
                            retrieve_topk,
                            one_hop_sample_size,
                            False,  # avoid shared cache mutation across threads
                        )
                    )
                for idx, fut in enumerate(tqdm(futures, desc=f"Selecting refine subset (parallel x{selection_workers})")):
                    triple_sets[idx] = fut.result()

        universe = set().union(*triple_sets) if triple_sets else set()
        if not universe or max_subset_size <= 0:
            return [], {
                "selected_size": 0,
                "total_queries": len(query_data),
                "covered": 0,
                "universe": len(universe),
                "coverage_ratio": 0.0,
            }

        covered: Set[str] = set()
        remaining = set(range(len(triple_sets)))
        selected_idx: List[int] = []

        def cov_ratio() -> float:
            return (len(covered) / len(universe)) if universe else 1.0

        while remaining and len(selected_idx) < max_subset_size and cov_ratio() < target_coverage:
            best_i = None
            best_gain = -1
            for i in remaining:
                gain = len(triple_sets[i] - covered)
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
            if best_i is None or best_gain <= 0:
                break
            selected_idx.append(best_i)
            covered |= triple_sets[best_i]
            remaining.remove(best_i)

        selected_queries = [query_data[i] for i in selected_idx]
        stats = {
            "selected_size": len(selected_queries),
            "total_queries": len(query_data),
            "covered": len(covered),
            "universe": len(universe),
            "coverage_ratio": (len(covered) / len(universe)) if universe else 0.0,
        }
        return selected_queries, stats

    # ------------------------------------------------------------------
    # LLM interaction part
    # ------------------------------------------------------------------
    def _answerable_judgement(self, query: str, triples_string: str) -> Tuple[bool, str]:
        """
        Judge if the given question is answerable based on the provided KG context.
        
        Returns:
        -------
        - answerable: bool
        - raw_response: str
        """
        system_prompt = REAFINER_JUDGEMENT_SYSTEM_PROMPT
        user_prompt = REAFINER_JUDGEMENT_USER_PROMPT.format(
            question=query,
            triples_string=triples_string,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            raw = self.llm_generator.generate_response(
                messages, temperature=0.0, max_new_tokens=256
            )
        except Exception as e:
            # fallback
            error_message = {"error": f"Answerable Judgement Generation Error: {e}"}
            print(error_message)
            return error_message['error'], None

        print(raw)
        # Parse the output: extract <judge> tag
        judge_match = re.search(r'<judge>(.*?)</judge>', raw, re.IGNORECASE | re.DOTALL)
        if judge_match:
            judge_text = judge_match.group(1).strip().lower()
            answerable = judge_text.startswith("yes")
        else:
            # Fallback: try to find Yes/No in the text
            text_lower = raw.lower()
            if "yes" in text_lower[:100]:
                answerable = True
            elif "no" in text_lower[:100]:
                answerable = False
            else:
                error_message = [{"error": f"Answerable Judgement Error Format: {raw}"}]
                print(error_message)
                return error_message[0]['error'], None
        return answerable, raw
    
    def _error_abduction(self, interaction_history: List[RetrievalStepResult]) -> Tuple[str, str]:
        """
        Analyze the error reasons based on the given interaction history.
        """
        interaction_history_str = "\n".join(
            [f"Step{i+1}:\n['Query': {result.query}, 'Subgraph_hop': {result.num_hops}, 'Subgraph_content': {str(result.retrieved_subgraph)}, 'Answerable': {result.answerable}]\n" for i, result in enumerate(interaction_history[-self.history_horizon_size:])]
            )
        system_prompt = REAFINER_ERROR_ABDUCTION_SYSTEM_PROMPT
        user_prompt = REAFINER_ERROR_ABDUCTION_USER_PROMPT.format(
            interaction_history=interaction_history_str,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            raw = self.llm_generator.generate_response(
                messages, temperature=0.0
            )
        except Exception as e:
            # Hard failure: treat as abduction generation error
            error_message = [{"error": f"Abduction Generation Error: {e}"}]
            print(error_message)
            return error_message[0]["error"], None

        # Parse the output: prefer <abduction>...</abduction>, but be lenient:
        # if no well-formed tag, fall back to using the raw content (or the
        # inner text of an error field) as the error reason.
        abduction_match = re.search(r"<abduction>(.*?)</abduction>", raw, re.IGNORECASE | re.DOTALL)
        if abduction_match:
            reason = abduction_match.group(1).strip()
            return reason, raw

        # Fallback 1: if raw looks like JSON (list/dict), try to extract an "error" field
        raw_text = raw
        try:
            parsed = None
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = json_repair.loads(raw)
            if isinstance(parsed, list) and parsed:
                first = parsed[0]
                if isinstance(first, dict) and "error" in first:
                    raw_text = str(first["error"])
            elif isinstance(parsed, dict) and "error" in parsed:
                raw_text = str(parsed["error"])
        except Exception:
            # ignore JSON parsing errors and just use raw
            pass

        # Fallback 2: try to grab everything after a lone <abduction> tag,
        # even if the closing tag is malformed or missing.
        loose_match = re.search(r"<abduction>(.*)", raw_text, re.IGNORECASE | re.DOTALL)
        if loose_match:
            reason = loose_match.group(1).strip()
        else:
            # Final fallback: use the entire text as the reason
            reason = raw_text.strip()
        return reason, raw

    def _parse_action_string(self, action: str) -> Tuple[str, List[str]]:
        """
        Parse an action string like 'insert_edge("subject", "relation", "object")'
        Returns (function_name, [arg1, arg2, ...])
        Handles entity names containing commas, parentheses, quotes, and multiline outputs.
        """
        action = action.strip()
        # Normalize internal whitespace/newlines so multiline model output still parses
        action_single_line = re.sub(r"\s+", " ", action)
        # Match function name and arguments (allow content to have been multiline)
        # Pattern: function_name("arg1", "arg2", ...) or function_name('arg1', 'arg2', ...)
        pattern = r"(\w+)\s*\((.*)\)\s*$"
        match = re.match(pattern, action_single_line)
        if not match:
            raise ValueError(f"Invalid action format: {action}")
        function_name = match.group(1)
        args_str = match.group(2).strip()
        # Parse arguments by finding quoted strings (handles escaped quotes)
        parsed_args = []
        i = 0
        while i < len(args_str):
            # Skip whitespace and commas
            while i < len(args_str) and args_str[i] in " \t,":
                i += 1
            if i >= len(args_str):
                break
            # Determine quote type (single or double)
            quote_char = args_str[i]
            if quote_char not in ['"', "'"]:
                raise ValueError(f"Expected quoted string at position {i} in: {action}")
            i += 1  # Skip opening quote
            arg_value = []
            # Parse until matching closing quote (handling escaped quotes)
            while i < len(args_str):
                if args_str[i] == "\\" and i + 1 < len(args_str):
                    # Escaped character
                    arg_value.append(args_str[i + 1])
                    i += 2
                elif args_str[i] == quote_char:
                    # Found closing quote
                    parsed_args.append("".join(arg_value))
                    i += 1
                    break
                else:
                    arg_value.append(args_str[i])
                    i += 1
            else:
                # No closing quote found
                raise ValueError(f"Unclosed quote in: {action}")
        if not parsed_args:
            raise ValueError(f"No valid arguments found in: {action}")
        return function_name, parsed_args

    def _kg_refinement_action(self, query: str, triples_string: str, error_abduction_reason: str) -> Tuple[List[str], str]:
        """
        Generate a series of actions to refine the knowledge graph based on the given error reasons.
        """
        text_set = set()
        for triple in triples_string:
            sub, rel, obj = triple['subject'], triple['relation'], triple['object']
            # Safely map entities to file ids; avoid introducing new keys without defaults.
            sub_id = self._get_node_id(sub, self.entity_to_id)
            obj_id = self._get_node_id(obj, self.entity_to_id)

            sub_file_id = self.node_id_to_file_id.get(sub_id)
            if sub_file_id is not None and sub_file_id in self.text_id_to_node_name:
                text_set.add(self.text_id_to_node_name[sub_file_id])

            obj_file_id = self.node_id_to_file_id.get(obj_id)
            if obj_file_id is not None and obj_file_id in self.text_id_to_node_name:
                text_set.add(self.text_id_to_node_name[obj_file_id])
        system_prompt = REAFINER_KG_REFINEMENT_ACTION_SYSTEM_PROMPT
        user_prompt = REAFINER_KG_REFINEMENT_ACTION_USER_PROMPT.format(
            original_text=str(list(text_set)[:50]),
            question=query,
            triples_string=triples_string,
            error_reasons=error_abduction_reason,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            raw = self.llm_generator.generate_response(
                messages, temperature=0, max_new_tokens=2048
            )
            # print(f"{user_prompt}")
            print(raw)
        except Exception as e:
            # fallback
            error_message = [{"error": f"KG Refinement Action Generation Error: {e}"}]
            print(error_message)
            return error_message[0]['error'], None
        # clip
        if raw.count("delete_edge") > 20 or raw.count("insert_edge") > 30:
            error_message = [{"error": "clip"}]
            print(error_message)
            return error_message[0]['error'], None

        # Parse the output: prefer <refinement>...</refinement>, but also support
        # error-style outputs that embed an action string, e.g.
        # [{'error': 'KG Refinement Error Format: insert_edge("A","r","B")'}]
        refinement_match = re.search(r'<refinement>(.*?)</refinement>', raw, re.IGNORECASE | re.DOTALL)
        if refinement_match:
            refinement_actions_str = refinement_match.group(1).strip().strip("|")
        else:
            # Fallback: look for one or more action calls directly in the raw text.
            # We collect all occurrences of insert_edge/delete_edge/replace_node(...)
            calls = re.findall(
                r'(?:insert_edge|delete_edge|replace_node)\s*\(.*?\)',
                raw,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if not calls:
                error_message = [{"error": f"KG Refinement Error Format: {raw}"}]
                print(error_message)
                return error_message[0]['error'], None
            refinement_actions_str = "|".join(calls)

        # parse action list and store function objects
        # Support both '|' and newlines as separators, and tolerate extra whitespace.
        refinement_action_list = []
        for action in re.split(r"[\|\n]+", refinement_actions_str):
            action = action.strip()
            if not action:
                continue
            try:
                function_name, args = self._parse_action_string(action)
                if function_name == "insert_edge":
                    if len(args) != 3:
                        raise ValueError(f"insert_edge requires 3 arguments, got {len(args)}")
                    sub, rel, obj = args[0], args[1], args[2]
                    sub = self._resolve_entity_arg(sub)
                    obj = self._resolve_entity_arg(obj)
                    if sub is None or obj is None:
                        print(f"Skip insert_edge with unresolved internal id args: {args}")
                        continue
                    refinement_action_list.append(lambda s=sub, r=rel, o=obj: self._insert_edge(s, r, o))
                elif function_name == "delete_edge":
                    if len(args) != 3:
                        raise ValueError(f"delete_edge requires 3 arguments, got {len(args)}")
                    sub, rel, obj = args[0], args[1], args[2]
                    sub = self._resolve_entity_arg(sub)
                    obj = self._resolve_entity_arg(obj)
                    if sub is None or obj is None:
                        print(f"Skip delete_edge with unresolved internal id args: {args}")
                        continue
                    refinement_action_list.append(lambda s=sub, o=obj: self._delete_edge(s, o))
                elif function_name == "replace_node":
                    if len(args) != 2:
                        raise ValueError(f"replace_node requires 2 arguments, got {len(args)}")
                    old_ent, new_ent = args[0], args[1]
                    old_ent = self._resolve_entity_arg(old_ent)
                    new_ent = self._resolve_entity_arg(new_ent)
                    if old_ent is None or new_ent is None:
                        print(f"Skip replace_node with unresolved internal id args: {args}")
                        continue
                    refinement_action_list.append(lambda old=old_ent, new=new_ent: self._replace_node(old, new))
                else:
                    print(f"Error: Unknown action format: {function_name}")
                    continue
            except Exception as e:
                error_message = [{"error": f"KG Refinement Action Error Format: {action}, Error: {str(e)}"}]
                print(error_message)
                return error_message[0]['error'], None
        return refinement_action_list, raw
    
    def _kg_refinement(self, triples_string: str, error_abduction_reason: str) -> Tuple[List[Dict[str, str]], str]:
        """
        Refine the knowledge graph based on the given error reasons.
        """
        system_prompt = REAFINER_KG_REFINEMENT_SYSTEM_PROMPT
        user_prompt = REAFINER_KG_REFINEMENT_USER_PROMPT.format(
            triples_string=triples_string,
            error_reasons=error_abduction_reason,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            raw = self.llm_generator.generate_response(
                messages, temperature=0.0
            )
        except Exception as e:
            # fallback
            error_message = [{"error": f"Refinement Generation Error: {e}"}]
            print(error_message)
            return error_message, None

        # Parse the output: extract <refinement> tag
        refinement_match = re.search(r'<refinement>(.*?)</refinement>', raw, re.IGNORECASE | re.DOTALL)
        refined_triples = []
        if refinement_match:
            refinement_json = refinement_match.group(1).strip()
            try:
                # try normal json.loads first
                refined_triples = json.loads(refinement_json)
            except json.JSONDecodeError:
                # if fails, try json_repair to fix common JSON issues (like invalid escape sequences)
                try:
                    refined_triples = json_repair.loads(refinement_json)
                except Exception:
                    # fallback
                    error_message = [{"error": f"KG Refinement Error Format: {raw}"}]
                    print(error_message)
                    return error_message, None
        else:
            # try to extract JSON from raw text if <refinement> tag not found
            if '<refinement>' in raw:
                refinement_json = raw.split('<refinement>')[1].split('</refinement>')[0].strip()
            else:
                # fallback
                error_message = [{"error": f"KG Refinement Error Format: {raw}"}]
                print(error_message)
                return error_message, None
            try:
                # try normal json.loads first
                refined_triples = json.loads(refinement_json)
            except json.JSONDecodeError:
                # if fails, try json_repair to fix common JSON issues (like invalid escape sequences)
                try:
                    refined_triples = json_repair.loads(refinement_json)
                except Exception:
                    # fallback
                    error_message = [{"error": f"KG Refinement Error Format: {raw}"}]
                    print(error_message)
                    return error_message, None
        return refined_triples, raw

    # ------------------------------------------------------------------
    # KG update logic
    # ------------------------------------------------------------------
    def _insert_edge(self, sub: str, rel: str, obj: str) -> None:
        """
        Insert an edge into the KG.
        """
        sub_resolved = self._resolve_entity_arg(sub)
        obj_resolved = self._resolve_entity_arg(obj)
        if sub_resolved is None or obj_resolved is None:
            print(
                "Action Error: refuse insert_edge with unresolved internal id: ",
                sub,
                rel,
                obj,
            )
            return
        sub, obj = sub_resolved, obj_resolved
        subject_mapped_id = self._get_node_id(sub, self.entity_to_id)
        object_mapped_id = self._get_node_id(obj, self.entity_to_id)
        new_node_list = []
        # TODO: Add file_id to the node
        # Ensure subject node is in node_id_to_file_id (even if already in KG)
        if subject_mapped_id not in self.node_id_to_file_id:
            if object_mapped_id in self.node_id_to_file_id:
                self.node_id_to_file_id[subject_mapped_id] = self.node_id_to_file_id[object_mapped_id]
            else:
                self.node_id_to_file_id[subject_mapped_id] = None
        if subject_mapped_id not in self.kg.nodes:
            self.kg.add_node(
                subject_mapped_id,
                id=self._safe_sanitize(sub),
                type="entity",
                file_id=self.node_id_to_file_id[subject_mapped_id]
            )
            new_node_list.append(subject_mapped_id)
        # Ensure object node is in node_id_to_file_id (even if already in KG)
        if object_mapped_id not in self.node_id_to_file_id:
            if subject_mapped_id in self.node_id_to_file_id:
                self.node_id_to_file_id[object_mapped_id] = self.node_id_to_file_id[subject_mapped_id]
            else:
                self.node_id_to_file_id[object_mapped_id] = None
        if object_mapped_id not in self.kg.nodes:
            self.kg.add_node(
                object_mapped_id,
                id=self._safe_sanitize(obj),
                type="entity",
                file_id=self.node_id_to_file_id[object_mapped_id]
            )
            new_node_list.append(object_mapped_id)
        if not self.kg.has_edge(subject_mapped_id, object_mapped_id):
            self.kg.add_edge(
                subject_mapped_id,
                object_mapped_id,
                relation=self._safe_sanitize(rel),
                type="Relation"
            )
        # update the edge_list and edge_embeddings
        if (subject_mapped_id, object_mapped_id) not in self.edge_list:
            new_edge_embeddings = self.sentence_encoder.encode(f"{sub} {rel} {obj}", query_type="edge")
            new_edge_embeddings = new_edge_embeddings.reshape(-1, )
            new_edge_faiss_embeddings = np.array(new_edge_embeddings).astype('float32')
            # Reshape to 2D if needed (single vector should be shape [1, dim])
            if new_edge_faiss_embeddings.ndim == 1:
                new_edge_faiss_embeddings = new_edge_faiss_embeddings.reshape(1, -1)
            faiss.normalize_L2(new_edge_faiss_embeddings)
            next_faiss_id = (
                max(self.edge_faiss_id_to_list_idx.keys()) + 1
                if self.edge_faiss_id_to_list_idx
                else 0
            )
            self.edge_list.append((subject_mapped_id, object_mapped_id))
            self.edge_embeddings.append(new_edge_embeddings)
            self._append_cached_edge_score(new_edge_embeddings)
            self.edge_faiss_index.add_with_ids(new_edge_faiss_embeddings, np.array([next_faiss_id], dtype=np.int64))
            self.edge_faiss_id_to_list_idx[next_faiss_id] = len(self.edge_list) - 1
        # update the node_list and node_embeddings
        if new_node_list:
            # Encode human-readable names, never internal hash keys.
            name_list = [self._node_display_name(n) or str(n) for n in new_node_list]
            new_node_embeddings = self.sentence_encoder.encode(name_list, query_type="node")
            if isinstance(new_node_embeddings, torch.Tensor):
                new_node_embeddings = new_node_embeddings.cpu().numpy()
            new_node_embeddings = np.array(new_node_embeddings)
            if new_node_embeddings.ndim == 1:
                new_node_embeddings = new_node_embeddings.reshape(1, -1)
            new_node_embeddings_list = [emb.copy() for emb in new_node_embeddings]
            new_node_faiss_embeddings = new_node_embeddings.astype('float32')
            faiss.normalize_L2(new_node_faiss_embeddings)
            next_faiss_id = max(self.node_faiss_id_to_list_idx.keys()) + 1 if self.node_faiss_id_to_list_idx else 0
            start_list_idx = len(self.node_list)
            self.node_list.extend(new_node_list)
            self.node_embeddings.extend(new_node_embeddings_list)
            faiss_ids = np.array(list(range(next_faiss_id, next_faiss_id + len(new_node_list))), dtype=np.int64)
            self.node_faiss_index.add_with_ids(new_node_faiss_embeddings, faiss_ids)
            for i, faiss_id in enumerate(faiss_ids):
                self.node_faiss_id_to_list_idx[int(faiss_id)] = start_list_idx + i
            # Keep name -> internal-key map in sync for later retrieval/expansion.
            for n in new_node_list:
                disp = self._node_display_name(n)
                if disp:
                    self.node_id_to_attr_id[disp] = n
        # update the data
        self.data["KG"] = self.kg
        self.data["edge_list"] = self.edge_list
        self.data["node_list"] = self.node_list
        self.data["edge_embeddings"] = self.edge_embeddings
        self.data["node_embeddings"] = self.node_embeddings
        self.data["node_faiss_index"] = self.node_faiss_index
        self.data["edge_faiss_index"] = self.edge_faiss_index
        self.data["edge_faiss_id_to_list_idx"] = self.edge_faiss_id_to_list_idx
        self.data["node_faiss_id_to_list_idx"] = self.node_faiss_id_to_list_idx
        # update data in retriever
        self.retriever.KG = self.kg
        self.retriever.edge_list = self.edge_list
        self.retriever.node_list = self.node_list
        self.retriever.edge_faiss_index = self.edge_faiss_index
        self.retriever.node_faiss_index = self.node_faiss_index
        self.retriever.edge_faiss_id_to_list_idx = self.edge_faiss_id_to_list_idx
        self.retriever.node_faiss_id_to_list_idx = self.node_faiss_id_to_list_idx

    def _delete_edge(self, sub: str, obj: str) -> None:
        """
        Delete an edge from the KG.
        """
        sub_resolved = self._resolve_entity_arg(sub)
        obj_resolved = self._resolve_entity_arg(obj)
        if sub_resolved is None or obj_resolved is None:
            print(
                "Action Error: refuse delete_edge with unresolved internal id: ",
                sub,
                obj,
            )
            return
        sub, obj = sub_resolved, obj_resolved
        subject_mapped_id = self._get_node_id(sub, self.entity_to_id)
        object_mapped_id = self._get_node_id(obj, self.entity_to_id)
        edge_tuple = (subject_mapped_id, object_mapped_id)
        if not self.kg.has_edge(subject_mapped_id, object_mapped_id):
            print("Action Error: Edge not found in KG: ", sub, obj)
            return
        # Find the index of the edge in edge_list
        if edge_tuple not in self.edge_list:
            print("Action Error: Edge not found in edge_list: ", sub, obj)
            return
        list_idx = self.edge_list.index(edge_tuple)
        # Find the corresponding faiss_id by reverse lookup in mapping table
        faiss_id_to_remove = None
        for faiss_id, mapped_list_idx in self.edge_faiss_id_to_list_idx.items():
            if mapped_list_idx == list_idx:
                faiss_id_to_remove = faiss_id
                break
        if faiss_id_to_remove is None:
            print("Action Error: FAISS ID not found for edge: ", sub, obj)
            return
        
        # Remove edge from KG
        self.kg.remove_edge(subject_mapped_id, object_mapped_id)
        # Remove from FAISS index
        self.edge_faiss_index.remove_ids(np.array([faiss_id_to_remove], dtype=np.int64))
        # Remove from mapping table
        del self.edge_faiss_id_to_list_idx[faiss_id_to_remove]
        # Update remaining mappings: decrease list_idx for items after deleted one
        deleted_set = {list_idx}
        for faiss_id in self.edge_faiss_id_to_list_idx:
            current_idx = self.edge_faiss_id_to_list_idx[faiss_id]
            count_deleted_before = sum(1 for idx in deleted_set if idx < current_idx)
            self.edge_faiss_id_to_list_idx[faiss_id] -= count_deleted_before
        # Delete from lists (from back to front to avoid index shifting)
        del self.edge_list[list_idx]
        del self.edge_embeddings[list_idx]
        # Deletion shifts indices; simplest safe policy is to invalidate and recompute lazily.
        self._invalidate_edge_score_cache()
        # update the data
        self.data["KG"] = self.kg
        self.data["edge_faiss_index"] = self.edge_faiss_index
        self.data["edge_list"] = self.edge_list
        self.data["edge_embeddings"] = self.edge_embeddings
        self.data["edge_faiss_id_to_list_idx"] = self.edge_faiss_id_to_list_idx
        # update retriever
        self.retriever.KG = self.kg
        self.retriever.edge_list = self.edge_list
        self.retriever.edge_faiss_index = self.edge_faiss_index
        self.retriever.edge_faiss_id_to_list_idx = self.edge_faiss_id_to_list_idx

    def _replace_node(self, old_entity: str, new_entity: str) -> None:
        """
        Replace a node in the KG.
        """
        old_resolved = self._resolve_entity_arg(old_entity)
        new_resolved = self._resolve_entity_arg(new_entity)
        if old_resolved is None or new_resolved is None:
            print(
                "Action Error: refuse replace_node with unresolved internal id: ",
                old_entity,
                new_entity,
            )
            return
        old_entity, new_entity = old_resolved, new_resolved
        old_mapped_id = self._get_node_id(old_entity, self.entity_to_id)
        new_mapped_id = self._get_node_id(new_entity, self.entity_to_id)
        if not self.kg.has_node(old_mapped_id):
            print("Action Error: Node not found in KG: ", old_entity)
            return
        # obtain all edges connected to the old node, preserve for the new node
        edges_to_preserve = []
        edges_to_delete = []
        for neighbor in sorted(self.kg.successors(old_mapped_id)):
            neighbor_id = self._node_display_name(neighbor)
            if neighbor_id:
                relation = self.kg.edges[old_mapped_id, neighbor]["relation"]
                edges_to_preserve.append((new_entity, relation, neighbor_id))
                edges_to_delete.append((old_entity, neighbor_id))
        for neighbor in sorted(self.kg.predecessors(old_mapped_id)):
            neighbor_id = self._node_display_name(neighbor)
            if neighbor_id:
                relation = self.kg.edges[neighbor, old_mapped_id]["relation"]
                edges_to_preserve.append((neighbor_id, relation, new_entity))
                edges_to_delete.append((neighbor_id, old_entity))
        # delete the original edges
        for edge in edges_to_delete:
            self._delete_edge(edge[0], edge[1])
        # add the new edges and update the data
        for edge in edges_to_preserve:
            self._insert_edge(edge[0], edge[1], edge[2])
        # delete the original node
        self.kg.remove_node(old_mapped_id)
        # update the node data
        node_idx = self.node_list.index(old_mapped_id)
        faiss_id_to_remove = None
        for faiss_id, mapped_list_idx in self.node_faiss_id_to_list_idx.items():
            if mapped_list_idx == node_idx:
                faiss_id_to_remove = faiss_id
                break
        if faiss_id_to_remove is None:
            print("Action Error: FAISS ID not found for node: ", old_entity)
            return
        # Remove from FAISS index
        self.node_faiss_index.remove_ids(np.array([faiss_id_to_remove], dtype=np.int64))
        # Remove from mapping table
        del self.node_faiss_id_to_list_idx[faiss_id_to_remove]
        # Update remaining mappings: decrease list_idx for items after deleted one
        deleted_set = {node_idx}
        for faiss_id in self.node_faiss_id_to_list_idx:
            current_idx = self.node_faiss_id_to_list_idx[faiss_id]
            count_deleted_before = sum(1 for idx in deleted_set if idx < current_idx)
            self.node_faiss_id_to_list_idx[faiss_id] -= count_deleted_before
        # Delete from lists (from back to front to avoid index shifting)
        del self.node_list[node_idx]
        del self.node_embeddings[node_idx]
        # update the data
        self.data["KG"] = self.kg
        self.data["node_faiss_index"] = self.node_faiss_index
        self.data["node_list"] = self.node_list
        self.data["node_embeddings"] = self.node_embeddings
        self.data["node_faiss_id_to_list_idx"] = self.node_faiss_id_to_list_idx
        # update retriever
        self.retriever.KG = self.kg
        self.retriever.node_list = self.node_list
        self.retriever.node_faiss_index = self.node_faiss_index
        self.retriever.node_faiss_id_to_list_idx = self.node_faiss_id_to_list_idx

    def _generate_answer(self, query: str, subgraph_str: str) -> str:
        """
        Generate the final answer on the final refined subgraph (no Yes/No judgment).
        """
        return self.llm_generator.generate_with_context_kg(query, subgraph_str, temperature=0.0)

    def _get_node_id(self, entity_name, entity_to_id=None):
        """Returns existing or creates new hash key for an entity (internal graph key only)."""
        if entity_to_id is None:
            entity_to_id = self.entity_to_id
        name = str(entity_name)
        # Never register a bare sha256 string as an entity *name* (would re-hash and
        # also leak into prompts if used as node["id"]).
        if self._looks_like_internal_id(name):
            if hasattr(self, "kg") and name in self.kg.nodes:
                return name
            for _disp, key in entity_to_id.items():
                if key == name:
                    return name
            raise ValueError(f"Refusing to mint entity from internal hash id: {name}")
        if name not in entity_to_id:
            hash_object = hashlib.sha256((name + "_entity").encode("utf-8"))
            entity_to_id[name] = hash_object.hexdigest()
        return entity_to_id[name]

    @staticmethod
    def _looks_like_internal_id(value: Any) -> bool:
        """True for sha256-hex internal node keys that must never appear in prompts."""
        if value is None:
            return False
        s = str(value).strip()
        if len(s) != 64:
            return False
        try:
            int(s, 16)
            return True
        except ValueError:
            return False

    def _node_display_name(self, node_key: Any) -> str:
        """Human-readable entity name for prompts; never the internal hash key."""
        attrs = {}
        if hasattr(self, "kg") and node_key in self.kg.nodes:
            attrs = self.kg.nodes[node_key]
        name = attrs.get("id", None)
        if name is None or self._looks_like_internal_id(name):
            # Last resort: if node_key itself is not a hash, use it; else empty.
            if not self._looks_like_internal_id(node_key):
                return str(node_key)
            return ""
        return str(name)

    def _resolve_entity_arg(self, name: Any) -> Optional[str]:
        """Map action/prompt entity args to a human-readable name.

        Bare sha256 keys are resolved via the KG when possible; otherwise None
        so callers can skip writing hashes into the graph/prompts.
        """
        if name is None:
            return None
        text = str(name).strip()
        if not text:
            return None
        if not self._looks_like_internal_id(text):
            return text
        # Arg is an internal key: recover display name if the node exists.
        if text in self.kg.nodes:
            disp = self._node_display_name(text)
            return disp or None
        for disp, key in self.entity_to_id.items():
            if key == text and not self._looks_like_internal_id(disp):
                return str(disp)
        return None

    def _named_triple_from_edge(
        self, u: Any, v: Any, relation: Any
    ) -> Optional[Tuple[str, str, str]]:
        s = self._node_display_name(u)
        o = self._node_display_name(v)
        if not s or not o:
            return None
        if self._looks_like_internal_id(s) or self._looks_like_internal_id(o):
            return None
        return (s, str(relation or ""), o)

    def _sanitize_named_triple(
        self, s: Any, r: Any, o: Any
    ) -> Optional[Tuple[str, str, str]]:
        s2 = self._resolve_entity_arg(s)
        o2 = self._resolve_entity_arg(o)
        if s2 is None or o2 is None:
            return None
        return (s2, str(r or ""), o2)

    def _sanitize_named_triple_str(
        self, triple_str: str
    ) -> Optional[Tuple[str, str, str]]:
        parts = str(triple_str).split("  ", 2)
        if len(parts) != 3:
            return None
        return self._sanitize_named_triple(parts[0].strip(), parts[1].strip(), parts[2].strip())

    def _sanitize_context_list(self, context: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for triple_str in context or []:
            cleaned = self._sanitize_named_triple_str(triple_str)
            if cleaned is None:
                continue
            s = f"{cleaned[0]}  {cleaned[1]}  {cleaned[2]}"
            if s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out

    def _safe_sanitize(self, value):
        """Safely sanitize any value for XML output."""
        def _sanitize_xml_string(s: str) -> str:
            """Remove illegal XML characters from a string."""
            return _ILLEGAL_XML_RE.sub("", s)
        if value is None:
            return ""
        return _sanitize_xml_string(str(value))

    def _construct_subgraph(self, initial_nodes, num_hop: int = 1):
        """Construct a multi-hop subgraph around initial nodes up to num_hop.

        Node/edge attributes are copied from ``self.kg`` so callers can read
        human-readable ``id`` / ``relation`` fields (not internal hash keys).
        """
        subgraph = DiGraph()
        visited = set()
        queue = [(node, 0) for node in initial_nodes if node in self.kg.nodes]

        def _add_node(n):
            if n not in subgraph:
                subgraph.add_node(n, **dict(self.kg.nodes[n]))

        # Add initial nodes
        for node, _ in queue:
            _add_node(node)
            visited.add(node)

        # Breadth-first search to collect neighbors
        while queue:
            current_node, hop_count = queue.pop(0)
            if hop_count >= num_hop:
                continue
            # Add successors (outgoing edges)
            for neighbor in sorted(self.kg.successors(current_node)):
                neighbor_id = self.kg.nodes[neighbor].get("id", None)
                relation = self.kg.edges[(current_node, neighbor)].get("relation", "")
                edge_attrs = dict(self.kg.edges[(current_node, neighbor)])
                if neighbor_id is not None and str(neighbor_id).isdigit():
                    # Do not further explore this neighbor
                    _add_node(neighbor)
                    subgraph.add_edge(current_node, neighbor, **edge_attrs)
                    continue
                if neighbor not in visited:
                    visited.add(neighbor)
                    _add_node(neighbor)
                    queue.append((neighbor, hop_count + 1))
                else:
                    _add_node(neighbor)
                subgraph.add_edge(current_node, neighbor, **edge_attrs)

            # Add predecessors (incoming edges)
            for neighbor in sorted(self.kg.predecessors(current_node)):
                neighbor_id = self.kg.nodes[neighbor].get("id", None)
                edge_attrs = dict(self.kg.edges[(neighbor, current_node)])
                if neighbor_id is not None and str(neighbor_id).isdigit():
                    # Do not further explore this neighbor
                    _add_node(neighbor)
                    subgraph.add_edge(neighbor, current_node, **edge_attrs)
                    continue
                if neighbor not in visited:
                    visited.add(neighbor)
                    _add_node(neighbor)
                    queue.append((neighbor, hop_count + 1))
                else:
                    _add_node(neighbor)
                subgraph.add_edge(neighbor, current_node, **edge_attrs)

        return subgraph


__all__ = ["DeepRefine", "RetrievalStepResult", "RefinementResult"]