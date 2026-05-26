import copy
import hashlib
import logging
import os
import re
import numpy as np
import json_repair
import networkx as nx
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
from networkx import DiGraph

from .base import BaseInteraction

try:
    from autograph.rag_server.edge_retriever import EdgeRetriever
except ImportError:
    from autorefiner.src.rag_server.edge_retriever import EdgeRetriever

from autorefiner.src.rag_server.deeprefine_prompt import (
    REAFINER_JUDGEMENT_SYSTEM_PROMPT,
    REAFINER_JUDGEMENT_USER_PROMPT,
    REAFINER_ERROR_ABDUCTION_SYSTEM_PROMPT,
    REAFINER_ERROR_ABDUCTION_USER_PROMPT,
    REAFINER_KG_REFINEMENT_ACTION_SYSTEM_PROMPT,
    REAFINER_KG_REFINEMENT_ACTION_USER_PROMPT,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_ILLEGAL_XML_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\uD800-\uDFFF\uFFFE\uFFFF]")

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


def _format_original_chunk_for_action_prompt(original_chunk: Any) -> str:
    """Text for REAFINER_KG_REFINEMENT_ACTION_USER_PROMPT `original_text` (from data interaction_kwargs)."""
    if original_chunk is None:
        return ""
    if isinstance(original_chunk, str):
        return original_chunk
    if isinstance(original_chunk, (list, tuple)):
        return "\n\n".join(str(x) for x in original_chunk if str(x).strip())
    return str(original_chunk)


class RefinementInteraction(BaseInteraction):
    """Refinement pipeline as three ordered states: ANSWERABLE_JUDGEMENT -> ABDUCTION -> ACTION_GENERATION -> RAG.

    Each state corresponds to one LLM turn via _handle_engine_call; parsing and next-prompt
    construction happen in INTERACTING via generate_response_refinement().
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict: Dict[str, dict] = {}
        self.base_top_k = config.get("base_top_k", 10)
        self.max_hops = config.get("max_hops", 3)
        self.increment_hop = config.get("increment_hop", 1)
        self.max_triple_num = config.get("max_triple_num", 90)
        # Per judgement-round triple budget: hop 0 == base_top_k, then +40 each hop up to max_triple_num.
        # Mirrors Reafiner(..., max_triple_num_by_step=[10, 50, 90], max_hops=3).
        step_caps = config.get("max_triple_num_by_step")
        if step_caps is not None:
            self.max_triple_num_by_step = [int(x) for x in step_caps]
        else:
            self.max_triple_num_by_step = [
                min(self.base_top_k + 40 * i, self.max_triple_num) for i in range(max(1, self.max_hops))
            ]
        while len(self.max_triple_num_by_step) < self.max_hops:
            last = min(self.max_triple_num_by_step[-1], self.max_triple_num)
            self.max_triple_num_by_step.append(min(last + 40, self.max_triple_num))
        self.history_horizon_size = config.get("history_horizon_size", 3)
        self.max_format_retry = config.get("max_format_retry", 2)

    def _triple_cap_for_expansion(self, judgement_steps: int) -> int:
        """Triple budget for context after a failed judgement.

        `judgement_steps` is 1-based (the judgement just completed with answerable=False).
        The expanded subgraph shown before the next judgement uses cap at index `judgement_steps`
        (e.g. after 1st 'no' -> second-round cap, typically 50 when defaults are [10,50,90]).
        """
        idx = min(judgement_steps, len(self.max_triple_num_by_step) - 1)
        return min(self.max_triple_num_by_step[idx], self.max_triple_num)

    def _prune_working_kg_and_context(
        self,
        kg_full: DiGraph,
        entity_to_id: Dict[str, str],
        question: Optional[str],
        sentence_encoder: Optional[Any],
        max_items: int,
    ) -> Tuple[DiGraph, List[str], Dict[str, Any]]:
        """Keep full KG in *kg_full*; build working *kg* and *sorted_context* with at most *max_items* triples."""
        subgraph_triples = sorted(
            [
                (kg_full.nodes[u].get("id", u), d.get("relation", ""), kg_full.nodes[v].get("id", v))
                for u, v, d in kg_full.edges(data=True)
            ]
        )
        # No embedding sort/prune when already within budget (saves encoder calls).
        if max_items <= 0 or len(subgraph_triples) <= max_items:
            selected = subgraph_triples
        else:
            selected = self._rank_triples_by_query_similarity(
                subgraph_triples=subgraph_triples,
                query=question or "",
                sentence_encoder=sentence_encoder,
                max_items=max_items,
            )
        kg = DiGraph()
        for s, r, o in selected:
            sid = self._get_node_id(s, entity_to_id)
            oid = self._get_node_id(o, entity_to_id)
            if sid not in kg.nodes:
                kg.add_node(sid, id=s, type="entity")
            if oid not in kg.nodes:
                kg.add_node(oid, id=o, type="entity")
            if not kg.has_edge(sid, oid):
                kg.add_edge(sid, oid, relation=r)
        node_id_to_attr_id = {kg.nodes[n].get("id", str(n)): n for n in kg.nodes}
        sorted_context = [f"{s}  {r}  {o}" for s, r, o in selected]
        return kg, sorted_context, node_id_to_attr_id

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
        question: Optional[str] = None,
        **kwargs,
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())

        # Priority: per-sample KG (draft_kg / injected KG context) should be isolated per request.
        # We no longer support global full_graph_data here; every request must carry its own KG.
        sentence_encoder = kwargs.get("sentence_encoder")
        draft_kg = kwargs.get("draft_kg")  # list[{"subject","relation","object"}, ...]

        # Optional retrieval support is disabled when a per-sample KG is provided. We keep
        # the variables here only for backward compatibility if someone still passes a
        # full_graph_data dict explicitly.
        kg_orig = None
        edge_list: list = []
        edge_faiss_index = None
        edge_faiss_id_to_list_idx: dict = {}

        # Extract triples from pre-populated messages instead of re-retrieving.
        # The data preparation stage already retrieved the first-hop subgraph and built the prompt.
        initial_messages = kwargs.get("initial_messages", [])
        sorted_context: List[str] = []
        prompt_judgement_system = REAFINER_JUDGEMENT_SYSTEM_PROMPT
        prompt_judgement_user: Optional[str] = None

        parsed_prompt_triples: List[Dict[str, str]] = []
        
        if initial_messages:
            # Extract system and user prompts from pre-populated messages
            for msg in initial_messages:
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                else:
                    # Message object with .role and .content attributes
                    role = getattr(msg, "role", "")
                    content = getattr(msg, "content", "")
                
                if role == "system":
                    prompt_judgement_system = content
                elif role == "user":
                    prompt_judgement_user = content
                    # Parse triples from user message
                    # Supports:
                    # 1) JSON list under "Knowledge Graph (KG) context: [...]"
                    # 2) legacy "subject  relation  object" lines
                    if "Knowledge Graph (KG) context:" in content:
                        kg_section = content.split("Knowledge Graph (KG) context:")[-1].strip()
                        # Extract the JSON array substring if the prompt contains extra trailing text.
                        kg_json_text = kg_section
                        lb = kg_section.find("[")
                        rb = kg_section.rfind("]")
                        if lb != -1 and rb != -1 and rb > lb:
                            kg_json_text = kg_section[lb : rb + 1]
                        # Try JSON first
                        try:
                            kg_json = json_repair.loads(kg_json_text)
                            if isinstance(kg_json, list):
                                for t in kg_json:
                                    if not isinstance(t, dict):
                                        continue
                                    s = str(t.get("subject", "")).strip()
                                    r = str(t.get("relation", "")).strip()
                                    o = str(t.get("object", "")).strip()
                                    if s and o:
                                        parsed_prompt_triples.append({"subject": s, "relation": r, "object": o})
                        except Exception:
                            kg_json = None

                        # Fallback: legacy "subject  relation  object" per line
                        if not parsed_prompt_triples:
                            for line in kg_section.split("\n"):
                                line = line.strip()
                                if line and "  " in line:
                                    parts = line.split("  ", 2)
                                    if len(parts) == 3:
                                        sorted_context.append(line)

        # If prompt provided JSON triples but not legacy triple lines, populate sorted_context from JSON.
        if parsed_prompt_triples and not sorted_context:
            sorted_context = [
                f"{t['subject']}  {t.get('relation','')}  {t['object']}" for t in parsed_prompt_triples
            ]

        # Build a per-instance KG (mutable) from the per-sample triples if available.
        # This ensures different requests do NOT share KG state and RAG/refinement run on the sample's own KG.
        kg: DiGraph
        node_id_to_attr_id: Dict[str, Any]
        kg_full: Optional[DiGraph] = None
        entity_to_id: Dict[str, str] = {}

        triples_for_kg: List[Dict[str, str]] = []
        if isinstance(draft_kg, list) and draft_kg:
            for t in draft_kg:
                if not isinstance(t, dict):
                    continue
                s = str(t.get("subject", "")).strip()
                r = str(t.get("relation", "")).strip()
                o = str(t.get("object", "")).strip()
                if s and o:
                    triples_for_kg.append({"subject": s, "relation": r, "object": o})
        elif parsed_prompt_triples:
            triples_for_kg = parsed_prompt_triples

        if triples_for_kg:
            kg = DiGraph()
            for t in triples_for_kg:
                s = self._safe_sanitize(t.get("subject", ""))
                r = self._safe_sanitize(t.get("relation", ""))
                o = self._safe_sanitize(t.get("object", ""))
                if not s or not o:
                    continue
                sid = self._get_node_id(s, entity_to_id)
                oid = self._get_node_id(o, entity_to_id)
                if sid not in kg.nodes:
                    kg.add_node(sid, id=s, type="entity")
                if oid not in kg.nodes:
                    kg.add_node(oid, id=o, type="entity")
                if not kg.has_edge(sid, oid):
                    kg.add_edge(sid, oid, relation=r)
            kg_full = copy.deepcopy(kg)
            cap0 = min(self.max_triple_num_by_step[0], self.max_triple_num)
            kg, sorted_context, node_id_to_attr_id = self._prune_working_kg_and_context(
                kg_full, entity_to_id, question, sentence_encoder, cap0
            )
        else:
            # No per-sample KG: we currently require every request to carry its own KG
            # (e.g., via draft_kg or prompt-injected triples). This keeps behavior explicit
            # and avoids accidentally sharing a global full_graph_data.
            raise ValueError("draft_kg or prompt-injected KG context is required for RefinementInteraction")
        
        # Ensure prompt_judgement_user is set (fallback if neither initial_messages nor retrieval worked)
        if not prompt_judgement_user:
            triples_string = "\n".join(sorted_context) if sorted_context else "(no triples)"
            prompt_judgement_user = REAFINER_JUDGEMENT_USER_PROMPT.format(question=question or "", triples_string=triples_string)

        self._instance_dict[instance_id] = {
            "rag_state": False,
            "ground_truth": ground_truth,
            "question": question,
            "kg": kg,
            # For multi-hop expansion in judgement phase, use per-instance full KG if available.
            # NOTE: should be query-specific; if only a draft KG is provided, this equals the draft KG.
            "kg_full": kg_full,
            "node_id_to_attr_id": node_id_to_attr_id,
            "entity_to_id": entity_to_id,
            "prompt_judgement_system": prompt_judgement_system,
            "prompt_judgement_user": prompt_judgement_user,
            "refinement_initial_injected": False,
            "refinement_phase": "answerable_judgement",
            "interaction_history": [],
            "sorted_context": sorted_context,
            # Source document text for KG extraction (interaction_kwargs["original_chunk"]), not the subgraph lines.
            "original_chunk": kwargs.get("original_chunk"),
            "sentence_encoder": sentence_encoder,
            "error_abduction_reason": None,
            "format_retry_count": 0,
        }
        return instance_id

    async def generate_response_refinement(
        self,
        instance_id: str,
        messages: List[Dict[str, Any]],
        current_phase: str,
        **kwargs,
    ) -> Tuple[bool, str, float, dict]:
        """Parse last assistant message and return (should_terminate, next_user_message, reward, extra).
        extra may contain next_system, next_rag_state (enum value string or enum)."""
        inst = self._instance_dict[instance_id]
        content = ""
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                content = messages[i].get("content", "") or ""
                break

        reward = 0.0
        extra: Dict[str, Any] = {}

        if current_phase == "answerable_judgement":
            # Parse current judgement result
            judge_match = re.search(r"<judge>(.*?)</judge>", content, re.IGNORECASE | re.DOTALL)
            if not judge_match:
                # Strictly require <judge>...</judge>, otherwise treat as parse failure.
                print(
                    f"\033[91m [instance {instance_id}] "
                    f"[Failed to parse judgement content: missing <judge> tag]\nContent: {content} \033[0m"
                )
                return True, "Failed to parse judgement content.", 0.0, {}

            answerable_str = judge_match.group(1).strip().lower()
            answerable = answerable_str.startswith("yes")

            # Current judgement step index (1-based)
            prev_steps = sum(1 for h in inst["interaction_history"] if h.get("phase") == "judgement")
            judgement_steps = prev_steps + 1

            # Record structured interaction history, similar to reafiner.Reafiner
            inst["interaction_history"].append(
                {
                    "phase": "judgement",
                    "raw_response": content,
                    "query": inst.get("question", ""),
                    "subgraph_hop": judgement_steps,
                    "subgraph_content": inst.get("sorted_context", []),
                    "answerable": answerable,
                }
            )
            print(
                f"\033[94m [instance {instance_id}] "
                f"[Judgement Steps: {judgement_steps}, Answerable: {answerable}] \033[0m"
            )

            if answerable:
                # inst["rag_state"] = True
                if judgement_steps == 1:
                    return True, "No need to do any refinement.", 1.0, {}
                else:
                    inst["refinement_phase"] = "abduction"
                    # Build interaction history string in the same style as reafiner.py
                    history = inst["interaction_history"]
                    horizon = getattr(self, "history_horizon_size", 0) or 0
                    if horizon > 0 and len(history) > horizon:
                        used_history = history[:-horizon]
                    else:
                        used_history = history
                    hist_str = "\n".join(
                        [
                            "Step{}:\n['Query': {}, 'Subgraph_hop': {}, 'Subgraph_content': {}, 'Answerable': {}]\n".format(
                                i + 1,
                                h.get("query", ""),
                                h.get("subgraph_hop", ""),
                                str(h.get("subgraph_content", "")),
                                h.get("answerable", ""),
                            )
                            for i, h in enumerate(used_history)
                        ]
                    )
                    next_system = REAFINER_ERROR_ABDUCTION_SYSTEM_PROMPT
                    next_user = REAFINER_ERROR_ABDUCTION_USER_PROMPT.format(interaction_history=hist_str)
                    extra["next_system"] = next_system
                    extra["next_rag_state"] = "abduction"
                    return False, next_user, 1.0, extra

            # Not answerable: keep doing answerable_judgement up to max_hops.
            # 即使这一轮没法扩张子图，也继续用当前子图再判一轮，而不是立刻进 abduction。
            if judgement_steps < self.max_hops:
                # Expand on the per-instance full KG (query-specific). Do NOT share KG across requests.
                kg_orig = inst.get("kg_full")
                if kg_orig is not None:
                    node_list = list(kg_orig.nodes())
                    sorted_ctx = inst.get("sorted_context") or []
                    node_str_list: List[str] = []
                    for triple_str in sorted_ctx:
                        parts = triple_str.split("  ", 2)
                        if len(parts) == 3:
                            node_str_list.append(parts[0].strip())
                            node_str_list.append(parts[2].strip())
                    node_str_list = list(set(node_str_list))
                    id_to_node = {kg_orig.nodes[n].get("id", n): n for n in kg_orig.nodes()}
                    initial_nodes = [id_to_node[ns] for ns in node_str_list if ns in id_to_node]
                    if initial_nodes:
                        subgraph = self._construct_subgraph(
                            kg_orig, node_list, initial_nodes, num_hop=self.increment_hop
                        )
                        cap = self._triple_cap_for_expansion(judgement_steps)
                        inst["kg"], sorted_context, inst["node_id_to_attr_id"] = (
                            self._prune_working_kg_and_context(
                                copy.deepcopy(subgraph),
                                inst["entity_to_id"],
                                inst.get("question"),
                                inst.get("sentence_encoder"),
                                cap,
                            )
                        )
                        inst["sorted_context"] = sorted_context
                        triples_string = "\n".join(sorted_context) if sorted_context else "(no triples)"
                        inst["prompt_judgement_user"] = REAFINER_JUDGEMENT_USER_PROMPT.format(
                            question=inst["question"], triples_string=triples_string
                        )
                # 无论本轮是否成功扩张子图，只要还没到 max_hops，就继续下一轮 judgement
                extra["next_system"] = inst["prompt_judgement_system"]
                extra["next_rag_state"] = "answerable_judgement"
                return False, inst["prompt_judgement_user"], 1.0, extra

            # 达到 max_hops 才进入 abduction
            inst["refinement_phase"] = "abduction"
            history = inst["interaction_history"]
            horizon = getattr(self, "history_horizon_size", 0) or 0
            if horizon > 0 and len(history) > horizon:
                used_history = history[:-horizon]
            else:
                used_history = history
            hist_str = "\n".join(
                [
                    "Step{}:\n['Query': {}, 'Subgraph_hop': {}, 'Subgraph_content': {}, 'Answerable': {}]\n".format(
                        i + 1,
                        h.get("query", ""),
                        h.get("subgraph_hop", ""),
                        str(h.get("subgraph_content", "")),
                        h.get("answerable", ""),
                    )
                    for i, h in enumerate(used_history)
                ]
            )
            next_system = REAFINER_ERROR_ABDUCTION_SYSTEM_PROMPT
            next_user = REAFINER_ERROR_ABDUCTION_USER_PROMPT.format(interaction_history=hist_str)
            extra["next_system"] = next_system
            extra["next_rag_state"] = "abduction"
            return False, next_user, 1.0, extra

        if current_phase == "abduction":
            print(f"\033[94m [instance {instance_id}] [Abduction] \033[0m")
            print(f"Raw content:\n {content}")
            inst["interaction_history"].append({"phase": "abduction", "raw_response": content})
            # Abduction phase must output <abduction>...</abduction>, otherwise treat as parse failure.
            abduction_match = re.search(r"<abduction>(.*?)</abduction>", content, re.IGNORECASE | re.DOTALL)
            if not abduction_match:
                print(
                    f"\033[91m [instance {instance_id}] "
                    f"[Failed to parse abduction content: missing <abduction> tag]\nContent: {content} \033[0m"
                )
                inst["refinement_phase"] = "action_generation"
                return True, "Failed to parse abduction content.", 0.0, {}

            error_reason = abduction_match.group(1).strip()
            inst["error_abduction_reason"] = error_reason
            inst["refinement_phase"] = "action_generation"
            triples_string = "\n".join(inst["sorted_context"]) if inst.get("sorted_context") else ""
            next_system = REAFINER_KG_REFINEMENT_ACTION_SYSTEM_PROMPT
            next_user = REAFINER_KG_REFINEMENT_ACTION_USER_PROMPT.format(
                original_text=_format_original_chunk_for_action_prompt(inst.get("original_chunk")),
                triples_string=triples_string,
                question=inst["question"],
                error_reasons=error_reason,
            )
            extra["next_system"] = next_system
            extra["next_rag_state"] = "action_generation"
            return False, next_user, 1.0, extra

        if current_phase == "action_generation":
            print(f"\033[94m [instance {instance_id}] [Action Generation] \033[0m")
            print(f"Raw content:\n {content}")
            inst["interaction_history"].append({"phase": "action", "raw_response": content})

            refinement_match = re.search(r"<refinement>(.*?)</refinement>", content, re.IGNORECASE | re.DOTALL)
            selfclose_match = re.search(r"<refinement\s*/>", content, re.IGNORECASE) if not refinement_match else None
            parse_ok = refinement_match is not None or selfclose_match is not None
            apply_ok = False

            if parse_ok:
                try:
                    actions_text = refinement_match.group(1).strip() if refinement_match else ""
                    apply_ok = self._apply_refinement_actions(instance_id, actions_text)
                except Exception as e:
                    print(
                        f"\033[91m [instance {instance_id}] "
                        f"[Failed to apply refinement actions: {e}\nContent: {content}] \033[0m"
                    )

            if not parse_ok or not apply_ok:
                retry_count = inst.get("format_retry_count", 0)
                if retry_count < self.max_format_retry:
                    inst["format_retry_count"] = retry_count + 1
                    print(
                        f"\033[93m [instance {instance_id}] "
                        f"[Format retry {inst['format_retry_count']}/{self.max_format_retry}] \033[0m"
                    )
                    extra["next_system"] = inst["prompt_judgement_system"]
                    extra["next_rag_state"] = "action_generation"
                    extra["format_error"] = True
                    return False, _FORMAT_RETRY_PROMPT, 0.0, extra

                print(
                    f"\033[91m [instance {instance_id}] "
                    f"[Format retry exhausted ({self.max_format_retry}), fallback to original KG for RAG] \033[0m"
                )
                # Restore original KG so RAG runs on unmodified graph
                if inst.get("kg_full") is not None:
                    inst["kg"] = copy.deepcopy(inst["kg_full"])
                inst["format_retry_count"] = 0
                inst["rag_state"] = True
                extra["format_error"] = True
                extra["next_rag_state"] = "rag"
                return False, "You will perform graph based RAG based on the original knowledge graph.", 0.0, extra

            inst["format_retry_count"] = 0
            inst["rag_state"] = True
            extra["next_rag_state"] = "rag"
            return False, "You will perform graph based RAG based on your constructed knowledge graph.", 1.0, extra

        return True, "Unknown refinement phase.", 0.0, {}

    async def generate_response_refinement_simple(
        self,
        instance_id: str,
        messages: List[Dict[str, Any]],
        current_phase: str,
        **kwargs,
    ) -> Tuple[bool, str, float, dict]:
        """A more tolerant version of generate_response_refinement.

        - judgement: if no <judge> tag, fall back to searching 'yes'/'no' in plain text.
        - abduction: if no <abduction> tag, treat full content as error reason.
        - action_generation: if no <refinement> tag, treat full content as action string.
        """
        inst = self._instance_dict[instance_id]
        content = ""
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                content = messages[i].get("content", "") or ""
                break

        reward = 0.0
        extra: Dict[str, Any] = {}

        if current_phase == "answerable_judgement":
            # Tolerant judgement parsing
            judge_match = re.search(r"<judge>(.*?)</judge>", content, re.IGNORECASE | re.DOTALL)
            if judge_match:
                answerable_str = judge_match.group(1).strip().lower()
                answerable = answerable_str.startswith("yes")
            else:
                # Fallback: heuristic on plain text
                lower = content.lower()
                window = lower[:200]
                if "yes" in window and "no" not in window:
                    answerable = True
                elif "no" in window and "yes" not in window:
                    answerable = False
                else:
                    # Default to not answerable if unclear
                    answerable = False

            prev_steps = sum(1 for h in inst["interaction_history"] if h.get("phase") == "judgement")
            judgement_steps = prev_steps + 1

            inst["interaction_history"].append(
                {
                    "phase": "judgement",
                    "raw_response": content,
                    "query": inst.get("question", ""),
                    "subgraph_hop": judgement_steps,
                    "subgraph_content": inst.get("sorted_context", []),
                    "answerable": answerable,
                }
            )
            print(
                f"\033[94m [instance {instance_id}] "
                f"[Judgement Steps: {judgement_steps}, Answerable: {answerable}] \033[0m"
            )

            if answerable:
                if judgement_steps == 1:
                    # No refinement needed, but we still want to run RAG and compute
                    # the downstream reward (e.g., gbd_f1_reward) on the current KG.
                    # Treat the current KG (built from draft_kg / prompt KG) as the
                    # "refined" KG and transition directly into the RAG phase.
                    inst["rag_state"] = True
                    extra["next_rag_state"] = "rag"
                    # Keep reward=1.0 here to indicate a successful judgement step; the
                    # actual F1 gain is computed later from the RAG answer.
                    return False, "No refinement required. Proceed to graph-based RAG on the current KG.", 1.0, extra
                else:
                    inst["refinement_phase"] = "abduction"
                    history = inst["interaction_history"]
                    horizon = getattr(self, "history_horizon_size", 0) or 0
                    if horizon > 0 and len(history) > horizon:
                        used_history = history[:-horizon]
                    else:
                        used_history = history
                    hist_str = "\n".join(
                        [
                            "Step{}:\n['Query': {}, 'Subgraph_hop': {}, 'Subgraph_content': {}, 'Answerable': {}]\n".format(
                                i + 1,
                                h.get("query", ""),
                                h.get("subgraph_hop", ""),
                                str(h.get("subgraph_content", "")),
                                h.get("answerable", ""),
                            )
                            for i, h in enumerate(used_history)
                        ]
                    )
                    next_system = REAFINER_ERROR_ABDUCTION_SYSTEM_PROMPT
                    next_user = REAFINER_ERROR_ABDUCTION_USER_PROMPT.format(interaction_history=hist_str)
                    extra["next_system"] = next_system
                    extra["next_rag_state"] = "abduction"
                    return False, next_user, 1.0, extra

            # Not answerable: keep doing answerable_judgement up to max_hops.
            if judgement_steps < self.max_hops:
                # Expand on the per-instance full KG (query-specific). Do NOT share KG across requests.
                kg_orig = inst.get("kg_full")
                if kg_orig is not None:
                    node_list = list(kg_orig.nodes())
                    sorted_ctx = inst.get("sorted_context") or []
                    node_str_list: List[str] = []
                    for triple_str in sorted_ctx:
                        parts = triple_str.split("  ", 2)
                        if len(parts) == 3:
                            node_str_list.append(parts[0].strip())
                            node_str_list.append(parts[2].strip())
                    node_str_list = list(set(node_str_list))
                    id_to_node = {kg_orig.nodes[n].get("id", n): n for n in kg_orig.nodes()}
                    initial_nodes = [id_to_node[ns] for ns in node_str_list if ns in id_to_node]
                    if initial_nodes:
                        subgraph = self._construct_subgraph(
                            kg_orig, node_list, initial_nodes, num_hop=self.increment_hop
                        )
                        cap = self._triple_cap_for_expansion(judgement_steps)
                        inst["kg"], sorted_context, inst["node_id_to_attr_id"] = (
                            self._prune_working_kg_and_context(
                                copy.deepcopy(subgraph),
                                inst["entity_to_id"],
                                inst.get("question"),
                                inst.get("sentence_encoder"),
                                cap,
                            )
                        )
                        inst["sorted_context"] = sorted_context
                        triples_string = "\n".join(sorted_context) if sorted_context else "(no triples)"
                        inst["prompt_judgement_user"] = REAFINER_JUDGEMENT_USER_PROMPT.format(
                            question=inst["question"], triples_string=triples_string
                        )
                extra["next_system"] = inst["prompt_judgement_system"]
                extra["next_rag_state"] = "answerable_judgement"
                return False, inst["prompt_judgement_user"], 1.0, extra

            # 达到 max_hops 才进入 abduction
            inst["refinement_phase"] = "abduction"
            history = inst["interaction_history"]
            horizon = getattr(self, "history_horizon_size", 0) or 0
            if horizon > 0 and len(history) > horizon:
                used_history = history[:-horizon]
            else:
                used_history = history
            hist_str = "\n".join(
                [
                    "Step{}:\n['Query': {}, 'Subgraph_hop': {}, 'Subgraph_content': {}, 'Answerable': {}]\n".format(
                        i + 1,
                        h.get("query", ""),
                        h.get("subgraph_hop", ""),
                        str(h.get("subgraph_content", "")),
                        h.get("answerable", ""),
                    )
                    for i, h in enumerate(used_history)
                ]
            )
            next_system = REAFINER_ERROR_ABDUCTION_SYSTEM_PROMPT
            next_user = REAFINER_ERROR_ABDUCTION_USER_PROMPT.format(interaction_history=hist_str)
            extra["next_system"] = next_system
            extra["next_rag_state"] = "abduction"
            return False, next_user, 1.0, extra

        if current_phase == "abduction":
            print(f"\033[94m [instance {instance_id}] [Abduction-simple] \033[0m")
            print(f"Raw content:\n {content}")
            inst["interaction_history"].append({"phase": "abduction", "raw_response": content})

            # Tolerant: use tag if present, else use full content as error reason.
            abduction_match = re.search(r"<abduction>(.*?)</abduction>", content, re.IGNORECASE | re.DOTALL)
            if abduction_match:
                error_reason = abduction_match.group(1).strip()
            else:
                error_reason = content.strip()

            inst["error_abduction_reason"] = error_reason
            inst["refinement_phase"] = "action_generation"
            triples_string = "\n".join(inst["sorted_context"]) if inst.get("sorted_context") else ""
            next_system = REAFINER_KG_REFINEMENT_ACTION_SYSTEM_PROMPT
            next_user = REAFINER_KG_REFINEMENT_ACTION_USER_PROMPT.format(
                original_text=_format_original_chunk_for_action_prompt(inst.get("original_chunk")),
                triples_string=triples_string,
                question=inst["question"],
                error_reasons=error_reason,
            )
            extra["next_system"] = next_system
            extra["next_rag_state"] = "action_generation"
            return False, next_user, 1.0, extra

        if current_phase == "action_generation":
            print(f"\033[94m [instance {instance_id}] [Action Generation-simple] \033[0m")
            print(f"Raw content:\n {content}")
            inst["interaction_history"].append({"phase": "action", "raw_response": content})

            # Tolerant: use tag if present (including self-closing), else use full content.
            refinement_match = re.search(r"<refinement>(.*?)</refinement>", content, re.IGNORECASE | re.DOTALL)
            if refinement_match:
                actions_str = refinement_match.group(1).strip()
            elif re.search(r"<refinement\s*/>", content, re.IGNORECASE):
                actions_str = ""
            else:
                actions_str = content.strip()

            success = False
            try:
                print(f"\033[94m [instance {instance_id}] [KG Before Refine: Nodes: {len(inst['kg'].nodes())}, Edges: {len(inst['kg'].edges())}] \033[0m")
                success = self._apply_refinement_actions(instance_id, actions_str)
                print(f"\033[94m [instance {instance_id}] [KG After Refine: Nodes: {len(inst['kg'].nodes())}, Edges: {len(inst['kg'].edges())}] \033[0m")
            except Exception as e:
                print(
                    f"\033[91m [instance {instance_id}] "
                    f"[Failed to apply refinement actions (simple, exception): {e}\nContent: {content}] \033[0m"
                )
                success = False

            if not success:
                retry_count = inst.get("format_retry_count", 0)
                if retry_count < self.max_format_retry:
                    inst["format_retry_count"] = retry_count + 1
                    print(
                        f"\033[93m [instance {instance_id}] "
                        f"[Format retry {inst['format_retry_count']}/{self.max_format_retry}] \033[0m"
                    )
                    extra["next_system"] = inst["prompt_judgement_system"]
                    extra["next_rag_state"] = "action_generation"
                    extra["format_error"] = True
                    return False, _FORMAT_RETRY_PROMPT, 0.0, extra

                print(
                    f"\033[91m [instance {instance_id}] "
                    f"[Format retry exhausted ({self.max_format_retry}), fallback to original KG for RAG] \033[0m"
                )
                # Restore original KG so RAG runs on unmodified graph
                if inst.get("kg_full") is not None:
                    inst["kg"] = copy.deepcopy(inst["kg_full"])
                inst["format_retry_count"] = 0
                inst["rag_state"] = True
                extra["format_error"] = True
                extra["next_rag_state"] = "rag"
                return False, "You will perform graph based RAG based on the original knowledge graph.", 0.0, extra

            inst["format_retry_count"] = 0
            inst["rag_state"] = True
            extra["next_rag_state"] = "rag"
            return False, "You will perform graph based RAG based on your constructed knowledge graph.", 1.0, extra

        return True, "Unknown refinement phase.", 0.0, {}

    @staticmethod
    def _construct_subgraph(
        kg: DiGraph, node_list: List, initial_nodes: List, num_hop: int = 1
    ) -> DiGraph:
        """Construct a multi-hop subgraph around initial nodes (BFS). Same logic as reafiner.Reafiner._construct_subgraph."""
        subgraph = DiGraph()
        visited = set()
        node_set = set(node_list)
        queue = [(n, 0) for n in initial_nodes if n in node_set]
        for node, _ in queue:
            subgraph.add_node(node, **dict(kg.nodes[node]))
            visited.add(node)
        while queue:
            current_node, hop_count = queue.pop(0)
            if hop_count >= num_hop:
                continue
            for neighbor in sorted(kg.successors(current_node)):
                if neighbor not in visited:
                    visited.add(neighbor)
                    subgraph.add_node(neighbor, **dict(kg.nodes[neighbor]))
                    queue.append((neighbor, hop_count + 1))
                rel = kg.edges[(current_node, neighbor)].get("relation", "")
                subgraph.add_edge(current_node, neighbor, relation=rel)
            for neighbor in sorted(kg.predecessors(current_node)):
                if neighbor not in visited:
                    visited.add(neighbor)
                    subgraph.add_node(neighbor, **dict(kg.nodes[neighbor]))
                    queue.append((neighbor, hop_count + 1))
                rel = kg.edges[(neighbor, current_node)].get("relation", "")
                subgraph.add_edge(neighbor, current_node, relation=rel)
        return subgraph

    def _rank_triples_by_query_similarity(
        self,
        subgraph_triples: List[Tuple[str, str, str]],
        query: str,
        sentence_encoder: Optional[Any],
        max_items: int,
    ) -> List[Tuple[str, str, str]]:
        """Rank triples by query embedding similarity and return top-k."""
        if not subgraph_triples:
            return []
        if max_items <= 0:
            return subgraph_triples
        if not sentence_encoder or not query:
            return subgraph_triples[:max_items]

        try:
            edge_texts = [f"{s} {r} {o}" for s, r, o in subgraph_triples]
            query_emb = sentence_encoder.encode([query], query_type="edge")
            edge_embs = sentence_encoder.encode(edge_texts, query_type="edge")

            query_vec = np.asarray(query_emb)
            edge_vecs = np.asarray(edge_embs)
            if query_vec.ndim == 1:
                query_vec = query_vec.reshape(1, -1)
            if edge_vecs.ndim == 1:
                edge_vecs = edge_vecs.reshape(1, -1)

            q = query_vec[0]
            q_norm = np.linalg.norm(q) + 1e-12
            e_norm = np.linalg.norm(edge_vecs, axis=1) + 1e-12
            sims = (edge_vecs @ q) / (e_norm * q_norm)
            topk_idx = np.argsort(sims)[::-1][:max_items]
            return [subgraph_triples[int(i)] for i in topk_idx]
        except Exception as e:
            logger.warning("Failed to rank triples by query similarity: %s", e)
            return subgraph_triples[:max_items]

    @staticmethod
    def _safe_sanitize(value: Any) -> str:
        if value is None:
            return ""
        return _ILLEGAL_XML_RE.sub("", str(value))

    @staticmethod
    def _get_node_id(entity_name: str, entity_to_id: dict) -> str:
        if entity_name not in entity_to_id:
            entity_to_id[entity_name] = hashlib.sha256((entity_name + "_entity").encode()).hexdigest()
        return entity_to_id[entity_name]

    def _insert_edge(self, instance_id: str, sub: str, rel: str, obj: str) -> None:
        inst = self._instance_dict[instance_id]
        kg = inst["kg"]
        eid = inst["entity_to_id"]
        sid, oid = self._get_node_id(sub, eid), self._get_node_id(obj, eid)
        if sid not in kg.nodes:
            kg.add_node(sid, id=self._safe_sanitize(sub), type="entity")
        if oid not in kg.nodes:
            kg.add_node(oid, id=self._safe_sanitize(obj), type="entity")
        if not kg.has_edge(sid, oid):
            kg.add_edge(sid, oid, relation=self._safe_sanitize(rel))

    def _delete_edge(self, instance_id: str, sub: str, obj: str) -> None:
        inst = self._instance_dict[instance_id]
        kg, eid = inst["kg"], inst["entity_to_id"]
        sid, oid = self._get_node_id(sub, eid), self._get_node_id(obj, eid)
        if kg.has_edge(sid, oid):
            kg.remove_edge(sid, oid)

    def _replace_node(self, instance_id: str, old_entity: str, new_entity: str) -> None:
        inst = self._instance_dict[instance_id]
        kg, eid = inst["kg"], inst["entity_to_id"]
        old_id = self._get_node_id(old_entity, eid)
        if old_id not in kg.nodes:
            return
        edges_add = []
        for _u, v, d in list(kg.edges(old_id, data=True)):
            edges_add.append((new_entity, d.get("relation", ""), kg.nodes[v].get("id", str(v))))
        for u, _v, d in list(kg.in_edges(old_id, data=True)):
            edges_add.append((kg.nodes[u].get("id", str(u)), d.get("relation", ""), new_entity))
        kg.remove_node(old_id)
        for s, r, o in edges_add:
            try:
                self._insert_edge(instance_id, s, r, o)
            except Exception as e:
                logger.warning("replace_node: failed to re-insert edge (%s, %s, %s): %s", s, r, o, e)

    @staticmethod
    def _sanitize_action_text(text: str) -> str:
        """Lightweight fix-ups so common LLM typos don't cause parse failures."""
        text = text.replace("\u201c", '"').replace("\u201d", '"')  # " "
        text = text.replace("\u2018", "'").replace("\u2019", "'")  # ' '
        text = text.replace("\uff08", "(").replace("\uff09", ")")  # （ ）
        text = text.replace("\uff0c", ",")                        # ，
        text = text.replace("\u3001", ",")                        # 、
        text = text.replace("\uff5c", "|")                        # ｜
        return text

    _NON_ACTION_RE = re.compile(
        r'^(?:'
        r'\.{2,3}'             # ... or ..
        r'|\u2026'             # …
        r'|</?[a-zA-Z_][\w-]*(?:\s[^>]*)?\s*/?>'  # any XML-like tag
        r'|-+$'               # dashes
        r'|\*+$'              # asterisks
        r')$'
    )

    @classmethod
    def _is_non_action(cls, text: str) -> bool:
        """Return True if *text* is obviously not a refinement action call."""
        if cls._NON_ACTION_RE.match(text):
            return True
        if '(' not in text:
            return True
        return False

    def _apply_refinement_actions(self, instance_id: str, raw_actions: str) -> bool:
        """Apply refinement actions in a best-effort, robust way.

        Returns True when at least one action was applied OR the input is
        empty / contains only benign non-action content (e.g. ellipsis, stray
        XML tags).  Returns False only when genuinely unparseable content was
        present and zero actions succeeded.
        """
        if not raw_actions or raw_actions.isspace():
            return True

        raw_actions = self._sanitize_action_text(raw_actions)
        raw_parts = re.split(r'[\|\n]+', raw_actions.strip())

        applied = 0
        skipped = 0
        errors = 0

        for action in (a.strip() for a in raw_parts if a.strip()):
            if self._is_non_action(action):
                skipped += 1
                continue
            try:
                fn_name, args = self._parse_action_string(action)
                if fn_name == "insert_edge" and len(args) >= 3:
                    self._insert_edge(instance_id, args[0], args[1], args[2])
                    applied += 1
                elif fn_name == "delete_edge" and len(args) >= 2:
                    self._delete_edge(instance_id, args[0], args[-1])
                    applied += 1
                elif fn_name == "replace_node" and len(args) >= 2:
                    self._replace_node(instance_id, args[0], args[1])
                    applied += 1
                else:
                    errors += 1
                    logger.warning("Refinement action unsupported or wrong arity: %s (fn=%s, args=%s)", action, fn_name, args)
            except Exception as e:
                errors += 1
                logger.warning("Refinement action parse error: %s – %s", action, e)

        if applied > 0:
            return True
        if errors == 0:
            return True
        return False

    def _parse_action_string(self, action: str) -> Tuple[str, List[str]]:
        """
        Parse an action string like 'insert_edge("subject", "relation", "object")'
        Returns (function_name, [arg1, arg2, ...])
        Handles entity names containing commas, parentheses, and quotes.
        Falls back to comma-split for unquoted arguments.
        """
        action = action.strip()
        action_single_line = re.sub(r'\s+', ' ', action)
        pattern = r'(\w+)\s*\((.*)\)\s*$'
        match = re.match(pattern, action_single_line)
        if not match:
            raise ValueError(f"Invalid action format: {action}")
        function_name = match.group(1)
        args_str = match.group(2).strip()

        # --- Try quoted-string parsing first ---
        parsed_args: List[str] = []
        i = 0
        use_fallback = False
        while i < len(args_str):
            while i < len(args_str) and args_str[i] in ' \t,':
                i += 1
            if i >= len(args_str):
                break
            quote_char = args_str[i]
            if quote_char not in ('"', "'"):
                use_fallback = True
                break
            i += 1
            arg_value: List[str] = []
            while i < len(args_str):
                if args_str[i] == '\\' and i + 1 < len(args_str):
                    arg_value.append(args_str[i + 1])
                    i += 2
                elif args_str[i] == quote_char:
                    parsed_args.append(''.join(arg_value))
                    i += 1
                    break
                else:
                    arg_value.append(args_str[i])
                    i += 1
            else:
                use_fallback = True
                break

        if not use_fallback and parsed_args:
            return function_name, parsed_args

        # --- Fallback: comma-separated, strip surrounding quotes ---
        fallback_args = [
            a.strip().strip('"').strip("'").strip()
            for a in args_str.split(',')
        ]
        fallback_args = [a for a in fallback_args if a]
        if fallback_args:
            return function_name, fallback_args

        raise ValueError(f"No valid arguments found in: {action}")
    
    async def generate_response_simple(
        self, instance_id: str, retriever: EdgeRetriever, query: str, KG: nx.DiGraph, base_top_k: int = 10, sampling_params: dict = None, **kwargs
    ) -> Tuple[bool, str, float, dict]:
        """
        Generate the user response based on the assistant's output.
        Simple version.
        Just need to handle the case between refinement and RAG.
        """
        reward = 0.0
        should_terminate_sequence = False
        self._instance_dict[instance_id]["rag_state"] = False
        try:
            # need try because
            #TODO: the retrieved can be more advanced
            # perform RAG on the updated KG
            retriever = EdgeRetriever(self.retriever_config, self.llm_generator, self.reranker)
            retrieved_context = await retriever.retrieve_context(
                question=query,
                kg=KG,
                sampling_params=sampling_params,
                reward_function=self.reward_function
            )
            output = retrieved_context
            self._instance_dict[instance_id]["rag_state"] = True
            reward = 1.0
        except Exception as e:
            logger.warning(f"Failed to retrieve context: {e}\nQuery: {query}")
            should_terminate_sequence = True
            output = "Failed to retrieve context."
            reward = 0.0
            return should_terminate_sequence, output, reward, {}
        return should_terminate_sequence, output, reward, {}
    
    async def generate_response(
        self, instance_id: str, messages: List[Dict[str, Any]], **kwargs
    ) -> Tuple[bool, str, float, dict]:
        """
        Generate the user response based on the assistant's output.

        If the assistant's response includes <plan>...</plan>, ask the assistant to rewrite the query.
        Otherwise, ask the assistant to generate a response based on the current retrieved context.
        """
        should_terminate_sequence = False
        iterative = kwargs.get("iterative", True)
        content = ""
        for i in range(len(messages) - 1, -1, -1):
            item = messages[i]
            if item.get("role") == "assistant":
                content = item.get("content")
                break
        # the hierarchy is 
        # plan > answer > no plan or answer
        reward = 0.0
        # Check if the assistant's response includes <plan>...</plan>
        # give format reward
        self._instance_dict[instance_id]["rag_state"] = False
        try:
            result_json = json_repair.loads(content)
        except Exception as e:
            logger.warning(f"Failed to parse assistant content as JSON: {e}\nContent: {content}")
            should_terminate_sequence = True
            response = "The response is not in the correct format."
            reward = 0.0
            return should_terminate_sequence, response, reward, {}

        if isinstance(result_json, dict) and "answer" in result_json:
            # answer generation
            answer = result_json.get("answer", "")
            should_terminate_sequence = True
            response = "<answer>" + answer + "</answer>"
        else:
            should_terminate_sequence = True
            response = "The response is not in the correct format."
            reward = 0.0
        return should_terminate_sequence, response, reward, {}

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """
        Finalize the interaction by cleaning up the instance data.
        """
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
