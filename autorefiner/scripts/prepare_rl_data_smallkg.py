"""
prepare refinement RL training data for graph_refinement
Read queries from dataset under KGs path (e.g. KGs/hotpotqa/*.json or *.jsonl).
Each sample builds its own KG from its context via LLM extraction, then draft_answer
is computed via RAG on the sample's draft_kg.
"""
import argparse
import logging
import random
import pandas as pd
import json
import sys
import os
import asyncio
import json_repair
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

# add project path (go up to project root: autorefiner/scripts -> autorefiner -> project_root)
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from autograph.rag_server.deeprefine_prompt import (
    REAFINER_JUDGEMENT_SYSTEM_PROMPT,
    REAFINER_JUDGEMENT_USER_PROMPT,
    REAFINER_ERROR_ABDUCTION_SYSTEM_PROMPT,
    REAFINER_ERROR_ABDUCTION_USER_PROMPT,
    REAFINER_KG_REFINEMENT_ACTION_SYSTEM_PROMPT,
    REAFINER_KG_REFINEMENT_ACTION_USER_PROMPT,
)

# Import LLM and retriever for answer generation
from openai import AsyncOpenAI
from transformers import AutoTokenizer

# token-level F1 for EdgeRetriever.retrieve() reward only
from verl.third_party.autograph_r1.f1_reward import compute_f1 as token_f1
# span_check matches SGLangRollout._span_check (same normalization)
from verl.third_party.autograph_r1.gbd_reward import span_check

# Same retriever as rollout for draft_answer (EdgeRetriever), use retrieve() for RAG (topk, etc.)
from networkx import DiGraph
from autograph.rag_server.base_retriever import RetrieverConfig
from autograph.rag_server.edge_retriever import EdgeRetriever
from autograph.rag_server.reranker_api import Reranker
from autograph.rag_server.llm_api import LLMGenerator as AutographLLMGenerator

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> None:
    """Console (stdout) + optional file; tqdm 仍用 stderr，减少与进度条抢行。"""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers, force=True)


# Default KGs base path: dataset dir = KGS_BASE / dataset_name, contains dataset json/jsonl
KGS_BASE = "/data/haoyuhuang/data/AtlasTune/data/KGs"
# Default input file (single JSON) when not scanning dataset dir
DEFAULT_INPUT_FILE = "/data/haoyuhuang/data/AtlasTune/data/KGs/hotpotqa/hotpot_train_v1.1.json"
# Key in dataset JSON/JSONL for the query text (HotpotQA/MuSiQue use "question")
QUERY_KEY = "question"
# Key for context in HotpotQA format: list of [entity_name, [paragraph1, paragraph2, ...]]
CONTEXT_KEY = "context"
# Key for context in MuSiQue format: list of {title, paragraph_text, is_supporting, idx}
PARAGRAPHS_KEY = "paragraphs"

# KG extraction prompt: extract triples from document text
KG_EXTRACTION_SYSTEM_PROMPT = """You are an expert knowledge graph constructor.
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

KG_EXTRACTION_USER_PROMPT_TEMPLATE = "Extract for Document: {document}"

# Default chunk size (tokens) for splitting long documents before triple extraction
DEFAULT_CHUNK_TOKEN_SIZE = 1024

# GenAcc LLM judge: same defaults as verl SGLangRollout gen_acc_judge_* config
_DEFAULT_GEN_ACC_JUDGE_BASE_URL = "https://yunwu.ai/v1"
_DEFAULT_GEN_ACC_JUDGE_MODEL = "deepseek-v3"


def split_into_chunks(document_text, tokenizer, chunk_size=DEFAULT_CHUNK_TOKEN_SIZE):
    """
    Split document text into chunks by token count. Each chunk has at most chunk_size tokens.
    Returns list of text strings.
    """
    if not document_text or not document_text.strip():
        return []
    ids = tokenizer.encode(document_text, add_special_tokens=False)
    if len(ids) <= chunk_size:
        return [document_text]
    chunks = []
    for start in range(0, len(ids), chunk_size):
        end = min(start + chunk_size, len(ids))
        chunk_ids = ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        if chunk_text.strip():
            chunks.append(chunk_text)
    return chunks


def extract_context_from_record(record, context_key=CONTEXT_KEY):
    """
    Extract document text from context. Supports:
    - HotpotQA (context_key="context"): [[entity_name, [paragraph1, paragraph2, ...]], ...]
    - MuSiQue (context_key="paragraphs"): [{"title", "paragraph_text", "is_supporting", "idx"}, ...]
    Returns concatenated document string, or empty string if no context.
    """
    # Allow auto-detect: if context_key not in record, try MuSiQue then HotpotQA
    if context_key is None:
        if record.get(PARAGRAPHS_KEY) and isinstance(record[PARAGRAPHS_KEY], list):
            context_key = PARAGRAPHS_KEY
        else:
            context_key = CONTEXT_KEY
    raw = record.get(context_key)
    if not raw or not isinstance(raw, list):
        return ""
    # MuSiQue: list of dicts with "paragraph_text"
    if raw and isinstance(raw[0], dict) and "paragraph_text" in raw[0]:
        parts = []
        for item in raw:
            if isinstance(item, dict):
                text = item.get("paragraph_text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n\n".join(parts) if parts else ""
    # HotpotQA: [[entity_name, [paragraph1, ...]], ...]
    parts = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        paragraphs = item[1]
        if not isinstance(paragraphs, list):
            continue
        for p in paragraphs:
            if isinstance(p, str) and p.strip():
                parts.append(p.strip())
            elif p is not None:
                parts.append(str(p).strip())
    return "\n\n".join(parts) if parts else ""


async def extract_triples_from_context(llm_generator, document_text, max_new_tokens=8192, temperature=0.0):
    """
    Use LLM to extract knowledge graph triples from document text.
    Returns list of dicts [{"subject": "...", "relation": "...", "object": "..."}, ...].
    """
    if not document_text or not document_text.strip():
        return []
    messages = [
        {"role": "system", "content": KG_EXTRACTION_SYSTEM_PROMPT.strip()},
        {"role": "user", "content": KG_EXTRACTION_USER_PROMPT_TEMPLATE.format(document=document_text.strip())},
    ]
    try:
        response = await llm_generator.generate_response(
            messages, max_new_tokens=max_new_tokens, temperature=temperature, frequency_penalty=0.0
        )
        if not response or not isinstance(response, str):
            return []
        response = response.strip()
        # Try to parse JSON array
        parsed = json_repair.loads(response)
        if not isinstance(parsed, list):
            return []
        triples = []
        for item in parsed:
            if isinstance(item, dict) and "subject" in item and "relation" in item and "object" in item:
                s = str(item.get("subject", "")).strip()
                r = str(item.get("relation", "")).strip()
                o = str(item.get("object", "")).strip()
                if s and r and o:
                    triples.append({"subject": s, "relation": r, "object": o})
        return triples
    except Exception as e:
        logger.warning("Error extracting triples: %s", e)
        return []


async def extract_triples_from_context_chunked(
    llm_generator, document_text, tokenizer, chunk_size=DEFAULT_CHUNK_TOKEN_SIZE,
    max_new_tokens=8192, temperature=0.0
):
    """
    Split document into chunks by token count, extract triples from each chunk, then merge.
    Deduplicates by (subject, relation, object). Returns list of dicts.
    """
    if not document_text or not document_text.strip():
        return []
    chunks = split_into_chunks(document_text, tokenizer, chunk_size=chunk_size)
    if not chunks:
        return []
    if len(chunks) == 1:
        return await extract_triples_from_context(
            llm_generator, chunks[0], max_new_tokens=max_new_tokens, temperature=temperature
        )
    seen = set()
    merged = []
    for chunk in chunks:
        triples = await extract_triples_from_context(
            llm_generator, chunk, max_new_tokens=max_new_tokens, temperature=temperature
        )
        for t in triples:
            key = (t.get("subject", ""), t.get("relation", ""), t.get("object", ""))
            if key not in seen and all(key):
                seen.add(key)
                merged.append(t)
    return merged


def draft_kg_to_edge_str(draft_kg):
    """Convert draft_kg list of dicts to edge_str format: (subject-relation->object)."""
    if not draft_kg:
        return ""
    edge_str_lst = []
    for t in draft_kg:
        s = t.get("subject", "")
        r = t.get("relation", "")
        o = t.get("object", "")
        if s and r and o:
            edge_str_lst.append(f"({s}-{r}->{o})")
    return "\n".join(edge_str_lst)


def draft_kg_to_digraph(draft_kg):
    """
    Convert draft_kg (list of {subject, relation, object}) to networkx DiGraph.
    Node format compatible with EdgeRetriever: each node has attribute 'id' (same as node key).
    So EdgeRetriever.retrieve() can use topk retrieval instead of passing full KG to LLM.
    """
    kg = DiGraph()
    if not draft_kg:
        return kg
    for t in draft_kg:
        s = str(t.get("subject", "")).strip()
        r = str(t.get("relation", "")).strip()
        o = str(t.get("object", "")).strip()
        if not s or not r or not o:
            continue
        kg.add_edge(s, o, relation=r)
    for n in kg.nodes:
        kg.nodes[n]["id"] = n
    return kg


def load_queries_from_file(file_path, query_key=QUERY_KEY, max_records=None):
    """
    Load query records from a single JSON or JSONL file.
    Each record should have query_key (e.g. "question"). Other keys are kept.

    Args:
        file_path: Path to .json or .jsonl file
        query_key: Key for query string in each record (default "question")
        max_records: If set, yield at most this many records (for format check / small run).

    Yields:
        dict per record: at least {query_key: str}, plus any other keys from json
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        if path.suffix.lower() == ".jsonl":
            for line in f:
                if max_records is not None and n >= max_records:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if query_key not in rec or not rec[query_key]:
                    continue
                n += 1
                yield rec
        else:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                return
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = data.get("data", data.get("questions", data.get("instances", [data])))
                if not isinstance(items, list):
                    items = [items]
            else:
                return
            for rec in items:
                if max_records is not None and n >= max_records:
                    break
                if not isinstance(rec, dict) or query_key not in rec or not rec[query_key]:
                    continue
                n += 1
                yield rec


def load_queries_from_dataset(dataset_dir, query_key=QUERY_KEY, split=None, max_records=None):
    """
    Load query records from a dataset directory under KGs.
    Scans for *.json and *.jsonl; each record should have query_key (e.g. "question").
    Other keys (e.g. answer, id) are kept in the record for extra_info/ground_truth.

    Args:
        dataset_dir: Path to dataset dir, e.g. KGs/hotpotqa
        query_key: Key for query string in each record (default "question")
        split: If set, only load files whose name contains this (e.g. "dev", "train")
        max_records: If set, yield at most this many records (for format check / small run).

    Yields:
        dict per record: at least {query_key: str}, plus any other keys from json
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")
    n = 0
    for ext in ("*.json", "*.jsonl"):
        for path in sorted(dataset_dir.glob(ext)):
            if max_records is not None and n >= max_records:
                return
            if split and split not in path.name.lower():
                continue
            with open(path, "r", encoding="utf-8") as f:
                if path.suffix == ".jsonl":
                    for line in f:
                        if max_records is not None and n >= max_records:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if query_key not in rec or not rec[query_key]:
                            continue
                        n += 1
                        yield rec
                elif path.suffix == ".json":
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(data, list):
                        items = data
                    elif isinstance(data, dict):
                        items = data.get("data", data.get("questions", data.get("instances", [data])))
                        if not isinstance(items, list):
                            items = [items]
                    else:
                        continue
                    for rec in items:
                        if max_records is not None and n >= max_records:
                            break
                        if not isinstance(rec, dict) or query_key not in rec or not rec[query_key]:
                            continue
                        n += 1
                        yield rec


def find_original_kg_pkl(dataset_dir):
    """Return path to original_kg.pkl under dataset_dir (direct or in one subdir)."""
    dataset_dir = Path(dataset_dir)
    direct = dataset_dir / "original_kg.pkl"
    if direct.exists():
        return str(direct)
    for sub in dataset_dir.iterdir():
        if sub.is_dir():
            p = sub / "original_kg.pkl"
            if p.exists():
                return str(p)
    return str(direct)


def format_triples_string(triples):
    """format triples list to string"""
    if isinstance(triples, list):
        return json.dumps(triples, ensure_ascii=False, indent=2)
    elif isinstance(triples, str):
        try:
            # try to parse as JSON, if successful, format it
            parsed = json.loads(triples)
            return json.dumps(parsed, ensure_ascii=False, indent=2)
        except:
            return triples
    return str(triples)


def build_judgement_prompt(question, triples_string):
    """
    Build the first answerable judgement prompt with retrieved subgraph
    
    Args:
        question: The question string
        triples_string: Formatted triples string from retrieved subgraph
    
    Returns:
        List of messages for the judgement prompt
    """
    prompt = [
        {
            "role": "system",
            "content": REAFINER_JUDGEMENT_SYSTEM_PROMPT.strip()
        },
        {
            "role": "user",
            "content": REAFINER_JUDGEMENT_USER_PROMPT.format(
                question=question,
                triples_string=triples_string
            ).strip()
        }
    ]
    return prompt


def create_gen_acc_judge_generator(
    base_url=None,
    api_key=None,
    model_name=None,
):
    """Async OpenAI-compatible client for GenAcc judge (same role as SGLangRollout.gen_acc_judge_generator)."""
    base_url = base_url or os.environ.get(
        "GEN_ACC_JUDGE_BASE_URL", _DEFAULT_GEN_ACC_JUDGE_BASE_URL
    )
    api_key = (
        api_key
        if api_key is not None
        else os.environ.get("GEN_ACC_JUDGE_API_KEY", "EMPTY KEY")
    )
    model_name = model_name or os.environ.get(
        "GEN_ACC_JUDGE_MODEL", _DEFAULT_GEN_ACC_JUDGE_MODEL
    )
    client = AsyncOpenAI(base_url=base_url.rstrip("/"), api_key=api_key)
    return AutographLLMGenerator(client, model_name, backend="openai")


async def judge_check_async(judge_generator, prediction, gold_answers):
    """Same prompt / parsing as SGLangRollout._judge_check_async."""
    gold_str = ", ".join('"' + str(a) + '"' for a in gold_answers)
    prompt = (
        "Given the following prediction and set of gold answers, determine if the "
        "prediction contains or is semantically equivalent to any of the gold answers.\n\n"
        'Prediction: "' + str(prediction) + '"\n'
        "Gold Answers: " + gold_str + "\n\n"
        "Does the prediction contain any of the gold answers? "
        "Answer with ONLY 'Yes' or 'No'."
    )
    try:
        text = await judge_generator.generate_response(
            [{"role": "user", "content": prompt}],
            max_new_tokens=10,
            temperature=0.0,
            return_text_only=True,
        )
        if isinstance(text, str):
            return text.strip().lower().startswith("yes")
        return False
    except Exception as e:
        logger.warning("[judge_check_async] LLM call failed: %s", e)
        return False


async def compute_draft_gen_acc_async(prediction, gold_answers, judge_generator):
    """Same logic as SGLangRollout._compute_gen_acc_async: span_check | async judge. Returns 1.0 or 0.0."""
    if prediction is None or not str(prediction).strip():
        return 0.0
    if not gold_answers:
        return 0.0
    if isinstance(gold_answers, str):
        gold_answers = [gold_answers]
    gold_list = [str(a).strip() for a in gold_answers if a is not None and str(a).strip()]
    if not gold_list:
        return 0.0
    if span_check(prediction, gold_list):
        return 1.0
    if judge_generator is None:
        return 0.0
    if await judge_check_async(judge_generator, prediction, gold_list):
        return 1.0
    return 0.0


async def recompute_draft_gen_acc_for_rows(rows, judge_generator, max_concurrent=16, force=False):
    """
    Fill extra_info.interaction_kwargs.draft_gen_acc for each row (async judge when span misses).
    Skips rows that already have draft_gen_acc unless force=True.
    """
    if not rows:
        return
    sem = asyncio.Semaphore(max_concurrent)

    async def _one(r):
        async with sem:
            extra = r.get("extra_info", {}) or {}
            if not isinstance(extra, dict):
                return
            ik = extra.get("interaction_kwargs", {}) or {}
            if not isinstance(ik, dict):
                ik = {}
            if not force and ik.get("draft_gen_acc", None) is not None:
                return
            draft = ik.get("draft_answer", "")
            targets = ik.get("ground_truth", [])
            if isinstance(targets, str):
                targets = [targets]
            if not isinstance(targets, list):
                targets = []
            g = await compute_draft_gen_acc_async(draft, targets, judge_generator)
            ik = dict(ik)
            ik["draft_gen_acc"] = float(g)
            extra = dict(extra)
            extra["interaction_kwargs"] = ik
            r["extra_info"] = extra

    await asyncio.gather(*[_one(r) for r in rows])


async def process_row(record, row_index, llm_generator, tokenizer, query_key=QUERY_KEY, context_key=CONTEXT_KEY,
                     edge_retriever=None, full_graph_data_path=None, chunk_size=DEFAULT_CHUNK_TOKEN_SIZE,
                     gen_acc_judge_generator=None):
    """
    Process a single sample: extract draft_kg from context via LLM (chunked by token), build judgement prompt,
    compute draft_answer via RAG on draft_kg.
    """
    # Support both parquet-style (extra_info.question) and dataset-style (top-level question/query_key)
    question = record.get(query_key) or (isinstance(record.get("extra_info"), dict) and record["extra_info"].get("question")) or ""
    if isinstance(record.get("extra_info"), str):
        try:
            ei = json.loads(record["extra_info"])
            question = question or ei.get("question", "")
        except Exception:
            pass
    if not question:
        return None

    # Extract document from context (HotpotQA format)
    document_text = extract_context_from_record(record, context_key=context_key)
    if not document_text:
        return None

    # Extract triples from context via LLM (split by chunk_size tokens, then merge)
    draft_kg = await extract_triples_from_context_chunked(
        llm_generator, document_text, tokenizer, chunk_size=chunk_size
    )
    # 如果抽取 KG 失败或为空，则认为该 sample 处理失败，直接跳过
    if not draft_kg:
        return None
    triples_string = format_triples_string(draft_kg)
    judgement_prompt = build_judgement_prompt(question, triples_string)

    interaction_kwargs = {
        "name": "graph_refinement",
        "question": question,
        "draft_kg": draft_kg,  # [{"subject": "...", "relation": "...", "object": "..."}, ...]
        "original_chunk": document_text,  # 抽图所用的原始上下文全文（与 chunk 切分前一致）
    }
    if full_graph_data_path:
        interaction_kwargs["full_graph_data_path"] = full_graph_data_path

    # Draft answer: RAG via EdgeRetriever.retrieve() (与 sglang_rollout autorefine 一致：topk 检索再生成，不把整图给 LLM)
    if edge_retriever is not None:
        try:
            draft_kg_digraph = draft_kg_to_digraph(draft_kg)
            # sampling_params 与 rollout _handle_local_rag_engine_call 一致
            api_sampling_params = {
                "max_new_tokens": 512,
                "temperature": 0,
                "frequency_penalty": 0.0,
                "return_logprob": False,
            }
            result = await edge_retriever.retrieve(
                question=question,
                kg=draft_kg_digraph,
                sampling_params=api_sampling_params,
                reward_function=token_f1,
            )
            draft_answer = ""
            if isinstance(result, str):
                # EdgeRetriever currently returns a JSON string {"answer": "..."}.
                # Keep robust fallback to raw text to avoid extra failures.
                try:
                    out = json.loads(result) if result.strip() else {}
                except Exception:
                    out = {}
                    draft_answer = result
            else:
                out = result or {}
            if isinstance(out, dict):
                draft_answer = out.get("answer", draft_answer)
            elif not draft_answer and out is not None:
                draft_answer = str(out)
            if not isinstance(draft_answer, str):
                draft_answer = str(draft_answer) if draft_answer is not None else ""
            interaction_kwargs["draft_answer"] = draft_answer
        except Exception as e:
            logger.warning("draft_answer failed for row %s: %s", row_index, e)
            # RAG 失败也视为该 sample 处理失败，跳过，不写入结果
            return None

    # ground_truth: from parquet extra_info, or dataset json (answer / answers). Normalize to list for interaction_kwargs and reward.
    old_interaction_kwargs = {}
    if isinstance(record.get("extra_info"), dict):
        old_interaction_kwargs = record["extra_info"].get("interaction_kwargs", {})
    if isinstance(old_interaction_kwargs, dict) and "supporting_context" in old_interaction_kwargs:
        interaction_kwargs["supporting_context"] = old_interaction_kwargs["supporting_context"]

    gt_raw = None
    if isinstance(old_interaction_kwargs, dict) and "ground_truth" in old_interaction_kwargs:
        gt_raw = old_interaction_kwargs["ground_truth"]
    if gt_raw is None and "answer" in record:
        gt_raw = record["answer"]
    if gt_raw is None and "answers" in record:
        gt_raw = record["answers"]
    if isinstance(gt_raw, list):
        gt_list = [str(x).strip() for x in gt_raw if x is not None]
    elif gt_raw is not None:
        gt_list = [str(gt_raw).strip()]
    else:
        gt_list = []
    # MuSiQue: merge answer_aliases into ground_truth
    aliases = record.get("answer_aliases")
    if isinstance(aliases, list):
        for a in aliases:
            if a is not None and str(a).strip() and str(a).strip() not in gt_list:
                gt_list.append(str(a).strip())
    elif isinstance(aliases, str) and aliases.strip() and aliases.strip() not in gt_list:
        gt_list.append(aliases.strip())
    if not gt_list:
        gt_list = [""]  # avoid rollout .get("ground_truth")[0] IndexError
    interaction_kwargs["ground_truth"] = gt_list
    # GenAcc = span | async LLM judge (same as SGLangRollout._compute_gen_acc_async)
    interaction_kwargs["draft_gen_acc"] = await compute_draft_gen_acc_async(
        interaction_kwargs.get("draft_answer"), gt_list, gen_acc_judge_generator
    )

    prompt_templates = {
        "prompt_template_judgement": {
            "system": REAFINER_JUDGEMENT_SYSTEM_PROMPT.strip(),
            "user": REAFINER_JUDGEMENT_USER_PROMPT.strip()
        },
        "prompt_template_abduction": {
            "system": REAFINER_ERROR_ABDUCTION_SYSTEM_PROMPT.strip(),
            "user": REAFINER_ERROR_ABDUCTION_USER_PROMPT.strip()
        },
        "prompt_template_action": {
            "system": REAFINER_KG_REFINEMENT_ACTION_SYSTEM_PROMPT.strip(),
            "user": REAFINER_KG_REFINEMENT_ACTION_USER_PROMPT.strip()
        }
    }
    interaction_kwargs.update(prompt_templates)

    new_extra_info = {
        "index": str(row_index),
        "need_tools_kwargs": record.get("need_tools_kwargs", False),
        "question": question,
        "split": record.get("split", "train"),
        "interaction_kwargs": interaction_kwargs,
        **prompt_templates
    }

    reward_model = record.get("reward_model")
    if isinstance(reward_model, str):
        try:
            reward_model = json.loads(reward_model)
        except Exception:
            reward_model = {}
    if not isinstance(reward_model, dict):
        reward_model = {}
    # Reward manager reads reward_model["ground_truth"]; f1_reward expects ground_truth["target"] as list
    reward_model["ground_truth"] = {"target": gt_list}

    return {
        "data_source": record.get("data_source", "graph_refinement"),
        "prompt": judgement_prompt,
        "ability": "graph_refinement",
        "reward_model": reward_model,
        "extra_info": new_extra_info,
        "metadata": record.get("metadata"),
    }


def _get_draft_gt_from_row(r):
    extra = r.get("extra_info", {}) or {}
    ik = extra.get("interaction_kwargs", {}) or {}
    draft = ik.get("draft_answer", "")
    targets = ik.get("ground_truth", [])
    if isinstance(targets, str):
        targets = [targets]
    if not isinstance(targets, list):
        targets = []
    return draft, targets, extra, ik


def filter_rows_by_gen_acc(rows, gen_acc_threshold: float = 1.0):
    """
    Drop rows with draft_gen_acc >= gen_acc_threshold (default 1.0 removes GenAcc==1).
    Requires draft_gen_acc on rows (run recompute_draft_gen_acc_for_rows first if missing).
    If gen_acc_threshold > 1.0 (e.g. 1.1), never removes; still normalizes stored draft_gen_acc when present.
    """
    if not rows:
        return rows, 0

    kept = []
    removed = 0
    for r in rows:
        draft, targets, extra, ik = _get_draft_gt_from_row(r)

        g = None
        try:
            if isinstance(ik, dict) and ik.get("draft_gen_acc", None) is not None:
                g = float(ik.get("draft_gen_acc"))
        except Exception:
            g = None
        if g is None:
            # Span-only sync fallback (no async judge); avoids silent wrong filter if recompute was skipped
            gold_list = [str(t).strip() for t in targets if t is not None and str(t).strip()]
            g = 1.0 if (draft and gold_list and span_check(draft, gold_list)) else 0.0

        try:
            if isinstance(extra, dict):
                if "interaction_kwargs" not in extra or not isinstance(extra.get("interaction_kwargs"), dict):
                    extra["interaction_kwargs"] = ik if isinstance(ik, dict) else {}
                extra["interaction_kwargs"]["draft_gen_acc"] = float(g)
                r["extra_info"] = extra
        except Exception:
            pass

        if gen_acc_threshold <= 1.0 and g >= gen_acc_threshold:
            removed += 1
            continue
        kept.append(r)

    return kept, removed


def sort_rows_by_draft_gen_acc_desc(rows):
    """Sort by draft_gen_acc descending; missing -> -1."""
    def _score(r):
        try:
            extra = r.get("extra_info", {}) or {}
            ik = extra.get("interaction_kwargs", {}) or {}
            v = ik.get("draft_gen_acc", None)
            if v is None:
                return -1.0
            return float(v)
        except Exception:
            return -1.0

    return sorted(rows, key=_score, reverse=True)


def count_rows_meet_gen_acc_threshold(rows, gen_acc_threshold: float):
    """Count rows with draft_gen_acc < gen_acc_threshold (kept under default filter)."""
    cnt = 0
    for r in rows:
        try:
            _, targets, _, ik = _get_draft_gt_from_row(r)
            v = ik.get("draft_gen_acc", None) if isinstance(ik, dict) else None
            if v is None:
                draft = ik.get("draft_answer", "") if isinstance(ik, dict) else ""
                gold_list = [str(t).strip() for t in targets if t is not None and str(t).strip()]
                v = 1.0 if (draft and gold_list and span_check(draft, gold_list)) else 0.0
            if float(v) < gen_acc_threshold:
                cnt += 1
        except Exception:
            pass
    return cnt


async def process_batch(rows_batch, llm_generator, tokenizer, semaphore=None, full_graph_data_path=None,
                        query_key=QUERY_KEY, context_key=CONTEXT_KEY, edge_retriever=None,
                        chunk_size=DEFAULT_CHUNK_TOKEN_SIZE, gen_acc_judge_generator=None):
    """Process a batch of records asynchronously. Each item in rows_batch is (idx, record)."""
    tasks = []
    indices = []
    for idx, row in rows_batch:
        indices.append(idx)
        if semaphore:
            async def process_with_semaphore(r=row, i=idx):
                async with semaphore:
                    return await process_row(
                        r, i, llm_generator, tokenizer, query_key=query_key, context_key=context_key,
                        edge_retriever=edge_retriever, full_graph_data_path=full_graph_data_path,
                        chunk_size=chunk_size, gen_acc_judge_generator=gen_acc_judge_generator,
                    )
            tasks.append(process_with_semaphore())
        else:
            tasks.append(
                process_row(
                    row, idx, llm_generator, tokenizer, query_key=query_key, context_key=context_key,
                    edge_retriever=edge_retriever, full_graph_data_path=full_graph_data_path,
                    chunk_size=chunk_size, gen_acc_judge_generator=gen_acc_judge_generator,
                )
            )

    results = await asyncio.gather(*tasks, return_exceptions=True)
    processed_rows = []
    failed_count = 0
    for idx, result in zip(indices, results):
        if isinstance(result, Exception):
            logger.error(
                "Error processing row %s: %s",
                idx,
                result,
                exc_info=(type(result), result, result.__traceback__),
            )
            failed_count += 1
        elif result is None:
            failed_count += 1
        else:
            processed_rows.append(result)
    return processed_rows, failed_count


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare refinement RL data from KGs dataset dir")
    parser.add_argument("--input-file", type=str, default=DEFAULT_INPUT_FILE,
                        help="Path to a single JSON/JSONL file to load queries from. Default: hotpot_train_v1.1.json under KGs/hotpotqa.")
    parser.add_argument("--kgs-base", type=str, default=KGS_BASE, help="Base path for KGs (e.g. .../data/KGs)")
    parser.add_argument("--dataset", type=str, default="hotpotqa", help="Dataset name under kgs_base (e.g. hotpotqa)")
    parser.add_argument("--split", type=str, default=None, help="Only load files containing this when using dataset dir (e.g. dev, train). Ignored if --input-file is used.")
    parser.add_argument("--output", type=str, default=None, help="Output parquet path. Default: {kgs_base}/{dataset}/{dataset}_{split}_refinement.parquet")
    parser.add_argument("--query-key", type=str, default=QUERY_KEY, help="Key for query text in dataset json (default: question)")
    parser.add_argument("--batch-size", type=int, default=16, help="Process batch size")
    parser.add_argument("--max-concurrent", type=int, default=16, help="Max concurrent LLM requests (extraction + draft_answer)")
    parser.add_argument("--no-draft-answer", action="store_true", help="Do not compute draft_answer via RAG on draft_kg")
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Force reprocess (re-extract draft_kg / recompute draft_answer) even if *_smallkg_all.parquet already exists.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of data for train (rest for valid). Set to 0 to disable train/valid split (single output).",
    )
    parser.add_argument(
        "--gen-acc-threshold",
        type=float,
        default=1.0,
        help="Filter out rows where draft GenAcc (span|async judge, same as SGLangRollout) >= this value. "
        "Default 1.0 drops drafts already correct. Use >1.0 (e.g. 1.1) for score-only / no removal.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/valid split and for sampling")
    parser.add_argument("--max-samples", type=int, default=1000,
                        help="Target max number of filtered samples to keep. During processing, stop early once reached. Set to 0 to process all.")
    parser.add_argument("--context-key", type=str, default=None,
                        help="Key for context in dataset (default: context for hotpotqa, paragraphs for musique)")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_TOKEN_SIZE,
                        help="Max tokens per chunk when extracting triples from long documents (default: 1024)")
    parser.add_argument("--max-load", type=int, default=0,
                        help="Only load first N records for format check or small run (0 = load all)")
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Append logs to this file (UTF-8) in addition to stdout.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG level logs")
    return parser.parse_args()


def create_llm_generator_for_extraction(llm_base_url="http://0.0.0.0:8129/v1", llm_model_name="Qwen/Qwen2.5-7B-Instruct"):
    """Create LLM generator for triple extraction (async)."""
    llm_client = AsyncOpenAI(base_url=llm_base_url, api_key="EMPTY KEY")
    return AutographLLMGenerator(llm_client, llm_model_name, backend="openai")


async def main_async():
    args = parse_args()
    setup_logging(verbose=bool(getattr(args, "verbose", False)), log_file=getattr(args, "log_file", None))
    dataset_dir = Path(args.kgs_base) / args.dataset
    out_dir = Path(args.kgs_base) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ratio = getattr(args, "train_ratio", 0.8)
    gen_acc_threshold = getattr(args, "gen_acc_threshold", 1.0)
    force_reprocess = bool(getattr(args, "force_reprocess", False))
    seed = getattr(args, "seed", 42)
    max_samples = getattr(args, "max_samples", 1000)
    max_concurrent = getattr(args, "max_concurrent", 16)

    # Fast path: reuse previously saved FULL train/valid parquet and only apply GenAcc filter (async judge when needed).
    # This avoids re-running expensive LLM extraction / draft_answer computation.
    if train_ratio > 0 and train_ratio < 1 and not force_reprocess:
        full_train_file = out_dir / f"{args.dataset}_train_refinement_gbd_smallkg_all.parquet"
        full_valid_file = out_dir / f"{args.dataset}_valid_refinement_gbd_smallkg_all.parquet"
        if full_train_file.exists() and full_valid_file.exists():
            logger.info(
                "Found existing FULL data:\n  - %s\n  - %s\n"
                "Skip reprocess; recompute draft_gen_acc (span|async judge) where missing; "
                "filter with gen_acc_threshold=%s.",
                full_train_file,
                full_valid_file,
                gen_acc_threshold,
            )

            train_rows = pd.read_parquet(str(full_train_file)).to_dict(orient="records")
            valid_rows = pd.read_parquet(str(full_valid_file)).to_dict(orient="records")
            all_rows = train_rows + valid_rows
            random.seed(seed)
            random.shuffle(all_rows)

            logger.info("Initializing GenAcc judge client (same stack as SGLangRollout)...")
            gen_acc_judge_gen = create_gen_acc_judge_generator()
            await recompute_draft_gen_acc_for_rows(
                all_rows, gen_acc_judge_gen, max_concurrent=max_concurrent, force=False
            )
            filtered_rows, removed_total = filter_rows_by_gen_acc(all_rows, gen_acc_threshold=gen_acc_threshold)
            if max_samples > 0:
                keep_n = min(len(filtered_rows), max_samples)
                filtered_rows = filtered_rows[:keep_n]
                if keep_n < max_samples:
                    logger.info(
                        "Filtered rows (%s) < --max-samples (%s); use all available filtered samples before split.",
                        keep_n,
                        max_samples,
                    )
                else:
                    logger.info("Keep top %s filtered samples before split.", max_samples)
            else:
                logger.info("Using all filtered samples (%s) before split.", len(filtered_rows))

            n_train = max(1, int(len(filtered_rows) * train_ratio))
            train_rows = filtered_rows[:n_train]
            valid_rows = filtered_rows[n_train:]
            for r in train_rows:
                if "extra_info" in r and isinstance(r["extra_info"], dict):
                    r["extra_info"]["split"] = "train"
            for r in valid_rows:
                if "extra_info" in r and isinstance(r["extra_info"], dict):
                    r["extra_info"]["split"] = "valid"
            train_rows = sort_rows_by_draft_gen_acc_desc(train_rows)
            valid_rows = sort_rows_by_draft_gen_acc_desc(valid_rows)

            train_file = out_dir / f"{args.dataset}_train_refinement_gbd_smallkg.parquet"
            valid_file = out_dir / f"{args.dataset}_valid_refinement_gbd_smallkg.parquet"
            pd.DataFrame(train_rows).to_parquet(str(train_file), index=False)
            pd.DataFrame(valid_rows).to_parquet(str(valid_file), index=False)

            logger.info("Saved FILTERED train (%s rows) to %s", len(train_rows), train_file)
            logger.info("Saved FILTERED valid (%s rows) to %s", len(valid_rows), valid_file)
            logger.info(
                "Statistics: total rows after filter: %s (removed %s with draft_gen_acc>=%s)",
                len(filtered_rows),
                removed_total,
                gen_acc_threshold,
            )
            return

    if args.output and train_ratio <= 0:
        output_file = args.output
    elif not args.output:
        suffix = f"_{args.split}" if args.split else ""
        output_file = str(out_dir / f"{args.dataset}{suffix}_refinement.parquet")
    else:
        output_file = None

    batch_size = args.batch_size

    # Context key: MuSiQue uses "paragraphs", HotpotQA uses "context"
    context_key = getattr(args, "context_key", None) or (
        PARAGRAPHS_KEY if args.dataset == "musique" else CONTEXT_KEY
    )
    max_load = getattr(args, "max_load", 0) or None  # 0 -> load all

    # Load query records: from --input-file if set and exists, else from dataset dir
    if getattr(args, "input_file", None) and Path(args.input_file).exists():
        logger.info(
            "Loading queries from input file: %s%s",
            args.input_file,
            f" (max {max_load} records)" if max_load else "",
        )
        records = list(load_queries_from_file(args.input_file, query_key=args.query_key, max_records=max_load))
    else:
        logger.info(
            "Loading queries from dataset dir: %s (split=%s)%s",
            dataset_dir,
            args.split,
            f", max_load={max_load}" if max_load else "",
        )
        records = list(load_queries_from_dataset(dataset_dir, query_key=args.query_key, split=args.split, max_records=max_load))
    # Filter to records that have context
    records_with_context = [r for r in records if extract_context_from_record(r, context_key=context_key)]
    logger.info("Loaded %s query records, %s have context", len(records), len(records_with_context))

    if not records_with_context:
        logger.warning("No records with context found. Exiting.")
        return

    llm_base_url = os.getenv("LLM_BASE_URL", "http://0.0.0.0:8129/v1")
    encoder_base_url = os.getenv("ENCODER_BASE_URL", "http://0.0.0.0:8128/v1")
    llm_model_name = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
    chunk_size = getattr(args, "chunk_size", DEFAULT_CHUNK_TOKEN_SIZE)

    # Tokenizer for chunking documents (same as extraction LLM for consistent token count)
    logger.info("Loading tokenizer for chunking (model=%s, chunk_size=%s)...", llm_model_name, chunk_size)
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=True)

    # LLM for triple extraction
    logger.info("Initializing LLM for KG extraction...")
    llm_generator = create_llm_generator_for_extraction(llm_base_url=llm_base_url, llm_model_name=llm_model_name)

    logger.info(
        "Initializing GenAcc judge (span | async LLM), same as SGLangRollout._compute_gen_acc_async..."
    )
    gen_acc_judge_generator = create_gen_acc_judge_generator()

    # EdgeRetriever for draft_answer: 与 rollout 一致用 retrieve() 做 RAG（topk 检索 + 生成，不把整图给 LLM）
    edge_retriever = None
    full_graph_data_path = None
    if not getattr(args, "no_draft_answer", False):
        logger.info(
            "Initializing EdgeRetriever for draft_answer (RAG via retrieve(), same as sglang_rollout)..."
        )
        emb_client = AsyncOpenAI(base_url=encoder_base_url, api_key="EMPTY KEY")
        reranker = Reranker(emb_client,model_name="Qwen/Qwen3-Embedding-0.6B")
        llm_client = AsyncOpenAI(base_url=llm_base_url, api_key="EMPTY KEY")
        llm_gen = AutographLLMGenerator(llm_client, llm_model_name, backend="openai")
        config = RetrieverConfig("re_edge")
        edge_retriever = EdgeRetriever(config, llm_gen, reranker)
        # Optional: full_graph_data_path if original_kg.pkl exists
        pkl_path = find_original_kg_pkl(dataset_dir)
        if os.path.exists(pkl_path):
            full_graph_data_path = pkl_path

    semaphore = asyncio.Semaphore(max_concurrent)
    rows_list = [(i, rec) for i, rec in enumerate(records_with_context)]
    batches = [rows_list[i:i + batch_size] for i in range(0, len(rows_list), batch_size)]
    logger.info(
        "Processing %s batches with batch_size=%s, max_concurrent=%s",
        len(batches),
        batch_size,
        max_concurrent,
    )
    logger.info(
        "Each sample: chunk doc by %s tokens -> extract draft_kg per chunk -> merge -> draft_answer via RAG on draft_kg",
        chunk_size,
    )

    all_processed_rows = []
    total_failed = 0
    qualified_count = 0
    stop_early = False
    pbar = tqdm(batches, desc="Processing batches")
    for batch_idx, batch in enumerate(pbar):
        processed_rows, failed_count = await process_batch(
            batch, llm_generator, tokenizer, semaphore,
            full_graph_data_path=full_graph_data_path,
            query_key=args.query_key,
            context_key=context_key,
            edge_retriever=edge_retriever,
            chunk_size=chunk_size,
            gen_acc_judge_generator=gen_acc_judge_generator,
        )
        all_processed_rows.extend(processed_rows)
        total_failed += failed_count
        batch_qualified = count_rows_meet_gen_acc_threshold(processed_rows, gen_acc_threshold=gen_acc_threshold)
        qualified_count += batch_qualified
        logger.info(
            "Batch %s/%s: Processed %s rows, Failed %s rows",
            batch_idx + 1,
            len(batches),
            len(processed_rows),
            failed_count,
        )
        logger.info(
            "  Qualified (draft_gen_acc < %s): +%s, total=%s",
            gen_acc_threshold,
            batch_qualified,
            qualified_count,
        )
        target_display = max_samples if max_samples > 0 else "all"
        pbar.set_postfix(
            qualified=qualified_count,
            target=target_display,
            processed=len(all_processed_rows),
            failed=total_failed,
        )
        if max_samples > 0 and qualified_count >= max_samples:
            stop_early = True
            logger.info(
                "Reached --max-samples target (%s) with qualified samples. Stop further processing early.",
                max_samples,
            )
            break
    pbar.close()

    logger.info(
        "Processed %s rows successfully%s",
        len(all_processed_rows),
        " (early stopped)" if stop_early else "",
    )
    logger.info("Failed %s rows", total_failed)

    random.seed(seed)
    shuffled_all = list(all_processed_rows)
    random.shuffle(shuffled_all)

    filtered_rows, removed_total = filter_rows_by_gen_acc(shuffled_all, gen_acc_threshold=gen_acc_threshold)
    if max_samples > 0:
        keep_n = min(len(filtered_rows), max_samples)
        filtered_rows = filtered_rows[:keep_n]
        if keep_n < max_samples:
            logger.info(
                "Filtered rows (%s) < --max-samples (%s); use all available filtered samples before split.",
                keep_n,
                max_samples,
            )
        else:
            logger.info("Keep top %s filtered samples before split.", max_samples)
    else:
        logger.info("Using all filtered samples (%s) before split.", len(filtered_rows))

    if train_ratio > 0 and train_ratio < 1 and all_processed_rows:
        n_train_all = max(1, int(len(shuffled_all) * train_ratio))
        full_train_rows = shuffled_all[:n_train_all]
        full_valid_rows = shuffled_all[n_train_all:]
        for r in full_train_rows:
            if "extra_info" in r and isinstance(r["extra_info"], dict):
                r["extra_info"]["split"] = "train"
        for r in full_valid_rows:
            if "extra_info" in r and isinstance(r["extra_info"], dict):
                r["extra_info"]["split"] = "valid"

        n_train = max(1, int(len(filtered_rows) * train_ratio))
        train_rows = filtered_rows[:n_train]
        valid_rows = filtered_rows[n_train:]
        for r in train_rows:
            if "extra_info" in r and isinstance(r["extra_info"], dict):
                r["extra_info"]["split"] = "train"
        for r in valid_rows:
            if "extra_info" in r and isinstance(r["extra_info"], dict):
                r["extra_info"]["split"] = "valid"

        out_dir = Path(args.kgs_base) / args.dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        full_train_file = out_dir / f"{args.dataset}_train_refinement_gbd_smallkg_all.parquet"
        full_valid_file = out_dir / f"{args.dataset}_valid_refinement_gbd_smallkg_all.parquet"

        _train_all, _ = filter_rows_by_gen_acc(full_train_rows, gen_acc_threshold=1.1)
        _valid_all, _ = filter_rows_by_gen_acc(full_valid_rows, gen_acc_threshold=1.1)
        _train_all = sort_rows_by_draft_gen_acc_desc(_train_all)
        _valid_all = sort_rows_by_draft_gen_acc_desc(_valid_all)
        pd.DataFrame(_train_all).to_parquet(str(full_train_file), index=False)
        pd.DataFrame(_valid_all).to_parquet(str(full_valid_file), index=False)

        # filtered_rows has already been filtered and size-capped before split.
        train_rows = sort_rows_by_draft_gen_acc_desc(train_rows)
        valid_rows = sort_rows_by_draft_gen_acc_desc(valid_rows)

        train_file = out_dir / f"{args.dataset}_train_refinement_gbd_smallkg.parquet"
        valid_file = out_dir / f"{args.dataset}_valid_refinement_gbd_smallkg.parquet"
        pd.DataFrame(train_rows).to_parquet(str(train_file), index=False)
        pd.DataFrame(valid_rows).to_parquet(str(valid_file), index=False)
        logger.info(
            "Saved FULL train (no GenAcc filter, %s rows) to %s",
            len(full_train_rows),
            full_train_file,
        )
        logger.info(
            "Saved FULL valid (no GenAcc filter, %s rows) to %s",
            len(full_valid_rows),
            full_valid_file,
        )
        logger.info("Saved FILTERED train (%s rows) to %s", len(train_rows), train_file)
        logger.info("Saved FILTERED valid (%s rows) to %s", len(valid_rows), valid_file)
        logger.info(
            "Filtered pool size before split: %s (removed %s with draft_gen_acc>=%s)",
            len(filtered_rows),
            removed_total,
            gen_acc_threshold,
        )
        logger.info(
            "Train/valid ratio: %s / %s (seed=%s)",
            f"{train_ratio:.0%}",
            f"{1 - train_ratio:.0%}",
            seed,
        )
    else:
        # For single-output mode, save BOTH full (contains unqualified rows) and filtered rows.
        full_all_rows, _ = filter_rows_by_gen_acc(shuffled_all, gen_acc_threshold=1.1)
        full_all_rows = sort_rows_by_draft_gen_acc_desc(full_all_rows)
        filtered_sorted = sort_rows_by_draft_gen_acc_desc(filtered_rows)

        if output_file:
            out_path = output_file
            stem, ext = os.path.splitext(out_path)
            full_out_path = f"{stem}_all{ext or '.parquet'}"
        else:
            out_path = str(out_dir / f"{args.dataset}_refinement.parquet")
            full_out_path = str(out_dir / f"{args.dataset}_refinement_all.parquet")

        pd.DataFrame(full_all_rows).to_parquet(full_out_path, index=False)
        df_processed = pd.DataFrame(filtered_sorted)
        df_processed.to_parquet(out_path, index=False)
        logger.info("Saved FULL processed data (including unqualified) to %s", full_out_path)
        logger.info("Saved FILTERED processed data to %s", out_path)
    logger.info("Statistics: total rows: %s", len(all_processed_rows))


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
