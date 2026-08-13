import json_repair
import os
import re
import requests
from typing import Union, Dict, List, Set

_ARTICLES = {"a", "an", "the"}

_LLM_JUDGE_PROMPT = (
    'Given the following prediction and set of gold answers, determine if the '
    'prediction contains or is semantically equivalent to any of the gold answers.\n\n'
    'Prediction: "{prediction}"\n'
    'Gold Answers: {gold_answers}\n\n'
    'Does the prediction contain any of the gold answers? '
    "Answer with ONLY 'Yes' or 'No'."
)

# Running transition stats (process-local, for log diagnostics)
_TRANSITION_STATS = {
    "00": 0,  # draft=0 -> refined=0
    "01": 0,  # draft=0 -> refined=1
    "10": 0,  # draft=1 -> refined=0
    "11": 0,  # draft=1 -> refined=1
    "total": 0,
}


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def normalize_string(s):
    if isinstance(s, list):
        s = " ".join(str(item).strip() for item in s)
    if not isinstance(s, str):
        s = str(s)
    s = s.lower().strip()
    s = re.sub(r'\s+', ' ', s)
    s = s.replace("\u2019", "'").replace("`", "'").replace("\u2018", "'")
    s = re.sub(r'[^\w\s]', '', s)
    return s


def _normalize_genacc(s):
    """Normalize for GenAcc: lowercase, remove punctuation and articles."""
    if not isinstance(s, str):
        s = str(s)
    s = s.lower().strip()
    s = re.sub(r'[^\w\s]', '', s)
    tokens = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(tokens)


def _strip_wrapping_quotes(text: str) -> str:
    s = text.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1].strip()
    return s


def _clean_gold_answers(gold_answers):
    if gold_answers is None:
        return []
    if isinstance(gold_answers, str):
        gold_answers = [gold_answers]
    cleaned = []
    for ans in gold_answers:
        if ans is None:
            continue
        s = _strip_wrapping_quotes(str(ans))
        if s and _normalize_genacc(s):
            cleaned.append(s)
    return cleaned


def _answers_equivalent(a, b) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return _normalize_genacc(a) == _normalize_genacc(b)


# ---------------------------------------------------------------------------
# GenAcc: span_check | judge_check
# ---------------------------------------------------------------------------

def span_check(prediction, gold_answers):
    """Return True if any normalized gold answer is a contiguous token span
    within the normalized prediction."""
    norm_pred = _normalize_genacc(prediction)
    for ans in gold_answers:
        norm_ans = _normalize_genacc(str(ans))
        if not norm_ans:
            continue
        if norm_ans in norm_pred:
            return True
    return False


def judge_check(prediction, gold_answers, api_url=None, model_name=None, timeout=60.0):
    """Use the same LLM as RAG to judge whether prediction contains any of
    gold_answers.  Communicates via the OpenAI-compatible chat completions
    endpoint (synchronous requests call)."""
    if api_url is None:
        api_url = os.environ.get(
            "REWARD_LLM_API_URL", "http://127.0.0.1:8129/v1/chat/completions"
        )
    if not api_url.endswith("/chat/completions"):
        api_url = api_url.rstrip("/") + "/chat/completions"
    if model_name is None:
        model_name = os.environ.get("REWARD_LLM_MODEL", "default")

    gold_str = ", ".join('"' + str(a) + '"' for a in gold_answers)
    prompt = _LLM_JUDGE_PROMPT.format(prediction=prediction, gold_answers=gold_str)

    try:
        resp = requests.post(
            api_url,
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 10,
                "temperature": 0,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip().lower()
        return text.startswith("yes")
    except Exception as e:
        print("[judge_check] LLM call failed, falling back to False: " + str(e))
        return False


def gen_acc(prediction, gold_answers, use_judge=True, api_url=None, model_name=None):
    """Compute Generation Accuracy: span_check | judge_check.

    Returns 1.0 (correct) or 0.0 (incorrect).
    """
    if prediction is None or not str(prediction).strip():
        return 0.0
    gold_answers = _clean_gold_answers(gold_answers)
    if not gold_answers:
        return 0.0
    if span_check(prediction, gold_answers):
        return 1.0
    if use_judge:
        if judge_check(prediction, gold_answers, api_url=api_url, model_name=model_name):
            return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Legacy helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------

def extract_refinement_result(solution_str):
    result = {
        "answerable": None,
        "error_reason": None,
        "refinement_actions": []
    }
    judge_match = re.search(r'<judge>(.*?)</judge>', solution_str, re.IGNORECASE | re.DOTALL)
    if judge_match:
        judge_text = judge_match.group(1).strip().lower()
        result["answerable"] = judge_text.startswith("yes")
    abduction_match = re.search(r'<abduction>(.*?)</abduction>', solution_str, re.IGNORECASE | re.DOTALL)
    if abduction_match:
        result["error_reason"] = abduction_match.group(1).strip()
    refinement_match = re.search(r'<refinement>(.*?)</refinement>', solution_str, re.IGNORECASE | re.DOTALL)
    if refinement_match:
        refinement_actions_str = refinement_match.group(1).strip().strip("|")
        for action in refinement_actions_str.split("|"):
            action = action.strip()
            if action:
                result["refinement_actions"].append(action)
    return result


def parse_action_string(action):
    action = action.strip()
    pattern = r'(\w+)\s*\((.*)\)\s*$'
    match = re.match(pattern, action)
    if not match:
        return None, []
    function_name = match.group(1)
    args_str = match.group(2).strip()
    parsed_args = []
    i = 0
    while i < len(args_str):
        while i < len(args_str) and args_str[i] in ' \t,':
            i += 1
        if i >= len(args_str):
            break
        quote_char = args_str[i]
        if quote_char not in ['"', "'"]:
            return None, []
        i += 1
        arg_value = []
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
            return None, []
    return function_name, parsed_args


# ---------------------------------------------------------------------------
# Solution extraction
# ---------------------------------------------------------------------------

def _try_parse_rag_json(text):
    """Try to parse a RAG JSON dict from text. Returns the 4-tuple or None."""
    try:
        obj = json_repair.loads(text)
    except Exception:
        return None
    if isinstance(obj, dict) and "answer" in obj:
        return (
            obj.get("answer"),
            obj.get("edge_coverage", 1),
            obj.get("semantic_reward", 0),
            obj.get("triple_repetition", 0),
        )
    if isinstance(obj, list) and obj and isinstance(obj[0], dict) and "answer" in obj[0]:
        return (
            obj[0].get("answer"),
            obj[0].get("edge_coverage", 1),
            obj[0].get("semantic_reward", 0),
            obj[0].get("triple_repetition", 0),
        )
    return None


def extract_solution(solution_str):
    try:
        assistant_parts = solution_str.split('assistant\n')
        if not assistant_parts:
            return None, 1, 0, 0
        last_assistant_response = assistant_parts[-1].strip()
        solution_dict = json_repair.loads(last_assistant_response)
        if isinstance(solution_dict, dict):
            return solution_dict.get("answer", None), solution_dict.get("edge_coverage", 1), solution_dict.get("semantic_reward", 0), solution_dict.get("triple_repetition", 0)
        elif isinstance(solution_dict, list):
            if solution_dict and isinstance(solution_dict[0], dict):
                return solution_dict[0].get("answer", None), solution_dict[0].get("edge_coverage", 1), solution_dict[0].get("semantic_reward", 0), solution_dict[0].get("triple_repetition", 0)
            return None, 1, 0, 0
        else:
            return None, 1, 0, 0
    except Exception as e:
        print("Error extracting solution: " + str(e))
        return None, 1, 0, 0


# ---------------------------------------------------------------------------
# F1 helpers (kept for reference / optional use)
# ---------------------------------------------------------------------------

def f1_check(answer, target):
    if answer is None:
        return 0.0
    if isinstance(target, list):
        return max((compute_f1(answer, alias) for alias in target), default=0.0)
    return compute_f1(answer, target)


def compute_f1(answer, target):
    pred_tokens = get_tokens(answer)
    gold_tokens = get_tokens(target)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common_tokens = pred_tokens.intersection(gold_tokens)
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def get_tokens(text):
    normalized = normalize_string(text)
    return set(normalized.split())


# ---------------------------------------------------------------------------
# Main reward function
# ---------------------------------------------------------------------------

def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Computes reward as GenAcc gain over draft_answer:

        reward = GenAcc(refined_answer, target) - GenAcc(draft_answer, target)

    GenAcc = span_check | judge_check  (see paper Figure 13).

    The rollout pre-computes ``refined_acc`` / ``draft_acc`` using the async
    LLM judge (same model as RAG).  If those values are present in
    ``rollout_reward_scores``, they are used directly.  Otherwise we fall back
    to span_check only (no LLM judge, since the HTTP endpoint is unavailable
    in the TaskRunner process).
    """
    # Extract current answer from the rollout response
    try:
        answer, _, _, triple_repetition = extract_solution(solution_str)
    except Exception:
        answer = None
        triple_repetition = 0.0

    target = _clean_gold_answers(ground_truth.get("target", []))

    # Read rollout_reward_scores once
    rollout_scores = {}
    if isinstance(extra_info, dict):
        rollout_scores = extra_info.get("rollout_reward_scores") or {}
        if not isinstance(rollout_scores, dict):
            rollout_scores = {}

    # Prefer the reward-only RAG answer extracted during rollout over
    # extract_solution (from the policy trajectory).
    rollout_answer = rollout_scores.get("refined_answer_text")
    if rollout_answer is not None:
        answer = rollout_answer

    precomputed_refined = rollout_scores.get("refined_acc")
    precomputed_draft = rollout_scores.get("draft_acc")
    reward_invalid = bool(rollout_scores.get("reward_invalid"))

    # Extract draft_answer
    draft_answer = None
    if isinstance(extra_info, dict):
        draft_answer = extra_info.get("draft_answer")
        if draft_answer is None:
            draft_answer = rollout_scores.get("draft_answer")
        if draft_answer is None:
            interaction_kwargs = extra_info.get("interaction_kwargs") or {}
            draft_answer = interaction_kwargs.get("draft_answer")

    # Empty / unusable GT or rollout-marked invalid: zero reward, no fake transitions.
    if reward_invalid or not target:
        print(
            "[refined answer: %s, draft answer: %s, ground truth: %s, "
            "reward_invalid: True, score: 0.0, reason: empty_or_invalid_gold]"
            % (answer, draft_answer, target)
        )
        return {
            "score": 0.0,
            "current_acc": 0.0,
            "draft_acc": 0.0,
            "transition": "00",
            "p_10": _TRANSITION_STATS["10"] / max(1, _TRANSITION_STATS["total"]),
            "p_01": _TRANSITION_STATS["01"] / max(1, _TRANSITION_STATS["total"]),
            "gain": 0.0,
            "format_penalty": 0.0,
            "triple_repetition": float(triple_repetition),
        }

    # GenAcc for refined answer
    if precomputed_refined is not None:
        current_acc = float(precomputed_refined)
    else:
        current_acc = gen_acc(answer, target, use_judge=False)

    # GenAcc for draft answer
    if precomputed_draft is not None:
        draft_acc = float(precomputed_draft)
    elif draft_answer:
        draft_acc = gen_acc(draft_answer, target, use_judge=False)
    else:
        draft_acc = 0.0

    # Identical draft/refined must share one label (defensive against judge noise).
    same_answer = bool(rollout_scores.get("same_draft_refined")) or _answers_equivalent(
        answer, draft_answer
    )
    if same_answer:
        shared = current_acc if precomputed_refined is not None else (
            draft_acc if precomputed_draft is not None else current_acc
        )
        current_acc = shared
        draft_acc = shared

    # Four-quadrant configurable gain:
    # q00: draft=0 -> refined=0
    # q01: draft=0 -> refined=1
    # q10: draft=1 -> refined=0
    # q11: draft=1 -> refined=1
    #
    # Defaults are chosen to explicitly encourage:
    #   P(1->0) lower than P(0->1)
    # by rewarding 0->1 strongly and penalizing 1->0.
    q00_gain = float(kwargs.get("q00_gain", 0.0))
    q01_gain = float(kwargs.get("q01_gain", 1.0))
    q10_gain = float(kwargs.get("q10_gain", kwargs.get("break_correct_draft_penalty", -0.2)))
    q11_gain = float(kwargs.get("q11_gain", kwargs.get("keep_correct_gain", 0.2)))

    if draft_acc not in (0.0, 1.0) or current_acc not in (0.0, 1.0):
        print(
            "[gbd_f1_reward] Warning: expected binary acc values, got "
            f"draft_acc={draft_acc}, current_acc={current_acc}"
        )
    draft_bin = 1 if float(draft_acc) == 1.0 else 0
    refined_bin = 1 if float(current_acc) == 1.0 else 0
    transition = f"{draft_bin}{refined_bin}"
    if transition == "00":
        gain = q00_gain
    elif transition == "01":
        gain = q01_gain
    elif transition == "10":
        gain = q10_gain
    else:
        gain = q11_gain

    # Format penalty
    format_penalty = 0.0
    if rollout_scores.get("format_error"):
        format_penalty = -0.1

    score = gain + format_penalty

    # Update running transition diagnostics
    _TRANSITION_STATS["total"] += 1
    _TRANSITION_STATS[transition] += 1
    total = max(1, _TRANSITION_STATS["total"])
    p_10 = _TRANSITION_STATS["10"] / total
    p_01 = _TRANSITION_STATS["01"] / total

    print(
        "[refined answer: %s, draft answer: %s, ground truth: %s, "
        "refined_acc: %s, draft_acc: %s, transition: %s, gain: %s, format_penalty: %s, "
        "P(1->0): %.4f, P(0->1): %.4f, precomputed: %s, same_answer: %s]"
        % (
            answer,
            draft_answer,
            target,
            current_acc,
            draft_acc,
            transition,
            gain,
            format_penalty,
            p_10,
            p_01,
            precomputed_refined is not None,
            same_answer,
        )
    )

    return {
        "score": score,
        "current_acc": current_acc,
        "draft_acc": draft_acc,
        "transition": transition,
        "p_10": p_10,
        "p_01": p_01,
        "gain": gain,
        "format_penalty": format_penalty,
        "triple_repetition": float(triple_repetition),
    }
