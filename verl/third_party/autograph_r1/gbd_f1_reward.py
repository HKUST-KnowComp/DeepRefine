"""GBD reward with continuous token-level F1 gain.

Drop-in replacement for gbd_reward.compute_score.  The binary GenAcc
(span_check | judge_check) produces reward false negatives when the
prediction is a correct-but-terser span of the gold answer (e.g.
"Scunthorpe" vs gold "Scunthorpe in North Lincolnshire, England"):
one-directional span matching scores it 0.0, which collapses GRPO groups
to all-00 and yields zero advantage.  Token F1 gives partial credit, so
such rollouts keep contributing gradient, and no LLM judge is needed.

The gain is a continuous generalization of the four-quadrant GBD reward:

    delta = refined_f1 - draft_f1
    gain  = q01_gain * max(0,  delta)               # improvement
          + q10_gain * max(0, -delta)               # degradation
          + q11_gain * min(draft_f1, refined_f1)    # keep-correct bonus
          + q00_gain   (only when both F1 < threshold)
    score = gain + format_penalty

When both F1 values are in {0, 1} this reduces EXACTLY to the binary
quadrant reward of gbd_reward.py, so existing training scripts and their
q00/q01/q10/q11 kwargs work unchanged.
"""

import json_repair
import re
from collections import Counter
from typing import Union, Dict, List, Set

_ARTICLES = {"a", "an", "the"}

# Running transition stats (process-local, for log diagnostics).
# Labels use a configurable F1 threshold (default 0.5).
_TRANSITION_STATS = {
    "00": 0,
    "01": 0,
    "10": 0,
    "11": 0,
    "total": 0,
}


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _normalize_genacc(s):
    """Normalize for F1: lowercase, remove punctuation and articles."""
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
# Token F1 (SQuAD-style, multiset)
# ---------------------------------------------------------------------------

def compute_f1(answer, target) -> float:
    pred_tokens = _normalize_genacc(answer).split()
    gold_tokens = _normalize_genacc(target).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def f1_check(answer, gold_answers) -> float:
    """Max token F1 of ``answer`` against any cleaned gold alias."""
    if answer is None or not str(answer).strip():
        return 0.0
    gold_answers = _clean_gold_answers(gold_answers)
    if not gold_answers:
        return 0.0
    return max(compute_f1(answer, gold) for gold in gold_answers)


# ---------------------------------------------------------------------------
# Solution extraction (identical to gbd_reward.py)
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
# Main reward function
# ---------------------------------------------------------------------------

def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Computes reward as continuous F1 gain over draft_answer.

        reward ~ F1(refined_answer, target) - F1(draft_answer, target)

    shaped by the quadrant gains (see module docstring).  The precomputed
    binary ``refined_acc`` / ``draft_acc`` in ``rollout_reward_scores`` are
    NOT used for scoring (they are the source of the false-negative
    problem); they are only printed for cross-checking.
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

    # Token F1 for refined / draft answers (deterministic, no judge).
    refined_f1 = f1_check(answer, target)
    draft_f1 = f1_check(draft_answer, target) if draft_answer else 0.0

    # Four-quadrant configurable gains (same kwargs as gbd_reward.py).
    q00_gain = float(kwargs.get("q00_gain", 0.0))
    q01_gain = float(kwargs.get("q01_gain", 1.0))
    q10_gain = float(kwargs.get("q10_gain", kwargs.get("break_correct_draft_penalty", -0.2)))
    q11_gain = float(kwargs.get("q11_gain", kwargs.get("keep_correct_gain", 0.2)))
    f1_threshold = float(kwargs.get("f1_threshold", 0.5))

    delta = refined_f1 - draft_f1
    gain = (
        q01_gain * max(0.0, delta)
        + q10_gain * max(0.0, -delta)
        + q11_gain * min(draft_f1, refined_f1)
    )
    if draft_f1 < f1_threshold and refined_f1 < f1_threshold:
        gain += q00_gain

    # Format penalty
    format_penalty = 0.0
    if rollout_scores.get("format_error"):
        format_penalty = -0.1

    score = gain + format_penalty

    # Transition diagnostics at the F1 threshold (keeps p_01 / p_10 logging
    # comparable with the binary-reward runs).
    draft_bin = 1 if draft_f1 >= f1_threshold else 0
    refined_bin = 1 if refined_f1 >= f1_threshold else 0
    transition = f"{draft_bin}{refined_bin}"

    _TRANSITION_STATS["total"] += 1
    _TRANSITION_STATS[transition] += 1
    total = max(1, _TRANSITION_STATS["total"])
    p_10 = _TRANSITION_STATS["10"] / total
    p_01 = _TRANSITION_STATS["01"] / total

    same_answer = bool(rollout_scores.get("same_draft_refined")) or _answers_equivalent(
        answer, draft_answer
    )

    print(
        "[refined answer: %s, draft answer: %s, ground truth: %s, "
        "refined_f1: %.4f, draft_f1: %.4f, delta: %.4f, transition: %s, gain: %s, "
        "format_penalty: %s, P(1->0): %.4f, P(0->1): %.4f, "
        % (
            answer,
            draft_answer,
            target,
            refined_f1,
            draft_f1,
            delta,
            transition,
            gain,
            format_penalty,
            p_10,
            p_01,
        )
    )

    return {
        "score": score,
        "current_acc": refined_f1,
        "draft_acc": draft_f1,
        "transition": transition,
        "p_10": p_10,
        "p_01": p_01,
        "gain": gain,
        "format_penalty": format_penalty,
        "triple_repetition": float(triple_repetition),
    }
