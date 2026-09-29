
from typing import Optional

from src.opsd_format import (
    KK_DATA_SOURCES,
    extract_boxed_answer,
    grade_boxed_answer,
    grade_kk_answer,
)

CODE_DATA_SOURCES = {"apps", "codecontests", "codeforces", "taco"}


def _compute_code_score(solution_str: str, ground_truth: str) -> dict:
    import sys
    from pathlib import Path

    prime_code_root = Path(__file__).resolve().parent.parent / "third_party" / "verl" / "verl" / "utils" / "reward_score"
    if str(prime_code_root) not in sys.path:
        sys.path.insert(0, str(prime_code_root))

    from prime_code import compute_score as prime_code_compute_score

    result = prime_code_compute_score(solution_str, ground_truth, continuous=False)
    if isinstance(result, tuple):
        success = result[0]
    else:
        success = result
    score = 1.0 if bool(success) else 0.0
    return {"score": score, "acc": score, "formatted": 1.0}


def _compute_math_score(solution_str: str, ground_truth: str) -> dict:
    predicted = extract_boxed_answer(solution_str)
    formatted = 1.0 if predicted is not None else 0.0
    correct = grade_boxed_answer(predicted, ground_truth) if predicted is not None else False
    return {
        "score": 1.0 if correct else 0.0,
        "acc": 1.0 if correct else 0.0,
        "formatted": formatted,
    }


def _compute_kk_score(solution_str: str, ground_truth: str) -> dict:
    """Score a K&K completion by parsing the boxed assignment list."""
    predicted = extract_boxed_answer(solution_str)
    formatted = 1.0 if predicted is not None else 0.0
    correct = grade_kk_answer(predicted, ground_truth) if predicted is not None else False
    return {
        "score": 1.0 if correct else 0.0,
        "acc": 1.0 if correct else 0.0,
        "formatted": formatted,
    }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> dict:
    del extra_info, kwargs

    src = str(data_source).strip().lower()
    if src in CODE_DATA_SOURCES:
        return _compute_code_score(solution_str, ground_truth)
    if src in KK_DATA_SOURCES:
        return _compute_kk_score(solution_str, ground_truth)
    return _compute_math_score(solution_str, ground_truth)


def compute_reward(responses: list[str], ground_truths: list[str]) -> list[float]:
    rewards = []
    for response, ground_truth in zip(responses, ground_truths, strict=True):
        predicted = extract_boxed_answer(response)
        rewards.append(1.0 if grade_boxed_answer(predicted, ground_truth) else 0.0)
    return rewards
