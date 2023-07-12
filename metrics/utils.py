import re
import string
from typing import Any, Union, List


def exact_match_stripped(pred: Any, ground_truth: Union[Any, List[Any], dict[Any]]):
    if isinstance(ground_truth, dict):
        # hotfix for trivia_qa
        return exact_match_stripped(pred, ground_truth["aliases"])
    if isinstance(ground_truth, list):
        return any(
            exact_match_stripped(pred, ground_truth_single)
            for ground_truth_single in ground_truth
        )
    else:
        return str(pred).strip() == str(ground_truth).strip()


def exact_match(pred: Any, ground_truth: Union[Any, List[Any], dict[Any]]):
    if isinstance(ground_truth, dict):
        # hotfix for trivia_qa
        return exact_match(pred, ground_truth["aliases"])
    if isinstance(ground_truth, list):
        return any(
            exact_match(pred, ground_truth_single)
            for ground_truth_single in ground_truth
        )
    else:
        return pred == ground_truth


def _normalize_text(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace.
    Copied from the [QuAC](http://quac.ai/) evaluation script found at
    https://s3.amazonaws.com/my89public/quac/scorer.py"""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text)))).strip()


def quasi_exact_match(
    pred: Any, ground_truth: Union[Any, List[Any], dict[Any]]
) -> float:
    """From CRFM HELM
    https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/metrics/basic_metrics.py
    """
    if isinstance(ground_truth, dict):
        # hotfix for trivia_qa
        return quasi_exact_match(pred, ground_truth["aliases"])
    if isinstance(ground_truth, list):
        return any(
            quasi_exact_match(pred, ground_truth_single)
            for ground_truth_single in ground_truth
        )
    else:
        return _normalize_text(str(ground_truth)) == _normalize_text(str(pred))
