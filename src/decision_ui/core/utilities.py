import math
from decision_ui.core.data_types import Choice, ChoiceAnswer, Score, ScoreAnswer, NoulAnswer


def softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    scaled = [x / temperature for x in logits]
    m = max(scaled)
    if m == -math.inf:
        return [0.0 for _ in logits]
    
    exps = [math.exp(x - m) for x in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def confidence_from_probs(probs: list[float]) -> float:
    """Confidence in [0, 1] as 1 - normalized entropy.

    TypeSafe does not publish its confidence formula. This is an interpretable
    approximation: 1.0 when all mass is on one option, 0.0 when uniform.
    """
    n = len(probs)
    if n <= 1:
        return 1.0
    ent = -sum(p * math.log(p) for p in probs if p > 0)
    return max(0.0, min(1.0, 1.0 - ent / math.log(n)))


def build_choice_answer(
    question: Choice, option_logits: list[float], temperature: float = 1.0
) -> ChoiceAnswer:
    keys = question.option_keys()
    probs = softmax(option_logits, temperature)
    dist = {k: round(p, 4) for k, p in zip(keys, probs)}
    best = keys[max(range(len(keys)), key=lambda i: probs[i])]
    return ChoiceAnswer(choice=best, probabilities=dist, confidence=round(confidence_from_probs(probs), 4))


def build_score_answer(question: Score, level_logits: list[float], temperature: float = 1.0) -> ScoreAnswer:
    levels = question.levels()
    probs = softmax(level_logits, temperature)
    expectation = sum(l * p for l, p in zip(levels, probs))
    dist = {str(l): round(p, 4) for l, p in zip(levels, probs)}
    return ScoreAnswer(
        score=round(expectation, 4),
        legend=question.legend,
        probabilities=dist,
        confidence=round(confidence_from_probs(probs), 4),
    )


def build_noul_answer(yes_logit: float, no_logit: float, temperature: float = 1.0) -> NoulAnswer:
    p_yes, _ = softmax([yes_logit, no_logit], temperature)
    return NoulAnswer(noul=round(p_yes, 4))
