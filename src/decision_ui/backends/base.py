"""Backend interface.

A backend turns ``(state, question)`` pairs into option logits. Every backend must
implement :meth:`score_options`, which is the single primitive the rest of OpenJev
builds on. Choice / Score / Noul are all expressed as "score these labels given this
prompt", which is exactly one forward pass on any causal LM.
"""

from __future__ import annotations

import time
from typing import Callable
from abc import ABC, abstractmethod
from decision_ui.prompt_renders import get_prompt_render as default_prompt_render

from decision_ui.core.data_types import (
    Answer,
    Choice,
    Noul,
    Score,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
)
from decision_ui.core.utilities import (
    build_choice_answer,
    build_noul_answer,
    build_score_answer,
)

NOUL_LABELS = ["yes", "no"]


class Backend(ABC):
    def __init__(self, prompt_render: Callable = None, **kwargs):
        self.prompt_render = prompt_render or default_prompt_render

    @property
    @abstractmethod
    def available_models(self) -> list[str]:
        """Return a list of available model IDs."""

    @abstractmethod
    def score_options(self, model: str, prompt: str, labels: list[str]) -> tuple[list[float], int]:
        """Return one logit per label plus the number of input tokens consumed."""

    def decide(self, request: SystemOneRequest, temperature: float = 1.0) -> SystemOneResponse:
        t0 = time.perf_counter()
        answers: dict[str, Answer] = {}
        usage = Usage()
        for qid, q in request.questions.items():
            if isinstance(q, Choice):
                labels = q.option_keys()
                logits, n_in = self.score_options(
                    request.model,
                    self.prompt_render(request.state, q.instructions, labels), 
                    labels
                )
                answers[qid] = build_choice_answer(q, logits, temperature)
            elif isinstance(q, Score):
                labels = [f"{lvl} ({q.legend[str(lvl)]})" for lvl in q.levels()]
                logits, n_in = self.score_options(
                    request.model,
                    self.prompt_render(request.state, q.instructions, labels), 
                    labels
                )
                answers[qid] = build_score_answer(q, logits, temperature)
            elif isinstance(q, Noul):
                logits, n_in = self.score_options(
                    request.model,
                    self.prompt_render(request.state, q.instructions, NOUL_LABELS), 
                    NOUL_LABELS
                )
                answers[qid] = build_noul_answer(logits[0], logits[1], temperature)
            else:  # pragma: no cover
                raise TypeError(f"Unknown question type: {type(q)!r}")
            usage.input_tokens += n_in
            usage.output_tokens += len(labels) if not isinstance(q, Noul) else 2
        latency_ms = (time.perf_counter() - t0) * 1000
        return SystemOneResponse(
            model=request.model, answers=answers, usage=usage, latency_ms=round(latency_ms, 2)
        )