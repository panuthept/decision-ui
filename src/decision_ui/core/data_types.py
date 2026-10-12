"""System One primitives: Choice, Score, Noul.

These models mirror the request / response shape of TypeSafe's ``POST /v1/systemone``
endpoint so that the official SDKs work against OpenJev by changing only the base URL.

OpenJev reproduces the *interface pattern*, not TypeSafe's undisclosed model or training.
"""

from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, Field, model_validator

# --------------------------------------------------------------------------------------
# Questions (request side)
# --------------------------------------------------------------------------------------


class Choice(BaseModel):
    """Pick exactly one option from a runtime-defined set (2..255 options)."""

    type: Literal["choice"] = "choice"
    instructions: str
    options: list[str] | None = None
    criteria: dict[str, str | None] | None = None

    @model_validator(mode="after")
    def _check_options(self) -> Choice:
        keys = self.option_keys()
        if len(keys) < 2:
            raise ValueError("Choice needs at least 2 options (use `options` or `criteria`).")
        if len(keys) > 255:
            raise ValueError("Choice supports at most 255 options.")
        return self

    def option_keys(self) -> list[str]:
        if self.criteria:
            return list(self.criteria.keys())
        return list(self.options or [])


class Score(BaseModel):
    """Grade the state on an ordered scale (2..10 levels). Returns a probability-weighted score.

    Accepts either the official request form ``criteria=["level 0 desc", "level 1 desc", ...]``
    or the response-style ``legend={"0": "...", "1": "..."}``. ``legend`` is always populated.
    """

    type: Literal["score"] = "score"
    instructions: str
    criteria: list[str] | None = None
    legend: dict[str, str] | None = None

    @model_validator(mode="after")
    def _normalise(self) -> "Score":
        if self.legend is None and self.criteria is not None:
            self.legend = {str(i): desc for i, desc in enumerate(self.criteria)}
        if self.legend is None:
            raise ValueError("Score needs `criteria` (list) or `legend` (dict).")
        if not (2 <= len(self.legend) <= 10):
            raise ValueError("Score needs between 2 and 10 levels.")
        for k in self.legend:
            int(k)  # raises if not numeric
        return self

    def levels(self) -> list[int]:
        return sorted(int(k) for k in self.legend)


class Noul(BaseModel):
    """A yes/no judgement. Returns P(true) in [0, 1]."""

    type: Literal["noul"] = "noul"
    instructions: str


Question = Union[Choice, Score, Noul]


class SystemOneRequest(BaseModel):
    """Body of ``POST /v1/systemone``."""

    state: Any
    questions: dict[str, Question]
    model: str = "openjev-latest"


# --------------------------------------------------------------------------------------
# Answers (response side)
# --------------------------------------------------------------------------------------


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float


Answer = Union[ChoiceAnswer, ScoreAnswer, NoulAnswer]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float | None = None