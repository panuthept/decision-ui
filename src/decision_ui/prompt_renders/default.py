import json
from typing import Any


def render_state(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, indent=2)

def render_prompt(state: Any, instructions: str, labels: list[str]) -> str:
    """Chat-style prompt whose next token is expected to be one of ``labels``."""
    options = "\n".join(f"- {label}" for label in labels)
    return (
        "You are a decision engine. Read the state, then answer the question by replying "
        "with exactly one of the allowed labels and nothing else.\n\n"
        f"### State\n{render_state(state)}\n\n"
        f"### Question\n{instructions}\n\n"
        f"### Allowed labels\n{options}\n"
    )