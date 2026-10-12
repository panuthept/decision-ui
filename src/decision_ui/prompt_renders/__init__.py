from __future__ import annotations

from typing import Any
from decision_ui.prompt_renders.default import render_prompt as default_render_prompt


def get_prompt_render(state: Any, instructions: str, labels: list[str], name: str = "default") -> str:
    if name == "default":
        return default_render_prompt(state, instructions, labels)
    raise ValueError(f"Unknown prompt name: {name}")


__all__ = ["get_prompt_render"]