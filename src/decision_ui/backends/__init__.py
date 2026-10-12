"""Backend registry."""

from __future__ import annotations

from .base import Backend


def get_backend(name: str = "mock", **kwargs) -> Backend:
    if name == "mock":
        from decision_ui.backends.mock import MockBackend

        return MockBackend()
    if name == "vllm":
        from decision_ui.backends.vllm import VLLMBackend

        return VLLMBackend(**kwargs)
    raise ValueError(f"Unknown backend '{name}'. Available: mock and vllm.")


__all__ = ["Backend", "get_backend"]
