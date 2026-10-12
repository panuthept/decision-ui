"""vLLM backend.

One forward pass per (state, question). Logits are read at the last position and
only the first token of each label is compared, then softmaxed across labels so
nothing outside the option set can win.

Phase 1 will add: shared-state prefill with KV-cache reuse across questions, and
multi-token label scoring (sum of log-probs) for labels that are not single tokens.
"""

from __future__ import annotations

from openai import OpenAI
from decision_ui.backends.base import Backend


class VLLMBackend(Backend):
    def __init__(self, api_key: str | None = None, base_url: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.base_url = base_url
        self._model_client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self._tokenizer_client = OpenAI(api_key=self.api_key, base_url=self.base_url.replace("/v1", ""))

    @property
    def available_models(self) -> list[str]:
        return [model.id for model in self._model_client.models.list().data]

    def _tokenize(self, model: str, text: str) -> list[int]:
        enc = self._tokenizer_client.post(
            "/tokenize",
            cast_to=dict,
            body={
                "model": model,
                "prompt": text,
            },
        )
        return enc["tokens"]

    def _detokenize(self, model: str, token_ids: list[int]) -> str:
        dec = self._tokenizer_client.post(
            "/detokenize",
            cast_to=dict,
            body={
                "model": model,
                "tokens": token_ids,
            },
        )
        return dec["prompt"]

    def _first_token(self, model: str, label: str) -> str:
        ids = self._tokenize(model, label)
        return self._detokenize(model, [ids[0]])

    def score_options(self, model: str, prompt: str, labels: list[str]) -> tuple[list[float], int]:
        token_ids = self._tokenize(model, prompt)

        response = self._model_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            logprobs=True,
            top_logprobs=20,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            }
        )
        pred_tokens = {top_logprob.token: top_logprob.logprob for top_logprob in response.choices[0].logprobs.content[0].top_logprobs}
        cand_tokens = [self._first_token(model, label) for label in labels]
        logits = [pred_tokens.get(token, float("-inf")) for token in cand_tokens]
        return logits, len(token_ids)
