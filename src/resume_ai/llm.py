from __future__ import annotations

import os

from ollama import Client

from .config import AgentLLMConfig


class LLMClientError(RuntimeError):
    """Raised when a model call fails."""


class MultiProviderLLMClient:
    """A minimal provider router. Currently supports local Ollama models."""

    def __init__(self) -> None:
        self._ollama_client = Client(host=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))

    def chat(self, system_prompt: str, user_prompt: str, config: AgentLLMConfig) -> str:
        provider = config.provider.lower().strip()
        if provider == "ollama":
            return self._chat_ollama(system_prompt, user_prompt, config)

        raise LLMClientError(
            f"Provider `{config.provider}` is not supported in this starter project. "
            "Use `ollama` or extend MultiProviderLLMClient."
        )

    def _chat_ollama(self, system_prompt: str, user_prompt: str, config: AgentLLMConfig) -> str:
        try:
            response = self._ollama_client.chat(
                model=config.model,
                options={"temperature": config.temperature},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMClientError(
                f"Ollama request failed for model `{config.model}`. "
                "Make sure `ollama serve` is running and the model is pulled."
            ) from exc

        message = response.get("message", {})
        content = message.get("content", "")
        if not content:
            raise LLMClientError(f"Model `{config.model}` returned an empty response.")
        return content.strip()
