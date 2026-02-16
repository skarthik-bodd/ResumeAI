from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class AgentLLMConfig:
    provider: str
    model: str
    temperature: float = 0.2


@dataclass
class Settings:
    embeddings_model: str = "BAAI/bge-small-en-v1.5"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    top_k: int = 8
    max_revision_rounds: int = 2
    supervisor: AgentLLMConfig = field(
        default_factory=lambda: AgentLLMConfig(provider="ollama", model="qwen2.5:14b", temperature=0.1)
    )
    intern: AgentLLMConfig = field(
        default_factory=lambda: AgentLLMConfig(provider="ollama", model="llama3.1:8b", temperature=0.4)
    )
    reviewer: AgentLLMConfig = field(
        default_factory=lambda: AgentLLMConfig(provider="ollama", model="deepseek-r1:14b", temperature=0.1)
    )


class ConfigError(ValueError):
    """Raised when configuration is invalid."""


def _load_agent(section_name: str, data: dict, default: AgentLLMConfig) -> AgentLLMConfig:
    section = data.get(section_name, {})
    if not isinstance(section, dict):
        raise ConfigError(f"`{section_name}` must be a mapping in config.")

    provider = str(section.get("provider", default.provider)).strip()
    model = str(section.get("model", default.model)).strip()
    temperature = float(section.get("temperature", default.temperature))

    if not provider:
        raise ConfigError(f"`{section_name}.provider` cannot be empty.")
    if not model:
        raise ConfigError(f"`{section_name}.model` cannot be empty.")

    return AgentLLMConfig(provider=provider, model=model, temperature=temperature)


def load_settings(path: str | Path | None = None) -> Settings:
    settings = Settings()
    if path is None:
        return settings

    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ConfigError("Config file root must be a mapping.")

    settings.embeddings_model = str(raw.get("embeddings_model", settings.embeddings_model))
    settings.chunk_size = int(raw.get("chunk_size", settings.chunk_size))
    settings.chunk_overlap = int(raw.get("chunk_overlap", settings.chunk_overlap))
    settings.top_k = int(raw.get("top_k", settings.top_k))
    settings.max_revision_rounds = int(raw.get("max_revision_rounds", settings.max_revision_rounds))

    settings.supervisor = _load_agent("supervisor", raw, settings.supervisor)
    settings.intern = _load_agent("intern", raw, settings.intern)
    settings.reviewer = _load_agent("reviewer", raw, settings.reviewer)

    if settings.chunk_size <= 0:
        raise ConfigError("`chunk_size` must be greater than 0.")
    if settings.chunk_overlap < 0:
        raise ConfigError("`chunk_overlap` cannot be negative.")
    if settings.chunk_overlap >= settings.chunk_size:
        raise ConfigError("`chunk_overlap` must be smaller than `chunk_size`.")
    if settings.top_k <= 0:
        raise ConfigError("`top_k` must be greater than 0.")
    if settings.max_revision_rounds <= 0:
        raise ConfigError("`max_revision_rounds` must be greater than 0.")

    return settings
