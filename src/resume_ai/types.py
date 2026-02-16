from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Document:
    source: str
    text: str


@dataclass
class Chunk:
    chunk_id: str
    source: str
    text: str


@dataclass
class RetrievalHit:
    chunk: Chunk
    score: float


@dataclass
class ReviewFeedback:
    decision: str
    score: float
    strengths: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    edits: list[str] = field(default_factory=list)
    summary: str = ""
    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SupervisorDecision:
    action: str
    reason: str
    focus: list[str] = field(default_factory=list)
    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunResult:
    final_resume: str
    draft_resume: str
    review_rounds: list[ReviewFeedback]
    supervisor_rounds: list[SupervisorDecision]
    retrieval_hits: list[RetrievalHit]
