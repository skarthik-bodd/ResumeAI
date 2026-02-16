from __future__ import annotations

import json

from .agents import InternAgent, ReviewerAgent, SupervisorAgent
from .config import Settings
from .prompts import format_retrieval_context
from .types import RetrievalHit, RunResult
from .vector_store import LocalVectorStore


class ResumeOrchestrator:
    def __init__(self, settings: Settings, vector_store: LocalVectorStore, intern: InternAgent, reviewer: ReviewerAgent, supervisor: SupervisorAgent) -> None:
        self.settings = settings
        self.vector_store = vector_store
        self.intern = intern
        self.reviewer = reviewer
        self.supervisor = supervisor

    def run(self, job_description: str) -> RunResult:
        hits = self.vector_store.search(job_description, top_k=self.settings.top_k)
        context = format_retrieval_context(
            [
                {
                    "source": hit.chunk.source,
                    "text": hit.chunk.text,
                }
                for hit in hits
            ]
        )

        draft_resume = self.intern.draft(job_description=job_description, context=context)
        current_resume = draft_resume

        review_rounds = []
        supervisor_rounds = []

        for round_number in range(1, self.settings.max_revision_rounds + 1):
            review = self.reviewer.review(job_description=job_description, resume=current_resume)
            review_rounds.append(review)

            decision = self.supervisor.decide(review_feedback=review, round_number=round_number)
            supervisor_rounds.append(decision)

            if decision.action == "accept":
                break

            feedback_blob = json.dumps(review.to_dict(), indent=2)
            current_resume = self.intern.revise(
                job_description=job_description,
                current_resume=current_resume,
                review_feedback=feedback_blob,
                supervisor_focus=decision.focus,
                context=context,
            )

        return RunResult(
            final_resume=current_resume,
            draft_resume=draft_resume,
            review_rounds=review_rounds,
            supervisor_rounds=supervisor_rounds,
            retrieval_hits=hits,
        )
