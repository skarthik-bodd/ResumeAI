from __future__ import annotations

import json
import re

from .config import Settings
from .llm import MultiProviderLLMClient
from .prompts import (
    INTERN_SYSTEM_PROMPT,
    REVIEWER_SYSTEM_PROMPT,
    SUPERVISOR_SYSTEM_PROMPT,
    intern_draft_user_prompt,
    intern_revision_user_prompt,
    reviewer_user_prompt,
    supervisor_user_prompt,
)
from .types import ReviewFeedback, SupervisorDecision


def _extract_json_object(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback for models that wrap JSON in markdown fences.
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}

    candidate = match.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return {}


class InternAgent:
    def __init__(self, llm: MultiProviderLLMClient, settings: Settings) -> None:
        self.llm = llm
        self.settings = settings

    def draft(self, job_description: str, context: str) -> str:
        return self.llm.chat(
            system_prompt=INTERN_SYSTEM_PROMPT,
            user_prompt=intern_draft_user_prompt(job_description=job_description, context=context),
            config=self.settings.intern,
        )

    def revise(
        self,
        job_description: str,
        current_resume: str,
        review_feedback: str,
        supervisor_focus: list[str],
        context: str,
    ) -> str:
        return self.llm.chat(
            system_prompt=INTERN_SYSTEM_PROMPT,
            user_prompt=intern_revision_user_prompt(
                job_description=job_description,
                current_resume=current_resume,
                review_feedback=review_feedback,
                supervisor_focus=supervisor_focus,
                context=context,
            ),
            config=self.settings.intern,
        )


class ReviewerAgent:
    def __init__(self, llm: MultiProviderLLMClient, settings: Settings) -> None:
        self.llm = llm
        self.settings = settings

    def review(self, job_description: str, resume: str) -> ReviewFeedback:
        raw = self.llm.chat(
            system_prompt=REVIEWER_SYSTEM_PROMPT,
            user_prompt=reviewer_user_prompt(job_description=job_description, resume=resume),
            config=self.settings.reviewer,
        )

        payload = _extract_json_object(raw)
        decision = str(payload.get("decision", "revise")).lower().strip()
        if decision not in {"accept", "revise"}:
            decision = "revise"

        try:
            score = float(payload.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(10.0, score))

        strengths = _ensure_list_of_strings(payload.get("strengths", []))
        risks = _ensure_list_of_strings(payload.get("risks", []))
        edits = _ensure_list_of_strings(payload.get("edits", []))
        summary = str(payload.get("summary", "")).strip()

        return ReviewFeedback(
            decision=decision,
            score=score,
            strengths=strengths,
            risks=risks,
            edits=edits,
            summary=summary,
            raw_text=raw,
        )


class SupervisorAgent:
    def __init__(self, llm: MultiProviderLLMClient, settings: Settings) -> None:
        self.llm = llm
        self.settings = settings

    def decide(
        self,
        review_feedback: ReviewFeedback,
        round_number: int,
    ) -> SupervisorDecision:
        raw = self.llm.chat(
            system_prompt=SUPERVISOR_SYSTEM_PROMPT,
            user_prompt=supervisor_user_prompt(
                round_number=round_number,
                max_rounds=self.settings.max_revision_rounds,
                reviewer_feedback_json=review_feedback.raw_text,
            ),
            config=self.settings.supervisor,
        )

        payload = _extract_json_object(raw)
        action = str(payload.get("action", review_feedback.decision)).lower().strip()
        if action not in {"accept", "revise"}:
            action = "revise"

        reason = str(payload.get("reason", "")).strip()
        focus = _ensure_list_of_strings(payload.get("focus", []))

        if not reason:
            reason = (
                "Reviewer signaled sufficient quality."
                if action == "accept"
                else "Reviewer requested targeted edits."
            )

        if action == "accept" and review_feedback.score < 7.5 and round_number < self.settings.max_revision_rounds:
            action = "revise"
            if not focus:
                focus = review_feedback.edits[:3]

        if action == "revise" and round_number >= self.settings.max_revision_rounds:
            action = "accept"
            reason = "Max rounds reached; finishing with current best draft."

        return SupervisorDecision(action=action, reason=reason, focus=focus, raw_text=raw)


def _ensure_list_of_strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
