from __future__ import annotations


def format_retrieval_context(chunks: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for i, item in enumerate(chunks, start=1):
        lines.append(f"[{i}] Source: {item['source']}")
        lines.append(item["text"])
        lines.append("")
    return "\n".join(lines).strip()


INTERN_SYSTEM_PROMPT = """You are an intern resume writer.
You are careful with facts and never invent skills, dates, titles, or achievements.
Write in concise US resume style using strong action verbs and measurable impact when evidence exists.
Output only Markdown.
"""


def intern_draft_user_prompt(job_description: str, context: str) -> str:
    return f"""Create a customized one-page resume in Markdown for this job description.
Focus on relevance and ATS-friendly phrasing while staying truthful to the supplied evidence.

Job description:
{job_description}

Candidate evidence (RAG context):
{context}

Required output structure:
1. # Candidate Name (placeholder if unknown)
2. ## Summary
3. ## Skills
4. ## Experience
5. ## Projects (if evidence exists)
6. ## Education (if evidence exists)

Rules:
- Do not fabricate details.
- If specific metrics are unavailable, write impact without numbers.
- Keep bullet points tight.
"""


def intern_revision_user_prompt(
    job_description: str,
    current_resume: str,
    review_feedback: str,
    supervisor_focus: list[str],
    context: str,
) -> str:
    focus_text = "\n".join([f"- {item}" for item in supervisor_focus]) or "- Improve overall alignment"
    return f"""Revise the resume using reviewer feedback and supervisor priorities.
Keep only verifiable facts from the provided evidence.

Job description:
{job_description}

Current resume:
{current_resume}

Reviewer feedback:
{review_feedback}

Supervisor focus areas:
{focus_text}

Candidate evidence (RAG context):
{context}

Output only the updated Markdown resume.
"""


REVIEWER_SYSTEM_PROMPT = """You are a senior FAANG-style resume reviewer.
Be strict about job alignment, clarity, impact, and factual consistency.
Return JSON only.
"""


def reviewer_user_prompt(job_description: str, resume: str) -> str:
    return f"""Review this resume against the job description and return JSON with this schema:
{{
  "decision": "accept" | "revise",
  "score": 0-10,
  "summary": "short paragraph",
  "strengths": ["..."],
  "risks": ["..."],
  "edits": ["specific changes intern should make"]
}}

Job description:
{job_description}

Resume draft:
{resume}
"""


SUPERVISOR_SYSTEM_PROMPT = """You are the orchestration supervisor for a resume-writing multi-agent system.
Given the latest reviewer JSON, decide whether the intern should revise or stop.
Return JSON only.
"""


def supervisor_user_prompt(
    round_number: int,
    max_rounds: int,
    reviewer_feedback_json: str,
) -> str:
    return f"""You are at revision round {round_number} of {max_rounds}.

Return JSON with this schema:
{{
  "action": "accept" | "revise",
  "reason": "one sentence",
  "focus": ["short, actionable direction for intern"]
}}

Rules:
- If reviewer score >= 8 and no critical risks, prefer accept.
- If rounds are exhausted, accept unless there are severe factual issues.
- Keep focus items specific and brief.

Reviewer feedback JSON:
{reviewer_feedback_json}
"""
