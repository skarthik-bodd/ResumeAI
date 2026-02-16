from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agents import InternAgent, ReviewerAgent, SupervisorAgent
from .chunking import build_chunks
from .config import ConfigError, load_settings
from .document_loader import DocumentLoadError, load_documents, read_file_text
from .llm import LLMClientError, MultiProviderLLMClient
from .orchestrator import ResumeOrchestrator
from .vector_store import LocalVectorStore, VectorStoreError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="resume-ai",
        description="Multi-agent RAG resume customizer.",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--job-description-file",
        required=True,
        help="Path to the job description file (.md, .txt, .pdf, .docx).",
    )
    parser.add_argument(
        "--documents",
        nargs="+",
        required=True,
        help="One or more files/directories containing candidate evidence docs.",
    )
    parser.add_argument(
        "--output",
        default="outputs/resume.md",
        help="Path to write final resume markdown.",
    )
    parser.add_argument(
        "--report-output",
        default="outputs/run_report.json",
        help="Path to write run metadata report.",
    )
    parser.add_argument(
        "--index-path",
        default=".cache/resume_index",
        help="Base path for index cache (without extension).",
    )
    parser.add_argument(
        "--reuse-index",
        action="store_true",
        help="Load existing index if present instead of rebuilding.",
    )
    parser.add_argument(
        "--supervisor-model",
        default=None,
        help="Optional runtime override for supervisor model name.",
    )
    parser.add_argument(
        "--intern-model",
        default=None,
        help="Optional runtime override for intern model name.",
    )
    parser.add_argument(
        "--reviewer-model",
        default=None,
        help="Optional runtime override for reviewer model name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        settings = load_settings(args.config)
        if args.supervisor_model:
            settings.supervisor.model = args.supervisor_model
        if args.intern_model:
            settings.intern.model = args.intern_model
        if args.reviewer_model:
            settings.reviewer.model = args.reviewer_model

        job_description = read_file_text(args.job_description_file).strip()
        if not job_description:
            raise ValueError("Job description file is empty.")

        documents = load_documents(args.documents)
        chunks = build_chunks(
            documents=documents,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        if not chunks:
            raise ValueError("No chunks were generated from candidate documents.")

        vector_store = LocalVectorStore(settings.embeddings_model)
        index_base = Path(args.index_path)

        if args.reuse_index:
            try:
                vector_store.load(index_base)
            except VectorStoreError:
                vector_store.build(chunks)
                vector_store.save(index_base)
        else:
            vector_store.build(chunks)
            vector_store.save(index_base)

        llm_client = MultiProviderLLMClient()
        orchestrator = ResumeOrchestrator(
            settings=settings,
            vector_store=vector_store,
            intern=InternAgent(llm=llm_client, settings=settings),
            reviewer=ReviewerAgent(llm=llm_client, settings=settings),
            supervisor=SupervisorAgent(llm=llm_client, settings=settings),
        )

        result = orchestrator.run(job_description)
        _write_outputs(args.output, args.report_output, result)

        print(f"Final resume written to: {Path(args.output).resolve()}")
        print(f"Run report written to: {Path(args.report_output).resolve()}")

    except (ConfigError, DocumentLoadError, VectorStoreError, LLMClientError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


def _write_outputs(resume_path: str, report_path: str, result) -> None:
    resume_target = Path(resume_path)
    resume_target.parent.mkdir(parents=True, exist_ok=True)
    resume_target.write_text(result.final_resume, encoding="utf-8")

    report_target = Path(report_path)
    report_target.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "draft_resume": result.draft_resume,
        "final_resume": result.final_resume,
        "review_rounds": [item.to_dict() for item in result.review_rounds],
        "supervisor_rounds": [item.to_dict() for item in result.supervisor_rounds],
        "retrieval_hits": [
            {
                "score": hit.score,
                "chunk_id": hit.chunk.chunk_id,
                "source": hit.chunk.source,
                "text": hit.chunk.text,
            }
            for hit in result.retrieval_hits
        ],
    }
    report_target.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
