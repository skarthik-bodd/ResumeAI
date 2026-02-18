# ResumeAI (Multi-Agent RAG Resume Builder)

A simple Python project that creates a customized resume for a job description using:
- RAG over candidate evidence files (`.docx`, `.pdf`, `.md`, `.txt`)
- A 3-agent loop:
1. Supervisor/Orchestrator agent
2. Intern writer agent
3. Big-tech reviewer agent

The intern drafts a resume, reviewer gives feedback, and supervisor decides whether the intern should revise.

**This is a work in progress project, I will add/update more concepts and techniques along my learning path**

## Architecture

1. Document Ingestion
- Reads candidate documents from files/directories.
- Supports `.md`, `.txt`, `.pdf`, `.docx`.

2. RAG Layer
- Splits documents into chunks.
- Creates embeddings with `sentence-transformers`.
- Uses cosine similarity retrieval for top-k relevant chunks.

3. Agent Loop
- Intern agent drafts a resume from retrieved context + job description.
- Reviewer agent outputs strict JSON feedback.
- Supervisor agent decides `accept` vs `revise` and sends focus points.
- Intern revises until accepted or max rounds reached.

## Model Recommendations (Free/Open Models)

These are strong local defaults for fun prototyping with Ollama:

1. Supervisor (`qwen2.5:14b`)
- Good instruction following and planning.
- Helps orchestration decisions stay consistent.

2. Intern (`llama3.1:8b`)
- Fast, decent writing quality, lower hardware cost.
- Good for iterative drafting and edits.

3. Reviewer (`deepseek-r1:14b`)
- Stronger reasoning and critique quality.
- Useful for tough feedback loops.

If your machine is smaller, use `qwen2.5:7b` for all three agents first.

Or for local testing use configs from lightweight.yaml:
```supervisor:
  provider: ollama
  model: qwen2.5:3b
  temperature: 0.1

intern:
  provider: ollama
  model: qwen2.5:3b
  temperature: 0.3

reviewer:
  provider: ollama
  model: qwen2.5:3b
  temperature: 0.1
```

## Quick Start

### 1. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### 2. Start Ollama and pull models

```bash
ollama serve
ollama pull qwen2.5:14b
ollama pull llama3.1:8b
ollama pull deepseek-r1:14b
```

### 3. Prepare input files

- Job description file: for example `data/job_description.md`
- Candidate docs directory: for example `data/candidate_docs/`

### 4. Run

```bash
resume-ai \
  --config configs/default.yaml \
  --job-description-file data/job_description.md \
  --documents data/candidate_docs \
  --supervisor-model qwen2.5:14b \
  --intern-model llama3.1:8b \
  --reviewer-model deepseek-r1:14b \
  --output outputs/resume.md \
  --report-output outputs/run_report.json
```

## CLI Options

- `--config`: YAML config path.
- `--job-description-file`: job description file (`.md/.txt/.pdf/.docx`).
- `--documents`: one or more files/directories with candidate evidence.
- `--output`: output markdown resume.
- `--report-output`: JSON report with retrieval and agent rounds.
- `--index-path`: base path for cached vector index.
- `--reuse-index`: reuse existing index if available.
- `--supervisor-model`: override supervisor model name at runtime.
- `--intern-model`: override intern model name at runtime.
- `--reviewer-model`: override reviewer model name at runtime.

## Example Config

```yaml
embeddings_model: BAAI/bge-small-en-v1.5
chunk_size: 1200
chunk_overlap: 200
top_k: 8
max_revision_rounds: 2

supervisor:
  provider: ollama
  model: qwen2.5:14b
  temperature: 0.1

intern:
  provider: ollama
  model: llama3.1:8b
  temperature: 0.4

reviewer:
  provider: ollama
  model: deepseek-r1:14b
  temperature: 0.1
```

## Notes

- This is intentionally simple and not production hardened.
- It does not perform factual verification beyond using supplied evidence context.
- You can extend `src/resume_ai/llm.py` to support non-Ollama providers.
