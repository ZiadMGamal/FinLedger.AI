# FinLedger.AI

[![CI](https://github.com/<your-username>/FinLedger.AI/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/FinLedger.AI/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)](https://streamlit.io/)
[![VectorDB](https://img.shields.io/badge/VectorDB-Qdrant-DC244C)](https://qdrant.tech/)

FinLedger.AI is an enterprise-grade Retrieval-Augmented Generation platform for financial document intelligence, designed for DocFinQA-scale SEC filings that combine long text and dense markdown tables.

## Project Identity

- System name: `FinLedger.AI`
- Created with: `Ziad Mohamed Gamal`
- Focus: production-level financial RAG + evaluation pipeline

## Core Stack

- Language: Python 3.10+
- Orchestration: LangChain
- Vector DB: Qdrant (primary) or FAISS
- LLM and embeddings: OpenAI (`gpt-4o-mini`, `text-embedding-3-small`)
- Evaluation: Ragas
- Backend: FastAPI
- Frontend: Streamlit
- Deployment: Docker and Docker Compose

## Key Features

- DocFinQA-specific ingestion for long SEC contexts
- Table-aware chunking that preserves markdown/tabular blocks
- Hybrid retrieval (vector + keyword BM25)
- Selectable retrieval mode: parent-child or contextual compression
- Citation-grounded financial response generation
- Automated evaluation with Faithfulness, Answer Relevancy, Context Recall
- Production API + dashboard + containerized runtime

## Repository Layout

```text
FinLedger.AI/
├── api/
│   ├── main.py
│   └── schemas.py
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── scripts/
│   ├── run_stack.ps1
│   ├── run_ingestion.ps1
│   └── stop_stack.ps1
├── src/
│   ├── core/
│   ├── pipeline/
│   ├── retrieval/
│   └── evaluation/
├── ui/
│   └── app.py
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.ui
├── requirements.txt
└── .env.example
```

## Quick Start

### 1) Environment

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set your key in `.env`:

```env
OPENAI_API_KEY=sk-...
```

### 2) Ingest DocFinQA

Place your dataset file at `data/raw/docfinqa_train.json`, then run:

```powershell
python -m src.pipeline.run_ingestion ingest --input-path data/raw/docfinqa_train.json
```

### 3) Run API and UI

```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

```powershell
streamlit run ui/app.py --server.port 8501
```

### 4) Docker Compose

```powershell
.\scripts\run_stack.ps1 -Detached
.\scripts\run_ingestion.ps1
```

UI: `http://127.0.0.1:8501`  
API docs: `http://127.0.0.1:8000/docs`

## API Endpoints

- `GET /health`
- `POST /query`
- `POST /evaluate`

## Screenshots

Add your screenshots to `assets/screenshots/` and update this section with real captures.

### Streamlit Assistant

![Streamlit Assistant](assets/screenshots/streamlit-assistant.png)

### Evaluation Dashboard

![Evaluation Dashboard](assets/screenshots/evaluation-dashboard.png)

### API Swagger

![API Swagger](assets/screenshots/api-swagger.png)

## CI

GitHub Actions workflow is included at `.github/workflows/ci.yml` and runs:

- dependency installation
- `ruff` static checks
- syntax compile check across `src`, `api`, and `ui`

Before publishing, replace `<your-username>` in the badge URL with your GitHub username.
