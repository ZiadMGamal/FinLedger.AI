# FinLedger.AI

FinLedger.AI is an enterprise-grade Retrieval-Augmented Generation platform for financial document intelligence, designed for DocFinQA-scale long-context SEC filings with dense markdown and tabular structures.

## Project Ownership

- System: FinLedger.AI
- Created by: Ziad Mohamed Gamal

## Phase-Driven Build Plan

1. Phase 1: Project architecture setup and configuration management
2. Phase 2: Ingestion and parsing pipeline for DocFinQA JSON
3. Phase 3: Optimized retrieval and RAG core
4. Phase 4: Ragas evaluation engine
5. Phase 5: FastAPI backend and Streamlit frontend
6. Phase 6: Docker and docker-compose multi-service deployment

## Repository Structure

```text
FinLedger.AI/
├── api/
│   ├── __init__.py
│   └── main.py
├── data/
│   ├── interim/
│   ├── processed/
│   └── raw/
├── src/
│   ├── __init__.py
│   ├── evaluation/
│   │   └── __init__.py
│   ├── pipeline/
│   │   └── __init__.py
│   ├── retrieval/
│   │   └── __init__.py
│   └── core/
│       ├── __init__.py
│       ├── config.py
│       └── logging.py
├── tests/
│   └── __init__.py
├── ui/
│   ├── __init__.py
│   └── app.py
├── .env.example
├── requirements.txt
└── README.md
```
