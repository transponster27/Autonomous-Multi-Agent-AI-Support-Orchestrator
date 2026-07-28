# Single-Agent RAG Baseline

> A foundational, local-first Retrieval-Augmented Generation (RAG) pipeline featuring Hybrid Search (FAISS + BM25) and Cross-Encoder Reranking. 

*Note: This `main` branch represents the initial single-agent baseline of the project. For the advanced multi-agent orchestration, validation loops, and modern CI/CD setup, please see the `feature/multi-agent-pipeline` branch.*

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview
This repository contains the baseline architecture for a local RAG system. It allows users to upload documents (PDF, DOCX, TXT, Images), index them using dense and sparse retrieval, and query them using a local LLM via Ollama.

## Setup & Installation
This project uses `uv` for fast, reproducible dependency management.

```bash
# 1. Clone and install uv
git clone <repo-url>
cd <repo-name>
pip install uv

# 2. Sync dependencies
uv sync --all-extras

# 3. Run the application
uv run uvicorn api:app --reload --host 0.0.0.0 --port 8000
uv run streamlit run app.py