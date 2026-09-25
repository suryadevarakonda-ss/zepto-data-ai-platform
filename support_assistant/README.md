
# Zepto Data & AI Platform — Support Assistant

## Overview

This project implements a Zepto customer-support assistant using:

- LangGraph for workflow orchestration
- Sentence Transformers for local embeddings
- `all-MiniLM-L6-v2` for semantic embeddings
- ChromaDB for vector storage and retrieval
- Pydantic for structured response validation
- FastAPI for the `/ask` API
- Docker for local containerized execution

The required baseline runs fully offline and deterministically with:

```text
MOCK_LLM=1
