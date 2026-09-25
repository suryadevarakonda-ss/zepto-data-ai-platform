# Zepto Data & AI Platform — Support Assistant

## Overview

This module implements a Zepto customer-support assistant using:

* LangGraph for workflow orchestration
* Sentence Transformers for local embeddings
* `all-MiniLM-L6-v2` for semantic embeddings
* ChromaDB for vector storage and retrieval
* Pydantic for structured response validation
* FastAPI for the `/ask` API
* Docker for local containerized execution

The required baseline runs fully offline and deterministically with:

```text
MOCK_LLM=1
```

When `MOCK_LLM` is not set, the application defaults to `1`.

## Module Contents

```text
support_assistant/
├── main.py
├── ingest.py
├── requirements.txt
├── Dockerfile
├── README.md
└── docs/
    ├── doc_01.txt
    ├── doc_02.txt
    └── doc_03.txt
```

The `docs/` directory contains the policy corpus used for retrieval.

## Pipeline Architecture

The support assistant uses a LangGraph workflow:

```text
Customer Question
       |
       v
Classify Intent
       |
       +----------------------+
       |                      |
       v                      v
Policy Question         General Question
       |                      |
       v                      v
Retrieve Top 3          Direct Answer
Documents                    |
       |                      |
       +----------+-----------+
                  |
                  v
             Final Response
```

### Step 1 — Intent Classification

The `classify_intent` node checks the customer question for policy-related keywords such as:

* delivery
* return
* refund
* membership
* tracking
* cancel
* gift card
* support hours

Questions containing these keywords are routed to the retrieval path.

Other questions are routed to the direct-answer path.

### Step 2 — Retrieval

For policy questions:

1. The query is converted into an embedding using `all-MiniLM-L6-v2`.
2. ChromaDB performs cosine-similarity retrieval.
3. The top 3 matching policy documents/chunks are retrieved.
4. The retrieved document IDs are returned as sources.

### Step 3 — Answer Generation

With the required default `MOCK_LLM=1`, the system produces a deterministic response using the retrieved context.

For general questions, the deterministic response states that the assistant can only answer questions about Zepto policies.

An optional real-LLM path is available when `MOCK_LLM=0`.

## Example Call Transcripts

The following examples use the required default:

```text
MOCK_LLM=1
```

### Example 1 — Policy Question

Request:

```http
POST /ask
Content-Type: application/json

{
  "query": "What is the delivery fee?"
}
```

Example response:

```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del",
  "sources": [
    "doc_01",
    "doc_05",
    "doc_02"
  ],
  "confidence": 1
}
```

The exact retrieved document/chunk ID and snippet depend on the indexed corpus and retrieval result.

### Example 2 — Another Policy Question

Request:

```http
POST /ask
Content-Type: application/json

{
  "query": "What is the refund policy?"
}
```

Example response:

```json
{
  "answer": "Based on the retrieved context: Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect; non-perishable packaged items may be returned within 7 days of delivery in unop",
  "sources": [
    "doc_02",
    "doc_06",
    "doc_05"
  ],
  "confidence": 1
}
```

The exact retrieved document/chunk ID and snippet depend on the indexed corpus and retrieval result.

### Example 3 — General Question

Request:

```http
POST /ask
Content-Type: application/json

{
  "query": "Tell me a joke"
}
```

Response:

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1
}
```

## Structured Response

The API returns a validated response with exactly these fields:

```json
{
  "answer": "string",
  "sources": ["document/chunk IDs"],
  "confidence": 0.0
}
```

The `confidence` value is constrained between `0.0` and `1.0` by Pydantic validation.

## FastAPI API

The main endpoint is:

```text
POST /ask
```

Request body:

```json
{
  "query": "What is the delivery policy?"
}
```

The root endpoint is:

```text
GET /
```

It returns a status message and the current `MOCK_LLM` setting.

## Default Mock Mode

The required baseline uses:

```text
MOCK_LLM=1
```

This is also the default value when the environment variable is not provided.

Mock mode is deterministic and does not require an external LLM API key.

## Optional Real LLM Mode

The application contains an optional Groq-based real-LLM path.

It can be enabled with:

```text
MOCK_LLM=0
```

This mode requires:

```text
GROQ_API_KEY
```

The real-LLM path uses the `llama-3.1-8b-instant` model and validates the response against the Pydantic `SupportResponse` schema.

The baseline submission does not require real-LLM usage.

## Running Locally

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
uvicorn main:app --reload
```

The API can then be accessed through the FastAPI server.

## Docker

Build the Docker image:

```bash
docker build -t zepto-support-assistant .
```

Run the container:

```bash
docker run -p 8000:8000 zepto-support-assistant
```

## Design Notes

* The baseline is designed to run without an external LLM service.
* Policy answers use retrieved corpus context.
* Retrieval returns the top 3 matching documents/chunks.
* Sources are included in the structured response.
* Pydantic validates the final response structure.
* LangGraph controls the classification and routing workflow.
* FastAPI provides the HTTP API.
* Docker provides a containerized execution option.

## Optional Extensions

Real-LLM usage and a hosted Hugging Face Space are optional extensions.

No real-LLM usage notes or Hugging Face Space URL are included here unless those optional extensions are actually attempted.

