
import os
import json
from typing import TypedDict, List

import chromadb
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import PromptTemplate


# ============================================================
# Configuration
# ============================================================

MOCK_LLM = os.getenv("MOCK_LLM", "1")

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ============================================================
# Pydantic Models
# ============================================================

class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)


class SupportResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


# ============================================================
# LangGraph State
# ============================================================

class SupportState(TypedDict, total=False):
    query: str
    intent: str
    retrieved_documents: List[str]
    retrieved_ids: List[str]
    answer: str
    sources: List[str]
    confidence: float


# ============================================================
# Embedding Model + ChromaDB
# ============================================================

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)


# ============================================================
# Structured Prompt
# ============================================================

prompt_template = PromptTemplate(
    input_variables=["context", "query"],
    template="""
ROLE:
You are a Zepto customer-support assistant. You answer questions using only
Zepto's provided policy documents.

CONTEXT:
{context}

TASK:
Answer the customer's question using the provided context.
If the answer is not available in the context, clearly say that the
provided Zepto policies do not contain the required information.

FORMAT:
Return a JSON object with exactly these fields:
{{
  "answer": "string",
  "sources": ["document/chunk IDs"],
  "confidence": 0.0
}}

LENGTH:
Keep the answer concise and directly relevant to the customer's question,
using no more than 3 sentences.

NEGATIVE CONSTRAINT:
Do not use information that is not present in the provided context.
Do not invent or assume Zepto policies.

FEW-SHOT EXAMPLE:

Example question:
"What is the delivery fee for an order below INR 149?"

Example context:
"Standard delivery is free on orders over INR 149; orders below this
threshold incur a flat INR 25 delivery fee."

Example answer:
{{
  "answer": "Orders below INR 149 incur a flat INR 25 standard delivery fee.",
  "sources": ["doc_01"],
  "confidence": 1.0
}}

CUSTOMER QUESTION:
{query}

Now answer using only the provided context.
"""
)


# ============================================================
# Optional Real LLM
# ============================================================

def create_real_llm():
    from langchain_groq import ChatGroq

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is required when MOCK_LLM=0."
        )

    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )


def parse_llm_response(raw_response: str) -> SupportResponse:
    cleaned = raw_response.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "", 1)
        cleaned = cleaned.replace("```", "", 1).strip()

    data = json.loads(cleaned)

    return SupportResponse(**data)


def generate_with_retry(prompt: str, max_attempts: int = 3):
    llm = create_real_llm()

    current_prompt = prompt
    last_error = None

    for attempt in range(max_attempts):
        try:
            result = llm.invoke(current_prompt)
            raw_response = result.content

            return parse_llm_response(raw_response)

        except Exception as error:
            last_error = error

            current_prompt = prompt + f"""

CORRECTION:
Your previous response was invalid.

Error:
{error}

Return ONLY valid JSON with exactly these fields:
{{
  "answer": "string",
  "sources": ["document/chunk IDs"],
  "confidence": 0.0
}}
"""

    return SupportResponse(
        answer=(
            f"ERROR: Unable to generate a valid structured response "
            f"after {max_attempts} attempts."
        ),
        sources=[],
        confidence=0.0
    )


# ============================================================
# LangGraph Node 1 — Classify Intent
# ============================================================

def classify_intent(state: SupportState):
    query = state["query"].lower()

    policy_keywords = [
        "delivery",
        "return",
        "refund",
        "membership",
        "tracking",
        "cancel",
        "gift card",
        "support hours"
    ]

    if any(keyword in query for keyword in policy_keywords):
        intent = "policy_question"
    else:
        intent = "general_question"

    return {
        "intent": intent
    }


# ============================================================
# LangGraph Node 2 — Retrieve and Answer
# ============================================================

def retrieve_and_answer(state: SupportState):
    query = state["query"]

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    documents = results["documents"][0]
    ids = results["ids"][0]

    context_parts = []

    for doc_id, document in zip(ids, documents):
        context_parts.append(
            f"[{doc_id}]\n{document}"
        )

    context = "\n\n".join(context_parts)

    # Required deterministic mock mode
    if MOCK_LLM != "0":
        top_chunk_snippet = documents[0][:200]

        return {
            "retrieved_documents": documents,
            "retrieved_ids": ids,
            "answer": f"Based on the retrieved context: {top_chunk_snippet}",
            "sources": ids,
            "confidence": 1.0
        }

    # Optional real LLM mode
    formatted_prompt = prompt_template.format(
        context=context,
        query=query
    )

    llm_response = generate_with_retry(formatted_prompt)

    return {
        "retrieved_documents": documents,
        "retrieved_ids": ids,
        "answer": llm_response.answer,
        "sources": ids,
        "confidence": llm_response.confidence
    }


# ============================================================
# LangGraph Node 3 — Direct Answer
# ============================================================

def direct_answer(state: SupportState):
    query = state["query"]

    # Required deterministic mock mode
    if MOCK_LLM != "0":
        return {
            "answer": (
                "I can only answer questions about Zepto policies right now."
            ),
            "sources": [],
            "confidence": 1.0
        }

    # Optional real LLM mode
    direct_prompt = f"""
ROLE:
You are a Zepto customer-support assistant.

TASK:
Answer the customer's question directly.

IMPORTANT:
Only answer questions about Zepto policies.
If the question is not about a Zepto policy, clearly state that
you can only answer Zepto policy questions.

FORMAT:
Return a JSON object with exactly these fields:
{{
  "answer": "string",
  "sources": [],
  "confidence": 0.0
}}

LENGTH:
Keep the answer concise and use no more than 3 sentences.

CUSTOMER QUESTION:
{query}
"""

    llm_response = generate_with_retry(direct_prompt)

    return {
        "answer": llm_response.answer,
        "sources": [],
        "confidence": llm_response.confidence
    }


# ============================================================
# LangGraph Routing
# ============================================================

def route_after_classify(state: SupportState):
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


workflow = StateGraph(SupportState)

workflow.add_node("classify_intent", classify_intent)
workflow.add_node("retrieve_and_answer", retrieve_and_answer)
workflow.add_node("direct_answer", direct_answer)

workflow.add_edge(START, "classify_intent")

workflow.add_conditional_edges(
    "classify_intent",
    route_after_classify,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer"
    }
)

workflow.add_edge("retrieve_and_answer", END)
workflow.add_edge("direct_answer", END)

support_graph = workflow.compile()


# ============================================================
# Ask Function
# ============================================================

def ask_question(query: str) -> SupportResponse:
    result = support_graph.invoke({
        "query": query
    })

    return SupportResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        confidence=result.get("confidence", 0.0)
    )


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="Zepto Support Assistant",
    description=(
        "Zepto policy support assistant using LangGraph, "
        "ChromaDB and local embeddings."
    ),
    version="1.0.0"
)


@app.post("/ask", response_model=SupportResponse)
def ask_endpoint(request: AskRequest):
    try:
        return ask_question(request.query)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


@app.get("/")
def root():
    return {
        "message": "Zepto Support Assistant is running.",
        "mock_llm": MOCK_LLM
    }
