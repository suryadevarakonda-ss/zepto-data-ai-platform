
import os
import chromadb
from sentence_transformers import SentenceTransformer


DOCS_DIR = "docs"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "zepto_policies"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def ingest_documents():
    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    print("Connecting to ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    documents = []
    ids = []
    metadatas = []

    for filename in sorted(os.listdir(DOCS_DIR)):
        if filename.endswith(".txt"):
            file_path = os.path.join(DOCS_DIR, filename)

            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read().strip()

            document_id = filename.replace(".txt", "")

            documents.append(text)
            ids.append(document_id)
            metadatas.append({
                "source": file_path
            })

    if not documents:
        raise ValueError("No policy documents found in the docs directory.")

    print(f"Found {len(documents)} documents.")

    embeddings = embedding_model.encode(
        documents,
        normalize_embeddings=True
    ).tolist()

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"Successfully ingested {len(documents)} documents.")
    print(f"ChromaDB collection: {COLLECTION_NAME}")
    print(f"Collection count: {collection.count()}")


if __name__ == "__main__":
    ingest_documents()
