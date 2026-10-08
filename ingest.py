"""Build the local Chroma vector store from knowledge_base.py using a local
Ollama embedding model. Run this once, and again whenever knowledge_base.py changes.
"""
import requests
import chromadb

from knowledge_base import DOCUMENTS

OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "tresvance_kb"


def embed(text: str) -> list[float]:
    resp = requests.post(OLLAMA_URL, json={"model": EMBED_MODEL, "prompt": text})
    resp.raise_for_status()
    return resp.json()["embedding"]


def main():
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    ids, embeddings, texts = [], [], []
    for doc in DOCUMENTS:
        print(f"Embedding: {doc['id']}")
        ids.append(doc["id"])
        texts.append(doc["text"])
        embeddings.append(embed(doc["text"]))

    collection.add(ids=ids, embeddings=embeddings, documents=texts)
    print(f"Ingested {len(ids)} documents into '{COLLECTION_NAME}' at {DB_PATH}")


if __name__ == "__main__":
    main()
