from __future__ import annotations
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer

# Ruta absoluta estable al directorio chroma_db (al lado del proyecto)
CHROMA_DIR = str((Path(__file__).resolve().parents[1] / "chroma_db").resolve())
COLLECTION_NAME = "kb_concentradora"

print("CHROMA_DIR =", CHROMA_DIR)

embedding_fn = SentenceTransformerEmbeddingFunction(
    #model_name="sentence-transformers/all-MiniLM-L6-v2"
    model_name="BAAI/bge-large-en-v1.5"
)

# 👇 ESTA es la forma persistente recomendada
_client = chromadb.PersistentClient(path=CHROMA_DIR)

_collection = _client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}
)

def upsert_docs(
    ids: List[str],
    texts: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None
) -> None:
    if metadatas is None:
        metadatas = [{} for _ in ids]

    _collection.upsert(ids=ids, documents=texts, metadatas=metadatas)

    # En algunas versiones, esto asegura flush a disco
    try:
        _client.persist()
    except Exception:
        pass

#def search(query: str, top_k: int = 3, *, distance_threshold: float | None = 0.45, min_chars_query: int = 6):

def search(query: str, top_k: int = 3, distance_threshold: float | None = 0.5) -> List[Dict[str, Any]]:
    res = _collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    out = []
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    for i in range(len(ids)):
        dist = float(dists[i])
        if distance_threshold is not None and dist > distance_threshold:
            continue
        out.append({
            "id": ids[i],
            "text": docs[i],
            "metadata": metas[i],
            "distance": dists[i],
        })
    return out

def count() -> int:
    return _collection.count()

def debug_collections() -> List[str]:
    cols = _client.list_collections()
    return [c.name for c in cols]
