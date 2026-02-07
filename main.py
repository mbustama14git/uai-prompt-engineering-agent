# main.py
# FastAPI app v2: incluye /search y /rag_debug (Chroma local) + tu endpoint /messages
# - Mantiene tu estructura actual
# - No revienta si no tienes gpt_key: /messages funcionará igual si tu generate_text no depende de OpenAI
#   (si tu generate_text usa OpenAI, puedes comentarlo o manejar el error dentro de generate_text)

from openai import OpenAI
import uvicorn

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse

import config
from ai.chat import generate_text

# Chroma local (persistente) -> tu módulo bd/chroma_store.py
from bd.chroma_store import search as chroma_search

# -------------------------
# OpenAI client (opcional)
# -------------------------
# Si no tienes gpt_key, deja config.gpt_key vacío y no se usará acá directamente.
client = OpenAI(api_key=getattr(config, "gpt_key", None))

app = FastAPI()

# CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Health / Root
# -------------------------
@app.get("/")
async def init():
    return "hi"

@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------------
# NEW: Retrieval endpoints
# -------------------------
@app.post("/search")
def search_endpoint(payload: dict = Body(...)):
    """
    Entrada esperada:
    {
      "query": "texto ...",
      "top_k": 3
    }

    Salida:
    {
      "query": "...",
      "top_k": 3,
      "hits": [ {id,text,metadata,distance}, ...]
    }
    """
    query = payload.get("query", "")
    top_k = int(payload.get("top_k", 3))

    try:
        hits = chroma_search(query, top_k=top_k)
        return {"query": query, "top_k": top_k, "hits": hits}
    except Exception as e:
        print(e)
        return JSONResponse(status_code=500, content={"error": f"Error en /search: {str(e)}"})


@app.post("/rag_debug")
def rag_debug_endpoint(payload: dict = Body(...)):
    """
    Entrada esperada:
    {
      "query": "texto ...",
      "top_k": 3,
      "max_chars": 2500
    }

    Salida incluye:
    - hits (top-k)
    - context (texto concatenado para RAG, útil para modo curso)
    """
    query = payload.get("query", "")
    top_k = int(payload.get("top_k", 3))
    max_chars = int(payload.get("max_chars", 2500))

    try:
        hits = chroma_search(query, top_k=top_k)

        context_parts = []
        for h in hits:
            src = (h.get("metadata") or {}).get("source", "")
            txt = (h.get("text") or "")[:max_chars]
            dist = h.get("distance", None)
            # dist puede venir None si no lo devuelve la implementación
            dist_str = f"{dist:.4f}" if isinstance(dist, (int, float)) else str(dist)

            context_parts.append(
                f"[SOURCE: {src} | id={h.get('id')} | dist={dist_str}]\n{txt}"
            )

        context = "\n\n---\n\n".join(context_parts)

        return {
            "query": query,
            "top_k": top_k,
            "hits": hits,
            "context": context
        }

    except Exception as e:
        print(e)
        return JSONResponse(status_code=500, content={"error": f"Error en /rag_debug: {str(e)}"})

# -------------------------
# Existing: Messages endpoint
# -------------------------
@app.post("/messages")
def messages(payload: dict = Body(...)):
    """
    Mantiene tu contrato actual:
    {
      "type": "...",
      "message": {
        "chat": {"id": "..."},
        "text": "..."
      }
    }
    """
    try:
        chat_id = payload["message"]["chat"]["id"]
        _type_app = payload.get("type", "web")
        message = payload["message"]["text"]

        prompt = f"pregunta: {message}"

        # Tu función actual (puede o no usar OpenAI internamente)
        response = generate_text(prompt, chat_id)

        return {"response": response}

    except Exception as e:
        print(e)
        return JSONResponse(
            status_code=500,
            content={"error": "Error interno del servidor"}
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
