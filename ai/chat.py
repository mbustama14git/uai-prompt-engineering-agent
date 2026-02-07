# ai/chat.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import os
import time

import config
from bd.chroma_store import search as chroma_search

# OpenAI (opcional)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# -----------------------------
# Template Prompt 
# -----------------------------
SYSTEM_PROMPT_ES = """\
Eres un asistente técnico senior de metalurgia y procesamiento de minerales en una planta concentradora.
Responde en español, con tono profesional y directo.
Usa únicamente la información del CONTEXTO proporcionado. Si el contexto no contiene la respuesta, dilo explícitamente.
No inventes datos ni números.
Cuando cites información, indica la fuente como [source: <archivo>] al final de la frase o párrafo.
"""

USER_TEMPLATE_ES = """\
PREGUNTA DEL USUARIO:
{question}

CONTEXTO (fragmentos recuperados):
{context}

INSTRUCCIONES:
1) Responde de forma clara y resumida.
2) Si faltan datos en el contexto, indica qué falta.
3) Si hay recomendaciones operacionales, enuméralas.
RESPUESTA:
"""


# -----------------------------
# Configuración
# -----------------------------
DEFAULT_TOP_K = 3
DEFAULT_MAX_CHARS_PER_DOC = 2000


@dataclass
class RagHit:
    id: str
    text: str
    source: str
    distance: float


@dataclass
class RagResult:
    question: str
    hits: List[RagHit]
    context: str


def _has_openai_key() -> bool:
    key = getattr(config, "gpt_key", None) or os.getenv("OPENAI_API_KEY")
    return bool(key and str(key).strip())


def _build_rag_context(question: str, top_k: int, max_chars_per_doc: int) -> RagResult:
    raw_hits = chroma_search(question, top_k=top_k)

    hits: List[RagHit] = []
    context_parts: List[str] = []

    for h in raw_hits:
        meta = h.get("metadata") or {}
        src = meta.get("source", "") or meta.get("src", "") or "unknown"
        txt = (h.get("text") or "")[:max_chars_per_doc]
        dist = h.get("distance", 0.0)
        _id = h.get("id", "")

        hits.append(RagHit(id=_id, text=txt, source=src, distance=float(dist)))

        context_parts.append(
            f"[source: {src} | id={_id} | distance={float(dist):.4f}]\n{txt}"
        )

    context = "\n\n---\n\n".join(context_parts)

    return RagResult(question=question, hits=hits, context=context)


def _render_prompt(question: str, context: str) -> Dict[str, str]:
    """
    Retorna el template ya renderizado (para debug y transparencia).
    """
    system = SYSTEM_PROMPT_ES
    user = USER_TEMPLATE_ES.format(question=question, context=context)
    return {"system": system, "user": user}


def _answer_without_llm(rag: RagResult, prompt_rendered: Dict[str, str]) -> str:
    """
    Modo SIN LLM: devuelve un formato pedagógico para validar retrieval + prompt final.
    """
    system_txt = prompt_rendered.get("system", "")
    user_txt = prompt_rendered.get("user", "")

    # Métricas simples para entender tamaño del prompt
    system_chars = len(system_txt)
    user_chars = len(user_txt)
    total_chars = system_chars + user_chars

    lines = []
    lines.append("MODO SIN LLM ✅ (solo retrieval + prompt renderizado)\n")

    lines.append("### Pregunta")
    lines.append(rag.question)

    lines.append("\n### Top documentos recuperados")
    if not rag.hits:
        lines.append("- (sin resultados) La base vectorial no retornó documentos para esta consulta.")
    else:
        for i, h in enumerate(rag.hits, 1):
            lines.append(f"{i}. source={h.source} | id={h.id} | distance={h.distance:.4f}")

    lines.append("\n### Tamaño del prompt (aprox)")
    lines.append(f"- SYSTEM chars: {system_chars}")
    lines.append(f"- USER chars:   {user_chars}")
    lines.append(f"- TOTAL chars:  {total_chars}")

    lines.append("\n### Prompt final que se enviaría al LLM")
    lines.append("#### SYSTEM")
    lines.append(system_txt)

    lines.append("\n#### USER")
    lines.append(user_txt)

    # Opcional: también puedes dejar el contexto aparte si quieres verlo “sin plantilla”
    lines.append("\n### Contexto usado (solo contexto, sin plantilla)")
    lines.append(rag.context if rag.context else "(vacío)")

    return "\n".join(lines)



def _answer_with_llm(prompt_rendered: Dict[str, str], model: str = "gpt-4o-mini") -> str:
    """
    Modo CON LLM: llama a OpenAI con el prompt armado.
    """
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK no está disponible. Instala 'openai' en el venv.")

    key = getattr(config, "gpt_key", None) or os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=key)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt_rendered["system"]},
            {"role": "user", "content": prompt_rendered["user"]},
        ],
        temperature=0.2,
    )

    return resp.choices[0].message.content


def generate_text(
    prompt: str,
    chat_id: str,
    *,
    use_llm: Optional[bool] = None,
    top_k: int = DEFAULT_TOP_K,
    max_chars_per_doc: int = DEFAULT_MAX_CHARS_PER_DOC,
    model: str = "gpt-4o-mini",
    debug: bool = False,
) -> Any:
    """
    Función principal para tu endpoint /messages.

    - prompt: string (ej "pregunta: ..."). Para claridad, aquí lo tratamos como pregunta.
    - chat_id: id conversación (lo dejamos por compatibilidad)
    - use_llm:
        - None => auto (si hay gpt_key usa LLM, si no, modo SIN LLM)
        - True => fuerza LLM
        - False => fuerza SIN LLM
    - debug:
        - False => retorna solo string
        - True  => retorna dict con respuesta + debug (hits/context/prompt/latencias)
    """
    t0 = time.time()

    # Limpieza simple: si viene "pregunta: X", nos quedamos con X
    question = prompt.strip()
    if question.lower().startswith("pregunta:"):
        question = question.split(":", 1)[1].strip()

    rag = _build_rag_context(question, top_k=top_k, max_chars_per_doc=max_chars_per_doc)
    prompt_rendered = _render_prompt(question, rag.context)

    # Decide modo
    if use_llm is None:
        use_llm = _has_openai_key()

    answer: str
    mode: str

    if not use_llm:
        mode = "no_llm"
        answer = _answer_without_llm(rag, prompt_rendered)
    else:
        mode = "llm"
        answer = _answer_with_llm(prompt_rendered, model=model)

    elapsed = time.time() - t0

    if not debug:
        return answer

    # Debug payload bien claro para UI/curso
    return {
        "mode": mode,
        "chat_id": chat_id,
        "question": question,
        "top_k": top_k,
        "max_chars_per_doc": max_chars_per_doc,
        "model": model if use_llm else None,
        "latency_s": round(elapsed, 3),
        "hits": [
            {"id": h.id, "source": h.source, "distance": h.distance, "text_preview": h.text[:300]}
            for h in rag.hits
        ],
        "context": rag.context,
        "prompt": prompt_rendered,
        "answer": answer,
    }
