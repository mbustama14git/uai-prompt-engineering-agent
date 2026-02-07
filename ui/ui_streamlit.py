import streamlit as st
import requests
import time

API_BASE_DEFAULT = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG Agente Metalúrgico", layout="wide")
st.title("🤖 Agente Metalúrgico")

# -----------------------------
# Session state
# -----------------------------
if "chat_id" not in st.session_state:
    st.session_state.chat_id = "local_chat_1"
if "history" not in st.session_state:
    st.session_state.history = []  # list of (role, text)
if "last_debug" not in st.session_state:
    st.session_state.last_debug = None

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.subheader("Conexión API")
    api_base = st.text_input("API_BASE", API_BASE_DEFAULT)
    st.session_state.chat_id = st.text_input("chat_id", st.session_state.chat_id)

    st.subheader("Modo")
    mode = st.radio("Respuesta", ["SIN LLM", "CON LLM"], index=0)

    st.subheader("Retrieval")
    top_k = st.slider("top_k", 1, 10, 3)
    max_chars = st.slider("max_chars por doc", 300, 6000, 2000, step=100)

    st.subheader("Debug")
    show_payload = st.checkbox("Mostrar payload enviado", value=True)
    show_hits = st.checkbox("Mostrar top-k hits", value=True)
    show_context = st.checkbox("Mostrar contexto armado", value=True)

    st.divider()
    if st.button("🧹 Limpiar chat"):
        st.session_state.history = []
        st.session_state.last_debug = None
        st.rerun()

# -----------------------------
# Layout
# -----------------------------
col_chat, col_debug = st.columns([1.2, 1])

with col_chat:
    st.subheader("Chat")
    for role, text in st.session_state.history:
        with st.chat_message(role):
            st.markdown(text)

    user_msg = st.chat_input("Escribe tu mensaje…")

# -----------------------------
# Helpers
# -----------------------------
def post_json(url: str, payload: dict, timeout: int = 120):
    t0 = time.time()
    r = requests.post(url, json=payload, timeout=timeout)
    dt = time.time() - t0
    return r, dt

def build_prompt_template(question: str, context: str) -> dict:
    system = (
        "Eres un asistente técnico senior de metalurgia y procesamiento de minerales en una planta concentradora.\n"
        "Responde en español, con tono profesional y directo.\n"
        "Usa únicamente la información del CONTEXTO proporcionado. Si el contexto no contiene la respuesta, dilo explícitamente.\n"
        "No inventes datos, números ni equipos.\n"
        "Cuando cites información, indica la fuente como [source: <archivo>] al final de la frase o párrafo.\n"
    )
    user = (
        f"PREGUNTA DEL USUARIO:\n{question}\n\n"
        f"CONTEXTO (fragmentos recuperados):\n{context}\n\n"
        "INSTRUCCIONES:\n"
        "1) Responde de forma clara y accionable.\n"
        "2) Si faltan datos en el contexto, indica qué falta.\n"
        "3) Si hay recomendaciones operacionales, enuméralas.\n"
        "RESPUESTA:\n"
    )
    return {"system": system, "user": user}

def prompt_metrics(prompt_obj: dict) -> dict:
    sys_txt = prompt_obj.get("system", "")
    usr_txt = prompt_obj.get("user", "")
    return {
        "system_chars": len(sys_txt),
        "user_chars": len(usr_txt),
        "total_chars": len(sys_txt) + len(usr_txt),
    }

def looks_like_no_llm_reply(text: str) -> bool:
    t = (text or "").lower()
    return ("modo sin llm" in t) or ("no se llama al modelo" in t) or ("prompt renderizado" in t) or ("llm no disponible" in t)

# -----------------------------
# On user message
# -----------------------------
if user_msg:
    # User message
    st.session_state.history.append(("user", user_msg))
    with col_chat:
        with st.chat_message("user"):
            st.markdown(user_msg)

    payload_messages = {
        "type": "web",
        "message": {
            "chat": {"id": st.session_state.chat_id},
            "text": user_msg
        }
    }

    payload_debug = {
        "query": user_msg,
        "top_k": top_k,
        "max_chars": max_chars
    }

    # Always fetch rag_debug (needed to build the prompt + show retrieval on right)
    rag_url = f"{api_base}/rag_debug"
    r_rag, dt_rag = post_json(rag_url, payload_debug)
    rag_data = r_rag.json() if r_rag.status_code == 200 else None

    context = (rag_data or {}).get("context", "")
    prompt_obj = build_prompt_template(user_msg, context) if rag_data else None
    metrics = prompt_metrics(prompt_obj) if prompt_obj else None

    if mode == "SIN LLM":
        with col_chat:
            with st.chat_message("assistant"):
                if rag_data is None:
                    st.error(f"Error /rag_debug {r_rag.status_code}: {r_rag.text}")
                else:
                    reply = (
                        "⚠️ **LLM no disponible (modo SIN LLM)**.\n\n"
                        "Abajo se muestra el **prompt** que se enviaría al modelo (SYSTEM + USER), "
                        "para revisar longitud y evidencia.\n\n"
                        f"**Prompt chars** → SYSTEM: {metrics['system_chars']} | USER: {metrics['user_chars']} | TOTAL: {metrics['total_chars']}\n\n"
                        "#### SYSTEM\n"
                        f"```text\n{prompt_obj['system']}\n```\n"
                        "#### USER\n"
                        f"```text\n{prompt_obj['user']}\n```"
                    )
                    st.markdown(reply)
                    st.session_state.history.append(("assistant", reply))

        st.session_state.last_debug = {
            "mode": "SIN LLM",
            "endpoint": "/rag_debug",
            "latency_s": dt_rag,
            "payload": payload_debug,
            "rag_debug_payload": payload_debug,
            "rag_debug": rag_data,
        }

    else:
        url = f"{api_base}/messages"
        r_msg, dt_msg = post_json(url, payload_messages)

        with col_chat:
            with st.chat_message("assistant"):
                if r_msg.status_code != 200:
                    st.error(f"Error {r_msg.status_code}: {r_msg.text}")
                else:
                    data = r_msg.json()
                    reply = data.get("response", "")

                    # If backend fell back to no-llm, show prompt in chat (as requested)
                    if looks_like_no_llm_reply(reply):
                        if rag_data is None:
                            fallback = "⚠️ **LLM no disponible (modo CON LLM)**, y además falló /rag_debug, así que no puedo armar el prompt."
                            st.warning(fallback)
                            st.session_state.history.append(("assistant", fallback))
                        else:
                            fallback = (
                                "⚠️ **LLM no disponible (modo CON LLM)**. El backend devolvió un fallback sin modelo.\n\n"
                                "📌 Te dejo el **prompt aquí en el chat** (SYSTEM + USER):\n\n"
                                f"**Prompt chars** → SYSTEM: {metrics['system_chars']} | USER: {metrics['user_chars']} | TOTAL: {metrics['total_chars']}\n\n"
                                "#### SYSTEM\n"
                                f"```text\n{prompt_obj['system']}\n```\n"
                                "#### USER\n"
                                f"```text\n{prompt_obj['user']}\n```"
                            )
                            st.markdown(fallback)
                            st.session_state.history.append(("assistant", fallback))
                    else:
                        st.markdown(reply)
                        st.session_state.history.append(("assistant", reply))

        st.session_state.last_debug = {
            "mode": "CON LLM",
            "endpoint": "/messages",
            "latency_s": dt_msg,
            "payload": payload_messages,
            "rag_debug_payload": payload_debug,
            "rag_debug": rag_data,
        }

# -----------------------------
# Debug panel (right) - ONLY hits + context
# -----------------------------
with col_debug:
    st.subheader("Panel Modo Pruebas")

    dbg = st.session_state.last_debug
    if not dbg:
        st.info("Envía un mensaje para ver aquí el debug del retrieval/contexto.")
    else:
        st.metric("Latencia (s)", f"{dbg.get('latency_s', 0):.2f}")
        st.caption(f"Modo: {dbg.get('mode')}")

        if show_payload:
            st.markdown("### Payload(s)")
            st.code(dbg.get("payload", {}), language="json")
            if dbg.get("rag_debug_payload"):
                st.code(dbg.get("rag_debug_payload", {}), language="json")

        rag_debug = dbg.get("rag_debug")
        if rag_debug:
            if show_hits:
                st.markdown("### Documentos recuperados (top-k)")
                hits = rag_debug.get("hits", [])
                if not hits:
                    st.warning("No hay hits (colección vacía o query sin match).")
                else:
                    for i, h in enumerate(hits, 1):
                        src = (h.get("metadata") or {}).get("source", "")
                        st.markdown(
                            f"**{i}. id:** `{h.get('id')}`  \n"
                            f"**dist:** `{h.get('distance')}`  \n"
                            f"**source:** `{src}`"
                        )
                        with st.expander("Ver texto (preview)"):
                            st.write((h.get("text") or "")[:1200])

            if show_context:
                st.markdown("### Contexto armado (concatenado)")
                st.text_area("context", value=rag_debug.get("context", ""), height=320)
        else:
            st.warning("No hay datos de /rag_debug para mostrar (¿API accesible?).")
