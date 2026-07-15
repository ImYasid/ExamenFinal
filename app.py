import ast
import math

import streamlit as st
import os
import pandas as pd
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from google import genai
from google.genai import types
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from google.genai.errors import ServerError

# ======================================================
# 1. Configuración de la página y API Key
# ======================================================
st.set_page_config(
    page_title="RAG arXiv - Examen Final",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Un poco de CSS propio: badges de categorías y barra de relevancia más legibles.
st.markdown(
    """
    <style>
    .categoria-badge {
        display: inline-block;
        background-color: #EEF2FF;
        color: #4338CA;
        border-radius: 999px;
        padding: 2px 10px;
        margin: 2px 4px 2px 0;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .fuente-titulo {
        font-weight: 600;
        margin-bottom: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

EJEMPLOS_PREGUNTAS = [
    "What are the main applications of Graph Neural Networks?",
    "How is reinforcement learning used in robotics?",
    "Recent advances in diffusion models for image generation.",
    "Techniques for improving retrieval-augmented generation systems.",
]

# Cargar API key (soporta entorno local .env y Streamlit Cloud st.secrets)
load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY and "GEMINI_API_KEY" in st.secrets:
    API_KEY = st.secrets["GEMINI_API_KEY"]

if not API_KEY:
    st.error(
        "🔑 No se encontró **GEMINI_API_KEY**. Defínela en un archivo `.env` local "
        "(`GEMINI_API_KEY=...`) o, si la app está desplegada en Streamlit Cloud, en "
        "**Settings → Secrets**."
    )
    st.stop()

client = genai.Client(api_key=API_KEY)


# ======================================================
# 2. Carga de Modelos y Base de Datos (Optimizado)
# ======================================================
@st.cache_resource(show_spinner="Preparando el motor de búsqueda (solo la primera vez)...")
def cargar_recursos():
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    df_corpus = pd.read_csv("corpus_reducido.csv")
    doc_embeddings = np.load("embeddings_cache.npy")

    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(
        name="arxiv_abstracts",
        metadata={"hnsw:space": "cosine"},
    )

    # Poblar (o repoblar si el corpus cambió) evita servir resultados desactualizados
    # sin que el usuario tenga forma de notarlo.
    ids_list = df_corpus["doc_id"].tolist()
    existing_ids = set(collection.get(include=[])["ids"]) if collection.count() > 0 else set()
    if set(ids_list) != existing_ids:
        if existing_ids:
            collection.delete(ids=list(existing_ids))
        embeddings_list = doc_embeddings.tolist()
        documents_list = df_corpus["content"].tolist()
        metadatas_list = df_corpus[["title", "terms"]].astype(str).to_dict("records")

        batch_size = 2000
        for i in range(0, len(ids_list), batch_size):
            fin = i + batch_size
            collection.add(
                ids=ids_list[i:fin],
                embeddings=embeddings_list[i:fin],
                documents=documents_list[i:fin],
                metadatas=metadatas_list[i:fin],
            )

    return emb_model, reranker, collection, len(df_corpus)


model_local, reranker, collection, n_documentos = cargar_recursos()


# ======================================================
# 3. Lógica del Sistema RAG
# ======================================================
def recuperar_documentos(query: str, top_k: int = 5, fetch_k: int = 20):
    query_emb = model_local.encode([query]).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=fetch_k)

    candidatos = []
    for doc_id, distance, document, metadata in zip(
        results["ids"][0], results["distances"][0], results["documents"][0], results["metadatas"][0]
    ):
        candidatos.append(
            {
                "doc_id": doc_id,
                "similitud_vectorial": round(1 - distance, 4),
                "titulo": metadata.get("title", ""),
                "categorias": metadata.get("terms", ""),
                "fragmento": document,
            }
        )

    pares_rerank = [[query, doc["fragmento"]] for doc in candidatos]
    scores = reranker.predict(pares_rerank)

    for i, doc in enumerate(candidatos):
        doc["score_rerank"] = float(scores[i])
        # El cross-encoder devuelve un logit sin acotar; lo convertimos a un
        # porcentaje 0-100% (sigmoide) solo para que sea legible en la interfaz.
        doc["relevancia_pct"] = round(1 / (1 + math.exp(-doc["score_rerank"])) * 100, 1)

    candidatos_ordenados = sorted(candidatos, key=lambda x: x["score_rerank"], reverse=True)
    return candidatos_ordenados[:top_k]


@retry(
    retry=retry_if_exception_type(ServerError),
    wait=wait_exponential(multiplier=2, min=2, max=10),
    stop=stop_after_attempt(4),
)
def _llamar_gemini(prompt: str, instrucciones: str):
    return client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=instrucciones, temperature=0.2),
    )


def generar_respuesta(query: str, evidencias: list[dict]) -> str:
    contexto = "\n\n".join(
        f"[{e['doc_id']}] Título: {e['titulo']}\nAbstract: {e['fragmento']}" for e in evidencias
    )
    prompt = f"Contexto recuperado:\n\n{contexto}\n\n---\n\nPregunta: {query}"

    instrucciones = (
        "Eres un experto científico. Responde ÚNICAMENTE con el contexto. "
        "Si no hay info suficiente, dilo explícitamente. Cita usando [doc_id]."
    )

    response = _llamar_gemini(prompt, instrucciones)
    return response.text


def parsear_categorias(raw) -> list[str]:
    """Convierte "['cs.LG', 'cs.CV']" (string) en una lista limpia de categorías."""
    if isinstance(raw, list):
        return raw
    try:
        parsed = ast.literal_eval(str(raw))
        if isinstance(parsed, list):
            return [str(t).strip() for t in parsed]
    except (ValueError, SyntaxError):
        pass
    return [t.strip() for t in str(raw).split(",") if t.strip()]


def mostrar_evidencias(evidencias: list[dict]):
    """Muestra cada fuente con su score de relevancia, categorías y el abstract completo,
    para que se pueda verificar la respuesta contra la evidencia real."""
    with st.expander(f"📚 Ver {len(evidencias)} evidencias utilizadas", expanded=False):
        for e in evidencias:
            st.markdown(f"<div class='fuente-titulo'>{e['titulo']}</div>", unsafe_allow_html=True)

            badges = "".join(
                f"<span class='categoria-badge'>{c}</span>" for c in parsear_categorias(e["categorias"])
            )
            st.markdown(badges, unsafe_allow_html=True)

            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.progress(min(max(e["relevancia_pct"] / 100, 0.0), 1.0))
            with col_b:
                st.caption(f"Relevancia: {e['relevancia_pct']:.0f}%")

            st.write(e["fragmento"])
            st.caption(f"ID interno: `{e['doc_id']}`")
            st.divider()


# ======================================================
# 4. Barra lateral: contexto del sistema, ejemplos y control de la conversación
# ======================================================
with st.sidebar:
    st.header("📚 RAG arXiv")
    st.caption("Sistema de Recuperación Aumentada por Generación con re-ranking.")
    st.caption("Elaborado por Yasid Jiménez")

    st.divider()
    st.markdown("**Cómo funciona**")
    st.markdown(
        "1. Se buscan los pasajes más similares por embeddings.\n"
        "2. Un *cross-encoder* reordena los mejores candidatos.\n"
        "3. Gemini genera la respuesta citando solo esa evidencia."
    )

    st.divider()
    m1, m2 = st.columns(2)
    m1.metric("Documentos indexados", f"{n_documentos:,}")
    m2.metric("Modelo LLM", "Gemini 2.5 Flash")

    st.divider()
    st.markdown("**💡 Prueba con una pregunta:**")
    for i, ejemplo in enumerate(EJEMPLOS_PREGUNTAS):
        if st.button(ejemplo, key=f"ejemplo_{i}", use_container_width=True):
            st.session_state["pregunta_pendiente"] = ejemplo
            st.rerun()

    st.divider()
    if st.button("🗑️ Nueva conversación", use_container_width=True):
        st.session_state.mensajes = []
        st.rerun()


# ======================================================
# 5. Interfaz del Chat (Historial y UI)
# ======================================================
st.title("Chat RAG — arXiv Paper Abstracts")
st.caption(
    "Pregunta en lenguaje natural sobre el corpus de abstracts de arXiv. "
    "Cada respuesta muestra la evidencia usada para construirla."
)

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

if not st.session_state.mensajes:
    st.info(
        "👋 Empieza escribiendo una pregunta abajo, o elige uno de los ejemplos en el panel "
        "izquierdo. El sistema solo responde con base en el corpus indexado y te lo dirá "
        "explícitamente si no encuentra información suficiente."
    )

for msg in st.session_state.mensajes:
    avatar = "🧑‍💻" if msg["rol"] == "user" else "📚"
    with st.chat_message(msg["rol"], avatar=avatar):
        st.markdown(msg["contenido"])
        if "evidencias" in msg:
            mostrar_evidencias(msg["evidencias"])

# La pregunta puede venir del cuadro de chat o de un botón de ejemplo en la barra lateral
pregunta_usuario = st.chat_input("Escribe tu consulta sobre arXiv (ej. What are Graph Neural Networks?)...")
if not pregunta_usuario and "pregunta_pendiente" in st.session_state:
    pregunta_usuario = st.session_state.pop("pregunta_pendiente")

if pregunta_usuario:
    st.chat_message("user", avatar="🧑‍💻").markdown(pregunta_usuario)
    st.session_state.mensajes.append({"rol": "user", "contenido": pregunta_usuario})

    with st.chat_message("assistant", avatar="📚"):
        try:
            with st.spinner("Buscando en el corpus y analizando..."):
                evidencias_recuperadas = recuperar_documentos(pregunta_usuario, top_k=5)
                respuesta_generada = generar_respuesta(pregunta_usuario, evidencias_recuperadas)

            st.markdown(respuesta_generada)
            mostrar_evidencias(evidencias_recuperadas)

            st.session_state.mensajes.append(
                {
                    "rol": "assistant",
                    "contenido": respuesta_generada,
                    "evidencias": evidencias_recuperadas,
                }
            )
        except Exception as exc:  # noqa: BLE001 - error de red/API, no queremos tumbar la app
            mensaje_error = (
                "⚠️ No se pudo generar una respuesta en este momento "
                "(problema de conexión con la API de Gemini). Intenta de nuevo en unos segundos."
            )
            st.error(mensaje_error)
            with st.expander("Detalle técnico del error"):
                st.code(str(exc))
            st.session_state.mensajes.append({"rol": "assistant", "contenido": mensaje_error})