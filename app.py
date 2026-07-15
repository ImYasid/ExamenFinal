import streamlit as st
import os
import pandas as pd
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
st.set_page_config(page_title="RAG arXiv - Examen Final", page_icon="📚", layout="wide")
st.title("📚 Chat RAG — arXiv Paper Abstracts")
st.caption("Sistema de Recuperación Aumentada por Generación (RAG) con Re-ranking. Elaborado por Yasid Jimenez.")

# Cargar API key (Soporta entorno local (.env) y la nube de Streamlit (st.secrets))
load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY and "GEMINI_API_KEY" in st.secrets:
    API_KEY = st.secrets["GEMINI_API_KEY"]

if not API_KEY:
    st.error("No se encontró la API Key de Gemini.")
    st.stop()

client = genai.Client(api_key=API_KEY)

# ======================================================
# 2. Carga de Modelos y Base de Datos
# ======================================================
@st.cache_resource(show_spinner="Preparando el motor de búsqueda (esto toma unos segundos la primera vez)...")
def cargar_recursos():
    # 1. Cargar modelos
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # 2. Cargar el corpus REDUCIDO (súper ligero)
    df_corpus = pd.read_csv("corpus_reducido.csv")
    
    # 3. Inicializar ChromaDB (en la nube, lo creará desde cero)
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(
        name="arxiv_abstracts", 
        metadata={"hnsw:space": "cosine"}
    )
    
    # 4. Poblar la base de datos SOLO si está vacía
    if collection.count() == 0:
        # Generar embeddings locales on-the-fly
        doc_embeddings = emb_model.encode(df_corpus["content"].tolist(), batch_size=32)
        
        # Guardar en ChromaDB
        collection.add(
            ids=df_corpus["doc_id"].tolist(),
            embeddings=doc_embeddings.tolist(),
            documents=df_corpus["content"].tolist(),
            metadatas=df_corpus[["title", "terms"]].astype(str).to_dict("records")
        )
        
    return emb_model, reranker, collection

model_local, reranker, collection = cargar_recursos()

# ======================================================
# 3. Lógica del Sistema RAG
# ======================================================
def recuperar_documentos(query: str, top_k: int = 5, fetch_k: int = 20):
    # Fase 1: Búsqueda Vectorial
    query_emb = model_local.encode([query]).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=fetch_k)

    candidatos = []
    for doc_id, distance, document, metadata in zip(
        results["ids"][0], results["distances"][0], results["documents"][0], results["metadatas"][0]
    ):
        candidatos.append({
            "doc_id": doc_id, "similitud_vectorial": round(1 - distance, 4),
            "titulo": metadata.get("title", ""), "categorias": metadata.get("terms", ""),
            "fragmento": document
        })

    # Fase 2: Re-ranking
    pares_rerank = [[query, doc["fragmento"]] for doc in candidatos]
    scores = reranker.predict(pares_rerank)
    
    for i, doc in enumerate(candidatos):
        doc["score_rerank"] = float(scores[i])
        
    candidatos_ordenados = sorted(candidatos, key=lambda x: x["score_rerank"], reverse=True)
    return candidatos_ordenados[:top_k]

@retry(retry=retry_if_exception_type(ServerError), wait=wait_exponential(multiplier=2, min=2, max=10), stop=stop_after_attempt(4))
def generar_respuesta(query: str, evidencias: list[dict]):
    contexto = "\n\n".join(f"[{e['doc_id']}] Título: {e['titulo']}\nAbstract: {e['fragmento']}" for e in evidencias)
    prompt = f"Contexto recuperado:\n\n{contexto}\n\n---\n\nPregunta: {query}"
    
    instrucciones = (
        "Eres un experto científico. Responde ÚNICAMENTE con el contexto. "
        "Si no hay info suficiente, dilo explícitamente. Cita usando [doc_id]."
    )
    
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt,
        config=types.GenerateContentConfig(system_instruction=instrucciones, temperature=0.2)
    )
    return response.text

# ======================================================
# 4. Interfaz del Chat (Historial y UI)
# ======================================================
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

# Mostrar historial
for msg in st.session_state.mensajes:
    with st.chat_message(msg["rol"]):
        st.markdown(msg["contenido"])
        # Si el mensaje tiene evidencias, las mostramos en un acordeón
        if "evidencias" in msg:
            with st.expander("📚 Ver evidencias recuperadas"):
                st.dataframe(pd.DataFrame(msg["evidencias"])[["doc_id", "score_rerank", "titulo", "categorias"]])

# Input del usuario
pregunta_usuario = st.chat_input("Escribe tu consulta sobre arXiv (ej. What are Graph Neural Networks?)...")

if pregunta_usuario:
    # 1. Mostrar mensaje del usuario
    st.chat_message("user").markdown(pregunta_usuario)
    st.session_state.mensajes.append({"rol": "user", "contenido": pregunta_usuario})

    # 2. Generar y mostrar respuesta del sistema
    with st.chat_message("assistant"):
        with st.spinner("Buscando en la base de datos y analizando..."):
            evidencias_recuperadas = recuperar_documentos(pregunta_usuario, top_k=5)
            respuesta_generada = generar_respuesta(pregunta_usuario, evidencias_recuperadas)
            
            st.markdown(respuesta_generada)
            with st.expander("📚 Ver evidencias recuperadas"):
                st.dataframe(pd.DataFrame(evidencias_recuperadas)[["doc_id", "score_rerank", "titulo", "categorias"]])
                
    # 3. Guardar en historial
    st.session_state.mensajes.append({
        "rol": "assistant", 
        "contenido": respuesta_generada, 
        "evidencias": evidencias_recuperadas
    })