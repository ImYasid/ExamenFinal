# Examen Final: Sistema RAG sobre arXiv Paper Abstracts

**Institución:** Escuela Politécnica Nacional (EPN) - Facultad de Ingeniería de Sistemas (FIS)  
**Estudiante:** Yasid Jimenez Jaramillo  
**Docente:** Iván Carrera  

---

## 🚀 Descripción del Examen

Este examen implementa un sistema avanzado de **Recuperación Aumentada por Generación (RAG)** diseñado para responder preguntas en lenguaje natural sobre un corpus de resúmenes científicos de **arXiv** (enfocado en Inteligencia Artificial, Machine Learning, Robótica y áreas afines). 

La solución destaca por un diseño híbrido optimizado: **procesamiento de embeddings y re-ranking de forma local** para garantizar velocidad y evadir límites de cuota, combinado con la potencia de **Google Gemini** para la generación de la respuesta final.

### 🔗 URL de la Aplicación Desplegada
> **[👉 HAZ CLIC AQUÍ PARA IR AL CHAT RAG EN VIVO](https://examenfinal-brhfuwckpm323gs3amanwl.streamlit.app/)**  
> *(Reemplaza este enlace con tu URL real de Streamlit Cloud)*

---

## 🛠️ Arquitectura del Sistema (Pipeline RAG)

El sistema procesa cada consulta del usuario a través de un flujo moderno de dos fases más generación:

1. **Recuperación Vectorial (Fase 1 - Filtro Grueso):**
   * Se utiliza el modelo local de Hugging Face `all-MiniLM-L6-v2` para codificar la consulta del usuario en un vector de 384 dimensiones.
   * Se realiza una búsqueda por similitud de coseno en la base de datos vectorial local **ChromaDB**, recuperando los **20 documentos candidatos** más cercanos del corpus precalculado de 9,000+ papers.
2. **Re-ranking Semántico (Fase 2 - Filtro Fino):**
   * Se re-ordenan los 20 candidatos utilizando el modelo Cross-Encoder `ms-marco-MiniLM-L-6-v2`. Esto asegura que los **5 documentos más relevantes** y con mayor coincidencia contextual real pasen a la fase de generación.
3. **Generación Aumentada por Recuperación (Fase 3):**
   * Los abstracts seleccionados se inyectan en un prompt estructurado con instrucciones de sistema estrictas (System Instructions).
   * **Gemini 2.5 Flash** procesa el prompt para redactar una respuesta fundamentada, citando las fuentes mediante identificadores (ej. `[doc_123]`) y declarando explícitamente cuando el corpus no tiene información suficiente para mitigar alucinaciones.

---

## 📁 Estructura del Repositorio

```text
ExamenFinal/
│
├── app.py                  # Aplicación interactiva de chat (Streamlit)
├── ExamenFinal.ipynb       # Jupyter Notebook con el desarrollo paso a paso
├── corpus_reducido.csv     # Dataset optimizado con ~9,000 papers (IA/ML)
├── embeddings_cache.npy    # Embeddings vectoriales precalculados de los documentos
├── requirements.txt        # Dependencias necesarias para ejecutar el proyecto
├── .gitignore              # Exclusiones de Git (evita subir credenciales y archivos pesados)
└── README.md               # Este archivo de documentación
