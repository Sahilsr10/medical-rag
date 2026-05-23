# 🏥 Medical Research RAG

A production-ready Retrieval-Augmented Generation (RAG) system for biomedical research papers. Upload PDFs, ask questions, get cited answers grounded only in your papers.

---

## Features

- **Drag-and-drop PDF upload** — any number of research papers
- **PubMedBERT embeddings** — domain-tuned on 33M PubMed abstracts (768-dim)
- **ChromaDB cosine retrieval** — interpretable similarity thresholds
- **Citation-enforced answers** — every claim cited as [Authors, Year]
- **Confidence score** — mean cosine similarity of retrieved chunks
- **Persistent vector store** — re-uploads skip re-indexing automatically
- **Adjustable retrieval settings** — top-k and similarity threshold sliders
- **Chat history** — previous questions kept in session

---

## Project Structure

```
medical_rag_app/
├── app.py               # Streamlit UI
├── rag_pipeline.py      # Core RAG engine (no Streamlit dependency)
├── requirements.txt
├── .env.example         # Copy to .env and fill in GROQ_API_KEY
├── .gitignore
├── .streamlit/
│   ├── config.toml      # Theme + server settings
│   └── secrets.toml.example
└── data/
    └── vector_store/    # ChromaDB (auto-created, gitignored)
```

---

## Quick Start (Local)

### 1. Clone & install

```bash
git clone https://github.com/your-username/medical-rag.git
cd medical-rag
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and paste your Groq API key
# Free key: https://console.groq.com
```

### 3. Run

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---


> **Note on persistence**: Streamlit Cloud has an ephemeral filesystem — the ChromaDB store resets on each app restart. Users will need to re-upload their PDFs after a restart. For persistent storage, see the Railway/Render section below.

---

## Deploy to Railway (Persistent storage)

Railway gives you a persistent volume so the vector store survives restarts.

```bash
# Install Railway CLI
npm install -g @railway/cli
railway login

# From your project folder
railway init
railway up
```

In the Railway dashboard:
- Set environment variable: `GROQ_API_KEY=gsk_...`
- Set start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

---

## API Key

This app uses [Groq](https://console.groq.com) for LLM inference (free tier available).

Model used: `llama-3.1-8b-instant` — fast and accurate for biomedical Q&A.

---

## Retrieval Settings

| Setting | Meaning |
|---|---|
| **top-k** | How many chunks to fetch from ChromaDB before threshold filter |
| **Similarity threshold (0.45)** | Cosine distance cutoff. Lower = more chunks (higher recall). Higher = fewer but more precise. |

Threshold guide:
- `0.30` → similarity ≥ 70% (strict, high precision)
- `0.45` → similarity ≥ 55% (recommended default)
- `0.60` → similarity ≥ 40% (lenient, high recall)
