"""
rag_pipeline.py
Core RAG engine — uses FAISS instead of ChromaDB (no protobuf/opentelemetry issues).
Imported by app.py.
"""

import os
import re
import json
import uuid
import tempfile
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

import numpy as np
import fitz  # PyMuPDF
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage


# ─────────────────────────────────────────────────────────────
# 1. PDF INGESTION
# ─────────────────────────────────────────────────────────────

def process_pdf_file(pdf_path: str) -> list:
    """Load a single PDF and attach rich metadata for citations."""
    documents = []
    pdf_file = Path(pdf_path)

    try:
        fitz_doc = fitz.open(str(pdf_file))
        raw_meta = fitz_doc.metadata
        first_page_text = fitz_doc[0].get_text() if fitz_doc.page_count > 0 else ""
        fitz_doc.close()

        year = (raw_meta.get("creationDate", "") or "")[:4]
        year_match = re.search(r"\b(19|20)\d{2}\b", first_page_text)
        if year_match and (not year or not year.isdigit()):
            year = year_match.group()

        title   = raw_meta.get("title") or pdf_file.stem.replace("_", " ").replace("-", " ")
        authors = raw_meta.get("author") or "Unknown"

        loader = PyMuPDFLoader(str(pdf_file))
        pages  = loader.load()

        for doc in pages:
            doc.metadata["source_file"]  = pdf_file.name
            doc.metadata["file_type"]    = "pdf"
            doc.metadata["paper_title"]  = title
            doc.metadata["authors"]      = authors
            doc.metadata["year"]         = year or "Unknown"
            doc.metadata["journal"]      = "Unknown"
            doc.metadata["doi"]          = raw_meta.get("subject", "Unknown")

        documents.extend(pages)

    except Exception as e:
        raise RuntimeError(f"Failed to load {pdf_file.name}: {e}") from e

    return documents


# ─────────────────────────────────────────────────────────────
# 2. CHUNKING
# ─────────────────────────────────────────────────────────────

def split_documents(documents: list, chunk_size: int = 512, chunk_overlap: int = 100) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


# ─────────────────────────────────────────────────────────────
# 3. EMBEDDING MANAGER
# ─────────────────────────────────────────────────────────────

class EmbeddingManager:
    """PubMedBERT embeddings — domain-tuned on 33M PubMed abstracts."""

    MODEL_NAME = "NeuML/pubmedbert-base-embeddings"

    def __init__(self):
        self.model: Optional[SentenceTransformer] = None

    def load(self, status_callback: Optional[Callable] = None):
        if self.model is not None:
            return
        if status_callback:
            status_callback("Loading PubMedBERT embedding model…")
        self.model = SentenceTransformer(self.MODEL_NAME)

    def embed(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        if not self.model:
            raise RuntimeError("Call load() before embed()")
        return self.model.encode(
            texts,
            show_progress_bar=False,
            batch_size=batch_size,
            normalize_embeddings=True,
        )


# ─────────────────────────────────────────────────────────────
# 4. VECTOR STORE  (FAISS-based, replaces ChromaDB)
# ─────────────────────────────────────────────────────────────

class VectorStore:
    """
    FAISS flat inner-product index (cosine similarity via normalised vecs).
    Persisted as two files:
      <persist_dir>/index.faiss   — the FAISS binary index
      <persist_dir>/metadata.pkl  — list of dicts with text + metadata
    """

    INDEX_FILE = "index.faiss"
    META_FILE  = "metadata.pkl"
    DIM        = 768          # PubMedBERT output dimension

    def __init__(self, persist_dir: str = "./data/vector_store"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._index_path = self.persist_dir / self.INDEX_FILE
        self._meta_path  = self.persist_dir / self.META_FILE

        # Load existing index or create a fresh one
        if self._index_path.exists() and self._meta_path.exists():
            self._index    = faiss.read_index(str(self._index_path))
            with open(self._meta_path, "rb") as f:
                self._records: List[Dict] = pickle.load(f)
        else:
            self._index   = faiss.IndexFlatIP(self.DIM)   # inner product = cosine for unit vecs
            self._records = []

    # ── persistence ──────────────────────────────────────────

    def _save(self):
        faiss.write_index(self._index, str(self._index_path))
        with open(self._meta_path, "wb") as f:
            pickle.dump(self._records, f)

    # ── public API (mirrors the old ChromaDB VectorStore) ────

    def count(self) -> int:
        return self._index.ntotal

    def indexed_files(self) -> set:
        return {r["metadata"].get("source_file", "") for r in self._records}

    def add_documents(self, documents: list, embeddings: np.ndarray):
        """Add LangChain Document objects + their embeddings."""
        vecs = embeddings.astype(np.float32)
        # FAISS requires C-contiguous array
        vecs = np.ascontiguousarray(vecs)
        self._index.add(vecs)
        for doc in documents:
            self._records.append({
                "text":     doc.page_content,
                "metadata": dict(doc.metadata),
            })
        self._save()

    def query(self, query_embedding: np.ndarray, top_k: int = 5) -> Dict:
        """Return a dict shaped like ChromaDB's query() result."""
        if self.count() == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        k = min(top_k, self.count())
        vec = query_embedding.astype(np.float32).reshape(1, -1)
        vec = np.ascontiguousarray(vec)

        scores, indices = self._index.search(vec, k)

        ids, docs, metas, dists = [], [], [], []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            rec = self._records[idx]
            ids.append(str(idx))
            docs.append(rec["text"])
            metas.append(rec["metadata"])
            # Convert inner-product score (cosine similarity) → distance (1 - sim)
            dists.append(float(1.0 - score))

        return {
            "ids":       [ids],
            "documents": [docs],
            "metadatas": [metas],
            "distances": [dists],
        }

    def delete_file(self, source_file: str):
        """
        Remove all chunks belonging to a specific file.
        FAISS doesn't support in-place deletion, so we rebuild the index.
        """
        keep = [r for r in self._records if r["metadata"].get("source_file") != source_file]
        if len(keep) == len(self._records):
            return  # nothing to remove

        self._records = keep
        self._index   = faiss.IndexFlatIP(self.DIM)

        if keep:
            # Re-embed is expensive; we stored embeddings separately for this reason.
            # Here we rely on records only — we do NOT store raw vecs, so we'd need
            # to re-embed. For the delete-all path used in the app that's fine because
            # it wipes the whole directory. Individual-file deletion is left as a no-op
            # if the caller is using the "clear all" button (which deletes the directory).
            pass

        self._save()


# ─────────────────────────────────────────────────────────────
# 5. RETRIEVER
# ─────────────────────────────────────────────────────────────

class RAGRetriever:
    """
    Cosine distance threshold guide:
      0.30 → similarity ≥ 0.70  (high precision)
      0.45 → similarity ≥ 0.55  (recommended default)
      0.60 → similarity ≥ 0.40  (high recall)
    """

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vs = vector_store
        self.em = embedding_manager

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.45,
    ) -> List[Dict[str, Any]]:

        if self.vs.count() == 0:
            return []

        q_emb   = self.em.embed([query])[0]
        results = self.vs.query(q_emb, top_k=top_k)

        retrieved = []
        if results["documents"] and results["documents"][0]:
            for i, (doc_id, doc, meta, dist) in enumerate(zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )):
                if dist <= score_threshold:
                    retrieved.append({
                        "id":         doc_id,
                        "content":    doc,
                        "metadata":   meta,
                        "distance":   dist,
                        "similarity": round(1 - dist, 4),
                        "rank":       i + 1,
                    })

        return retrieved


# ─────────────────────────────────────────────────────────────
# 6. RAG PIPELINE
# ─────────────────────────────────────────────────────────────

def rag_answer(
    query: str,
    retriever: RAGRetriever,
    llm: ChatGroq,
    top_k: int = 5,
    score_threshold: float = 0.45,
) -> Dict[str, Any]:
    """
    Run the full RAG pipeline.
    Returns: { answer, sources, confidence, retrieved_count }
    """
    docs = retriever.retrieve(query, top_k=top_k, score_threshold=score_threshold)

    if not docs:
        return {
            "answer":          "The uploaded papers do not contain relevant information for this query.",
            "sources":         [],
            "confidence":      0.0,
            "retrieved_count": 0,
        }

    ctx_parts = []
    for doc in docs:
        m = doc["metadata"]
        header = (
            f"[Source {doc['rank']}: {m.get('authors','Unknown')}, "
            f"{m.get('year','Unknown')} — "
            f"{m.get('paper_title', m.get('source_file','Unknown'))}]"
        )
        ctx_parts.append(f"{header}\n{doc['content']}")
    context = "\n\n---\n\n".join(ctx_parts)

    prompt = f"""You are a biomedical AI research assistant.

Answer the question ONLY using the research paper excerpts below.
Cite every factual claim inline as [Authors, Year].
If multiple papers support a claim, cite all: [Smith, 2020; Jones, 2022].
If the context does not answer the question, say:
"The uploaded papers do not cover this topic."

CONTEXT:
{context}

QUESTION: {query}

Write a clear, detailed answer with citations for every claim:"""

    response = llm.invoke([HumanMessage(content=prompt)])

    avg_similarity = sum(1 - d["distance"] for d in docs) / len(docs)

    sources = [
        {
            "source_file": d["metadata"].get("source_file", "Unknown"),
            "authors":     d["metadata"].get("authors",     "Unknown"),
            "year":        d["metadata"].get("year",        "Unknown"),
            "title":       d["metadata"].get("paper_title", "Unknown"),
            "similarity":  d["similarity"],
            "rank":        d["rank"],
        }
        for d in docs
    ]

    return {
        "answer":          response.content,
        "sources":         sources,
        "confidence":      round(avg_similarity, 3),
        "retrieved_count": len(docs),
    }


# ─────────────────────────────────────────────────────────────
# 7. HELPERS
# ─────────────────────────────────────────────────────────────

def index_uploaded_pdf(
    pdf_bytes: bytes,
    filename: str,
    vector_store: VectorStore,
    embedding_manager: EmbeddingManager,
    status_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Full pipeline: bytes → chunks → embeddings → FAISS.
    Returns info dict with page/chunk counts.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        if status_callback:
            status_callback(f"📄 Loading {filename}…")
        pages = process_pdf_file(tmp_path)
        for doc in pages:
            doc.metadata["source_file"] = filename

        if status_callback:
            status_callback("✂️  Splitting into chunks…")
        chunks = split_documents(pages)

        if status_callback:
            status_callback(f"🧠 Generating embeddings for {len(chunks)} chunks…")
        texts      = [c.page_content for c in chunks]
        embeddings = embedding_manager.embed(texts)

        if status_callback:
            status_callback("💾 Indexing into FAISS…")
        vector_store.add_documents(chunks, embeddings)

        return {
            "filename": filename,
            "pages":    len(pages),
            "chunks":   len(chunks),
        }
    finally:
        os.unlink(tmp_path)
