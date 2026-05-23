"""
rag_pipeline.py
Core RAG engine — extracted from the notebook, no Streamlit dependency.
Imported by app.py.
"""

import os
import re
import uuid
import tempfile
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import fitz  # PyMuPDF
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
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

        # Year — from PDF metadata, fallback to first-page text scan
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
        self.model: SentenceTransformer | None = None

    def load(self, status_callback=None):
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
# 4. VECTOR STORE
# ─────────────────────────────────────────────────────────────

class VectorStore:
    """ChromaDB with cosine distance — thresholds map to 0.0–1.0 similarity."""

    COLLECTION = "medical_papers"

    def __init__(self, persist_dir: str = "./data/vector_store"):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION,
            metadata={
                "description": "Medical research paper embeddings",
                "hnsw:space": "cosine",
            },
        )

    def count(self) -> int:
        return self.collection.count()

    def indexed_files(self) -> set:
        """Return set of source_file names already in the store."""
        if self.count() == 0:
            return set()
        results = self.collection.get(include=["metadatas"])
        return {m.get("source_file", "") for m in results["metadatas"]}

    def add_documents(self, documents: list, embeddings: np.ndarray):
        ids, metadatas, texts, vecs = [], [], [], []
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            ids.append(f"doc_{uuid.uuid4().hex[:8]}_{i}")
            meta = dict(doc.metadata)
            meta["doc_index"]      = i
            meta["content_length"] = len(doc.page_content)
            metadatas.append(meta)
            texts.append(doc.page_content)
            vecs.append(emb.tolist())
        self.collection.add(ids=ids, embeddings=vecs, metadatas=metadatas, documents=texts)

    def query(self, query_embedding: np.ndarray, top_k: int = 5):
        return self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self.count()) if self.count() > 0 else 1,
        )

    def delete_file(self, source_file: str):
        """Remove all chunks belonging to a specific file."""
        results = self.collection.get(
            where={"source_file": source_file},
            include=["metadatas"],
        )
        if results["ids"]:
            self.collection.delete(ids=results["ids"])


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
        self.vs  = vector_store
        self.em  = embedding_manager

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
# 6. RAG PIPELINE (ADVANCED)
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

    # Build cited context
    ctx_parts = []
    for doc in docs:
        m = doc["metadata"]
        header = f"[Source {doc['rank']}: {m.get('authors','Unknown')}, {m.get('year','Unknown')} — {m.get('paper_title', m.get('source_file','Unknown'))}]"
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
    status_callback=None,
) -> Dict[str, Any]:
    """
    Full pipeline: bytes → chunks → embeddings → ChromaDB.
    Returns info dict with page/chunk counts.
    """
    # Write to temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        if status_callback:
            status_callback(f"📄 Loading {filename}…")
        pages = process_pdf_file(tmp_path)
        # Rename source_file to the original uploaded name
        for doc in pages:
            doc.metadata["source_file"] = filename

        if status_callback:
            status_callback(f"✂️  Splitting into chunks…")
        chunks = split_documents(pages)

        if status_callback:
            status_callback(f"🧠 Generating embeddings for {len(chunks)} chunks…")
        texts      = [c.page_content for c in chunks]
        embeddings = embedding_manager.embed(texts)

        if status_callback:
            status_callback(f"💾 Indexing into ChromaDB…")
        vector_store.add_documents(chunks, embeddings)

        return {
            "filename": filename,
            "pages":    len(pages),
            "chunks":   len(chunks),
        }
    finally:
        os.unlink(tmp_path)
