"""
app.py  —  Medical Research RAG  |  Streamlit frontend
Run: streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from rag_pipeline import (
    EmbeddingManager,
    VectorStore,
    RAGRetriever,
    rag_answer,
    index_uploaded_pdf,
)

load_dotenv()

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical RAG",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"]          { background: #161b27; border-right: 1px solid #2a2f3e; }

/* ── Header ── */
.rag-header {
    background: linear-gradient(135deg, #1a2744 0%, #0d3b6e 60%, #0a2a52 100%);
    border: 1px solid #2563eb44;
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
}
.rag-header h1 { color: #e2e8f0; font-size: 2rem; margin: 0 0 6px 0; }
.rag-header p  { color: #94a3b8; margin: 0; font-size: 0.95rem; }

/* ── Cards ── */
.answer-card {
    background: #161b27;
    border: 1px solid #2a3650;
    border-left: 4px solid #2563eb;
    border-radius: 12px;
    padding: 22px 26px;
    margin: 16px 0;
    color: #e2e8f0;
    line-height: 1.7;
    font-size: 0.97rem;
}
.source-card {
    background: #0f1624;
    border: 1px solid #1e2d47;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 0.88rem;
}
.source-rank  { color: #60a5fa; font-weight: 700; }
.source-file  { color: #a5b4fc; }
.source-meta  { color: #64748b; font-size: 0.82rem; }

/* ── Confidence bar ── */
.conf-row     { display: flex; align-items: center; gap: 12px; margin: 16px 0 8px; }
.conf-label   { color: #94a3b8; font-size: 0.88rem; white-space: nowrap; }
.conf-bar-bg  { flex: 1; background: #1e2d47; border-radius: 99px; height: 10px; }
.conf-fill    { height: 10px; border-radius: 99px; background: linear-gradient(90deg,#1d4ed8,#3b82f6); }
.conf-value   { color: #93c5fd; font-weight: 600; font-size: 0.88rem; white-space: nowrap; }

/* ── Metric chips ── */
.chip-row { display: flex; gap: 10px; flex-wrap: wrap; margin: 8px 0 18px; }
.chip {
    background: #1e2d47; border: 1px solid #2a3f5f;
    border-radius: 999px; padding: 4px 14px;
    font-size: 0.82rem; color: #93c5fd;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] > div {
    background: #161b27 !important;
    border: 2px dashed #2a3f5f !important;
    border-radius: 12px !important;
}

/* ── Sidebar items ── */
.pdf-item {
    background: #0f1624;
    border: 1px solid #1e2d47;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.85rem;
    color: #cbd5e1;
}
.pdf-item span { color: #64748b; font-size: 0.78rem; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg,#1d4ed8,#2563eb) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    padding: 10px 24px !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* ── Input ── */
.stTextArea textarea, .stTextInput input {
    background: #161b27 !important;
    border: 1px solid #2a3650 !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SESSION-STATE INIT
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_embedding_manager():
    em = EmbeddingManager()
    em.load()
    return em

@st.cache_resource(show_spinner=False)
def get_vector_store():
    return VectorStore(persist_dir="./data/vector_store")

def get_llm(api_key: str) -> ChatGroq:
    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=1024,
    )

if "chat_history"    not in st.session_state: st.session_state.chat_history    = []
if "indexed_files"   not in st.session_state: st.session_state.indexed_files   = set()
if "last_result"     not in st.session_state: st.session_state.last_result     = None


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Medical RAG")
    st.markdown("---")

    # API Key
    st.markdown("### 🔑 Groq API Key")
    env_key = os.getenv("GROQ_API_KEY", "")
    api_key = st.text_input(
        "Enter your Groq API key",
        value=env_key,
        type="password",
        placeholder="gsk_...",
        help="Get a free key at console.groq.com",
    )

    st.markdown("---")

    # PDF Upload
    st.markdown("### 📂 Upload Research Papers")
    uploaded_files = st.file_uploader(
        "Drop PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more biomedical research PDFs",
    )

    if uploaded_files and api_key:
        vs = get_vector_store()
        em = get_embedding_manager()
        already_indexed = vs.indexed_files()

        new_files = [f for f in uploaded_files if f.name not in already_indexed]

        if new_files:
            with st.spinner(f"Indexing {len(new_files)} new file(s)…"):
                for uf in new_files:
                    status_box = st.empty()
                    try:
                        info = index_uploaded_pdf(
                            pdf_bytes=uf.read(),
                            filename=uf.name,
                            vector_store=vs,
                            embedding_manager=em,
                            status_callback=lambda msg: status_box.caption(msg),
                        )
                        st.session_state.indexed_files.add(uf.name)
                        status_box.empty()
                        st.success(f"✅ {uf.name}  ({info['pages']} pages, {info['chunks']} chunks)")
                    except Exception as e:
                        status_box.empty()
                        st.error(f"❌ {uf.name}: {e}")
        else:
            st.info("All uploaded files are already indexed.")

    # Show indexed papers
    vs = get_vector_store()
    indexed = vs.indexed_files()
    if indexed:
        st.markdown("---")
        st.markdown(f"### 📚 Indexed Papers  `{len(indexed)}`")
        for fname in sorted(indexed):
            st.markdown(
                f'<div class="pdf-item">📄 {fname}</div>',
                unsafe_allow_html=True,
            )

        if st.button("🗑️ Clear all papers", use_container_width=True):
            import shutil
            shutil.rmtree("./data/vector_store", ignore_errors=True)
            st.cache_resource.clear()
            st.session_state.indexed_files = set()
            st.session_state.chat_history  = []
            st.rerun()

    st.markdown("---")

    # Settings
    st.markdown("### ⚙️ Retrieval Settings")
    top_k = st.slider("Chunks to retrieve (top-k)", 1, 10, 5)
    threshold = st.slider(
        "Similarity threshold",
        0.10, 0.90, 0.45, 0.05,
        help="Lower = more results (lenient). Higher = fewer but more precise.",
    )


# ─────────────────────────────────────────────────────────────
# MAIN PANEL
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="rag-header">
  <h1>🏥 Medical Research RAG</h1>
  <p>Ask questions grounded in your uploaded biomedical papers — every answer is cited.</p>
</div>
""", unsafe_allow_html=True)

# Status chips
vs    = get_vector_store()
count = vs.count()
cols  = st.columns(3)
cols[0].metric("Papers indexed",  len(vs.indexed_files()))
cols[1].metric("Total chunks",    count)
cols[2].metric("Embedding model", "PubMedBERT-768d")

st.markdown("---")

# ── Guard: need API key + at least one paper ──
if not api_key:
    st.info("👈 Enter your Groq API key in the sidebar to get started.")
    st.stop()

if count == 0:
    st.info("👈 Upload at least one research PDF in the sidebar to begin.")
    st.stop()

# ── Query box ──
st.markdown("### 💬 Ask a Question")
query = st.text_area(
    "Your medical question",
    placeholder="e.g. How does ER stress contribute to heart failure?\ne.g. What are the main findings about lactate in cardiovascular disease?",
    height=100,
    label_visibility="collapsed",
)

col_ask, col_clear = st.columns([1, 5])
with col_ask:
    ask_btn = st.button("🔍 Ask", use_container_width=True)
with col_clear:
    if st.button("🧹 Clear history", use_container_width=False):
        st.session_state.chat_history = []
        st.session_state.last_result  = None
        st.rerun()

# ── Run query ──
if ask_btn and query.strip():
    em        = get_embedding_manager()
    retriever = RAGRetriever(vs, em)
    llm       = get_llm(api_key)

    with st.spinner("🔎 Searching papers and generating answer…"):
        try:
            result = rag_answer(
                query=query.strip(),
                retriever=retriever,
                llm=llm,
                top_k=top_k,
                score_threshold=threshold,
            )
            st.session_state.last_result = result
            st.session_state.chat_history.append({
                "question": query.strip(),
                "result":   result,
            })
        except Exception as e:
            st.error(f"❌ Error: {e}")

# ── Render last result prominently ──
if st.session_state.last_result:
    r = st.session_state.last_result

    # Confidence bar
    conf_pct = int(r["confidence"] * 100)
    conf_color = "#22c55e" if conf_pct >= 70 else "#f59e0b" if conf_pct >= 45 else "#ef4444"
    st.markdown(f"""
    <div class="conf-row">
      <span class="conf-label">Retrieval confidence</span>
      <div class="conf-bar-bg"><div class="conf-fill" style="width:{conf_pct}%;background:linear-gradient(90deg,#1d4ed8,{conf_color});"></div></div>
      <span class="conf-value">{conf_pct}%</span>
    </div>
    """, unsafe_allow_html=True)

    # Chips
    st.markdown(f"""
    <div class="chip-row">
      <span class="chip">📄 {r['retrieved_count']} chunks retrieved</span>
      <span class="chip">📚 {len(r['sources'])} sources cited</span>
    </div>
    """, unsafe_allow_html=True)

    # Answer
    st.markdown(f'<div class="answer-card">{r["answer"]}</div>', unsafe_allow_html=True)

    # Sources
    if r["sources"]:
        with st.expander("📚 Source details", expanded=True):
            for s in r["sources"]:
                sim_pct = int(s["similarity"] * 100)
                st.markdown(f"""
                <div class="source-card">
                  <span class="source-rank">#{s['rank']}</span>
                  <span class="source-file"> {s['source_file']}</span><br>
                  <span class="source-meta">
                    {s['authors']} · {s['year']}
                    &nbsp;|&nbsp; similarity: <b style="color:#60a5fa">{sim_pct}%</b>
                  </span><br>
                  <span class="source-meta" style="color:#475569">{s['title']}</span>
                </div>
                """, unsafe_allow_html=True)

# ── Chat history ──
if len(st.session_state.chat_history) > 1:
    st.markdown("---")
    st.markdown("### 🕘 Previous Questions")
    for entry in reversed(st.session_state.chat_history[:-1]):
        with st.expander(f"❓ {entry['question'][:90]}…"):
            r2 = entry["result"]
            st.markdown(f'<div class="answer-card">{r2["answer"]}</div>', unsafe_allow_html=True)
            if r2["sources"]:
                for s in r2["sources"]:
                    st.markdown(
                        f"**#{s['rank']}** `{s['source_file']}` — {s['authors']}, {s['year']} "
                        f"(similarity: {int(s['similarity']*100)}%)",
                    )
