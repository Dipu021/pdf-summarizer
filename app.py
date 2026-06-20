import streamlit as st
from pdf_processor import PDFProcessor
from summarizer import Summarizer

# --- Page Config ---
st.set_page_config(
    page_title="PDF Summarizer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📄 PDF Summarizer")
st.caption("Fast, local, and private PDF summarization powered by Ollama + Llama")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    model_name = st.selectbox(
        "Ollama Model",
        ["llama3.2:3b", "gemma2:2b", "phi3:3.8b", "llama3.1:8b", "llama3"],
        index=0,
        help="Smaller models are much faster. Use 3B/2B models for speed."
    )
    
    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.slider("Chunk Size", 2000, 6000, 4000, step=500)
    with col2:
        chunk_overlap = st.slider("Overlap", 0, 400, 150, step=50)
    
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, step=0.1)
    max_workers = st.slider("Parallel Workers", 2, 8, 6, help="Higher = faster (if GPU allows)")
    
    st.markdown("---")
    st.markdown("**Requirements**")
    st.markdown("• Ollama running on port 11434\n• Model pulled (`ollama pull llama3.2:3b`)")

# --- Main Area ---
uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"], help="Max recommended size: ~50MB")

if uploaded_file is not None:
    st.info(f"**File:** {uploaded_file.name} | Size: {round(uploaded_file.size / (1024*1024), 2)} MB")
    
    if st.button("✨ Generate Summary", type="primary", use_container_width=True):
        try:
            with st.spinner("📖 Reading and chunking PDF..."):
                processor = PDFProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                documents = processor.load_from_bytes(uploaded_file.read(), uploaded_file.name)
                chunks = processor.chunk_documents(documents)

            st.success(f"✅ Loaded {len(documents)} pages → {len(chunks)} chunks")

            if len(chunks) > 50:
                st.warning("⚠️ Large document detected. This may take longer.")

            with st.spinner(f"🤖 Summarizing with **{model_name}** (parallel mode)..."):
                summarizer = Summarizer(
                    model_name=model_name,
                    temperature=temperature
                )
                
                summary = summarizer.summarize(chunks, max_workers=max_workers)

            st.markdown("### 📝 Final Summary")
            st.markdown(summary)

            col1, col2 = st.columns([1, 1])
            with col1:
                st.download_button(
                    label="⬇️ Download Summary",
                    data=summary,
                    file_name=f"{uploaded_file.name.replace('.pdf', '')}_summary.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with col2:
                if st.button("🔄 Summarize Again", use_container_width=True):
                    st.rerun()

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.info("Make sure Ollama is running: `ollama serve`")

else:
    st.info("👆 Upload a PDF to get started", icon="📤")