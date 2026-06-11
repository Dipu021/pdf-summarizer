import streamlit as st
from pdf_processor import PDFProcessor
from summarizer import Summarizer

# --- Page config ---
st.set_page_config(
    page_title="PDF Summarizer",
    page_icon="📄",
    layout="centered"
)

st.title("📄 PDF Summarizer")
st.caption("Upload a PDF and get a concise summary powered by Llama 3.")

# --- Sidebar settings ---
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox("Ollama Model", ["llama3", "llama3.1", "mistral"], index=0)
    chunk_size = st.slider("Chunk Size (chars)", 1000, 5000, 3000, step=500)
    chunk_overlap = st.slider("Chunk Overlap (chars)", 0, 500, 200, step=50)
    st.markdown("---")
    st.markdown("Make sure **Ollama** is running locally on port `11434`.")

# --- File upload ---
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    st.info(f"📎 **{uploaded_file.name}** — {round(uploaded_file.size / 1024, 1)} KB")

    if st.button("✨ Summarize", use_container_width=True):
        with st.spinner("Reading and chunking PDF..."):
            processor = PDFProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            documents = processor.load_from_bytes(uploaded_file.read(), uploaded_file.name)
            chunks = processor.chunk_documents(documents)

        st.success(f"✅ {len(documents)} pages → {len(chunks)} chunks")

        with st.spinner(f"Summarizing with {model_name}... this may take a moment."):
            try:
                summarizer = Summarizer(model_name=model_name)
                summary = summarizer.summarize(chunks)

                st.markdown("### 📝 Summary")
                st.markdown(summary)

                st.download_button(
                    label="⬇️ Download Summary",
                    data=summary,
                    file_name=f"{uploaded_file.name.replace('.pdf', '')}_summary.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.markdown("Make sure Ollama is running: `ollama serve` and the model is pulled: `ollama pull llama3`")
