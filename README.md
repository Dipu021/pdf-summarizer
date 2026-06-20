# PDF Summarizer

A local PDF summarizer using **Llama 3** (via Ollama) + **LangChain** + **Streamlit**.

## Project Structure

```
pdf_summarizer/
├── app.py              # Streamlit UI
├── pdf_processor.py    # PDF loading + chunking
├── summarizer.py       # LangChain map_reduce summarization
├── requirements.txt
└── README.md
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Install and start Ollama
```bash
# Install Ollama from https://ollama.com
ollama pull llama3.2:3b
ollama pull gemma2:2b
ollama serve
```

### 3. Run the app
```bash
streamlit run app.py
```

---

## How It Works

1. **Upload** a PDF via the UI
2. **PyMuPDF** extracts text page by page
3. **RecursiveCharacterTextSplitter** splits text into overlapping chunks
4. **map_reduce chain** — each chunk is summarized independently (map), then all summaries are combined into one final summary (reduce)
5. **Download** the summary as a `.txt` file

## Tuning Tips

| Setting | Recommendation |
|---|---|
| Chunk Size | 3000 chars for most PDFs |
| Chunk Overlap | 200 chars to avoid losing context at boundaries |
| Temperature | 0.3 for focused, factual summaries |
