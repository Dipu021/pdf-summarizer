from typing import List, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
import streamlit as st


# Improved Prompts
MAP_PROMPT = PromptTemplate.from_template("""
You are an expert summarizer. Extract the most important information from the text below.
Focus on key facts, arguments, findings, and conclusions. Be concise and clear.

TEXT:
{text}

KEY POINTS:""")

COMBINE_PROMPT = PromptTemplate.from_template("""
Create a coherent, well-structured final summary from the following key points.
- Remove redundancy
- Maintain logical flow
- Use clear, professional language
- Keep total length under 400 words

KEY POINTS:
{text}

FINAL SUMMARY:""")


class Summarizer:
    def __init__(self, model_name: str = "llama3", base_url: str = "http://localhost:11434", temperature: float = 0.3):
        self.llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
            num_ctx=8192  # Increase context for better combining
        )
        self.map_chain = MAP_PROMPT | self.llm | StrOutputParser()
        self.combine_chain = COMBINE_PROMPT | self.llm | StrOutputParser()

    def summarize_chunk(self, chunk: Any) -> str:
        """Summarize a single chunk."""
        try:
            return self.map_chain.invoke({"text": chunk.page_content})
        except Exception as e:
            print(f"[ERROR] Chunk summarization failed: {e}")
            return f"[Summary failed for this section]"

    def summarize(self, chunks: List[Any], max_workers: int = 6) -> str:
        if not chunks:
            return "No content to summarize."

        if len(chunks) == 1:
            return self.summarize_chunk(chunks[0])

        # --- Parallel Map Step ---
        st.info(f"📊 Summarizing {len(chunks)} sections in parallel...")
        progress_bar = st.progress(0)
        chunk_summaries = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {executor.submit(self.summarize_chunk, chunk): i 
                             for i, chunk in enumerate(chunks)}
            
            for idx, future in enumerate(as_completed(future_to_chunk)):
                result = future.result()
                chunk_summaries.append(result)
                progress = (idx + 1) / len(chunks)
                progress_bar.progress(progress)

        # --- Combine Step ---
        progress_bar.progress(1.0)
        st.info("🔄 Combining all section summaries into final output...")
        
        combined_text = "\n\n".join(chunk_summaries)
        final_summary = self.combine_chain.invoke({"text": combined_text})
        
        return final_summary.strip()