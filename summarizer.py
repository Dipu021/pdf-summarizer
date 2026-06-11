from typing import List, Any
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama


MAP_PROMPT = PromptTemplate.from_template("""
You are a precise summarizer. Extract only the key points from this section.
Be concise. Use bullet points.

SECTION:
{text}

KEY POINTS:""")

COMBINE_PROMPT = PromptTemplate.from_template("""
Combine these key points into a single clear summary.
- Focus on the most important ideas only
- Remove duplicates  
- Write in flowing paragraphs
- Keep it under 300 words

KEY POINTS:
{text}

FINAL SUMMARY:""")


class Summarizer:
    def __init__(self, model_name: str = "llama3", base_url: str = "http://localhost:11434"):
        self.llm = ChatOllama(model=model_name, base_url=base_url, temperature=0.3)
        self.map_chain = MAP_PROMPT | self.llm | StrOutputParser()
        self.combine_chain = COMBINE_PROMPT | self.llm | StrOutputParser()

    def summarize(self, chunks: List[Any]) -> str:
        print(f"[INFO] Mapping {len(chunks)} chunks...")
        # Map: summarize each chunk independently
        mapped = [
            self.map_chain.invoke({"text": chunk.page_content})
            for chunk in chunks
        ]

        # Combine: merge all chunk summaries into one
        combined_text = "\n\n".join(mapped)
        print("[INFO] Combining chunk summaries...")
        return self.combine_chain.invoke({"text": combined_text}).strip()