"""
Streamlit Web UI for AI Web Scrapper using LangChain RAG, Ollama & Meta Llama 3.2.

Author: godmode-dev
License: MIT
"""

import streamlit as st
import pandas as pd
from rag.rag_chain import WebScraperRAGChain

st.set_page_config(
    page_title="AI Web Scrapper — LangChain RAG & Llama 3.2",
    page_icon="🌐",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main { background-color: #0e1117; color: #e6edf3; }
    .badge-tag {
        background: linear-gradient(135deg, #a855f7, #6366f1);
        padding: 4px 12px;
        border-radius: 8px;
        color: #fff;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .chunk-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_rag_chain():
    return WebScraperRAGChain()


def main():
    st.sidebar.title("🌐 AI Web Scrapper")
    st.sidebar.markdown("<span class='badge-tag'>Meta Llama 3.2 + LangChain</span>", unsafe_allow_html=True)
    st.sidebar.markdown("---")

    target_url = st.sidebar.text_input(
        "Enter Target Website URL",
        value="https://en.wikipedia.org/wiki/Artificial_intelligence",
        placeholder="https://example.com"
    )

    ollama_model = st.sidebar.selectbox("Select LLM Model", options=["llama3.2", "llama3.1", "mistral"], index=0)
    top_k_chunks = st.sidebar.slider("RAG Top-K Vector Chunks", min_value=1, max_value=6, value=3)

    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Resume Project Showcase**: Implements Web Scraping, Dense Document Ingestion, LangChain Chains, Vector Retrieval, and Ollama Llama 3.2.")

    # App Title
    st.title("🌐 AI-Powered Web Scrapper & RAG Summarizer")
    st.markdown(
        "An intelligent information extraction agent built with **LangChain**, **Ollama**, and **Meta Llama 3.2**. "
        "Scrapes target web pages, cleans DOM noise, builds vector passages, and generates factual summaries."
    )

    rag = get_rag_chain()

    # Ingest Website Button
    if st.button("🚀 Scrape & Ingest Website", type="primary") or "scraped_info" in st.session_state:
        if "scraped_info" not in st.session_state or st.session_state.get("current_url") != target_url:
            with st.spinner(f"Scraping & Indexing content from `{target_url}`..."):
                scraped_data = rag.ingest_url(target_url)
                st.session_state["scraped_info"] = scraped_data
                st.session_state["current_url"] = target_url

        scraped_info = st.session_state["scraped_info"]

        # Display Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Page Title", scraped_info["title"][:25] + "...")
        with col2:
            st.metric("Generated Text Chunks", scraped_info["num_chunks"])
        with col3:
            st.metric("Ingestion Status", scraped_info["status"].upper())

        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(["💬 Ask Questions (RAG)", "📜 Auto Executive Summary", "📦 Inspect Vector Chunks"])

        # Tab 1: Q&A
        with tab1:
            st.subheader("Ask Questions About Webpage Content")
            user_query = st.text_input("Enter question", value="What are the key concepts described on this page?")

            if st.button("Generate RAG Answer"):
                with st.spinner("Retrieving vector passages & invoking Llama 3.2..."):
                    res = rag.query(user_query, top_k=top_k_chunks)
                    
                    st.markdown(f"### 🤖 Answer ({res['model']})")
                    st.write(res["answer"])

                    with st.expander("🔍 View Retrieved Context Passages"):
                        for idx, passage in enumerate(res["retrieved_context"], 1):
                            st.markdown(f"**Chunk #{idx}:**")
                            st.info(passage)

        # Tab 2: Summarizer
        with tab2:
            st.subheader("Auto Executive Website Summary")
            if st.button("Generate Web Summary"):
                with st.spinner("Generating summary with Meta Llama 3.2..."):
                    summary_res = rag.summarize_website()
                    st.success(summary_res["answer"])

        # Tab 3: Inspect Chunks
        with tab3:
            st.subheader("Extracted Text Passages & Chunks")
            for i, chunk in enumerate(scraped_info["chunks"], 1):
                st.markdown(f"<div class='chunk-box'><strong>Passage #{i}</strong><br>{chunk}</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
