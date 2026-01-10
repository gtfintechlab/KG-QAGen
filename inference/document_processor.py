"""
Document processing utilities for loading and cleaning HTML documents.
Handles HTML cleaning, text chunking, and document loading from files.
"""

import logging
import os

import config

from bs4 import BeautifulSoup


def clean_html(raw_html: str) -> str:
    """
    Minimal HTML-to-text cleaning with BeautifulSoup.
    Removes <script> and <style>, then collapses whitespace.

    Args:
        raw_html: Raw HTML content as string

    Returns:
        Cleaned text with whitespace normalized
    """
    soup = BeautifulSoup(raw_html, "html.parser")
    for script in soup(["script", "style"]):
        script.decompose()
    text = soup.get_text(separator=" ")
    text = " ".join(text.split())
    return text


def chunk_text(doc_id: str, text: str, chunk_size: int) -> list[str]:
    """
    Splits text into a list of substrings, each at most chunk_size characters.
    This is not for RAG, this is chunking in case of a large document.

    Args:
        doc_id: Document identifier for logging
        text: Text content to chunk
        chunk_size: Maximum size of each chunk in characters

    Returns:
        List of text chunks
    """
    if len(text) > chunk_size:
        logging.warning(
            f"Doc {doc_id} length {len(text)} > {chunk_size}, chunking..."
        )
        chunks = [text[i: i + chunk_size]
                  for i in range(0, len(text), chunk_size)]
        logging.info(
            f"Doc {doc_id} chunked into {len(chunks)} parts. Each up to {chunk_size} chars."
        )
        return chunks
    else:
        logging.info(
            f"Doc {doc_id} loaded, length={len(text)} chars."
        )
        return [text]


def load_document_text(doc_id: str, html_path: str, max_char_for_system: int, testing_regime: str = 'FULL') -> list[str]:
    """
    Load raw HTML and clean it, or load correct pieces for Oracle baseline.
    If longer than max_char_for_system, chunk it; else return as single chunk.

    Args:
        doc_id: Document identifier
        html_path: Base path to HTML documents directory
        max_char_for_system: Maximum characters before chunking is required
        testing_regime: 'FULL', 'RAG', or 'GOLD'

    Returns:
        List of text chunks (one or more) or empty list on error
    """
    file_path = os.path.join(
        html_path, f'{doc_id}{".html" if testing_regime == "FULL" else "_gold.txt"}')

    if not os.path.exists(file_path):
        logging.error(f"HTML file not found: {file_path}")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            html_data = file.read()

        if testing_regime == 'GOLD':
            return html_data

        # FULL regime
        cleaned = clean_html(html_data)
        if not cleaned:
            logging.warning(f"Doc {doc_id} is empty after cleaning.")
            return []
        return chunk_text(doc_id, cleaned, max_char_for_system)

    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return []


def load_vector_db_text(vector_store, doc_id: str, query: str, top_k: int):
    """
    Backward-compatible wrapper for existing static RAG code paths.
    Retrieves top_k chunks for query and returns a merged string.
    """
    docs = retrieve_vector_db_docs(vector_store, doc_id, query, top_k)
    return merge_and_dedupe_docs(docs, max_chars=getattr(config, "DYN_RAG_MAX_EVIDENCE_CHARS", 120000))


def retrieve_vector_db_docs(vector_store, doc_id: str, query: str, top_k: int):
    retriever = vector_store.as_retriever(
        search_kwargs={"k": top_k, "filter": {"docID": str(doc_id)}}
    )
    return retriever.get_relevant_documents(query)


def merge_and_dedupe_docs(docs, max_chars: int) -> str:
    seen = set()
    merged = []
    total = 0

    for d in docs:
        key = d.metadata.get("chunk_id", hash(d.page_content))
        if key in seen:
            continue
        seen.add(key)

        text = d.page_content
        # enforce max char budget
        remaining = max_chars - total
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[:remaining]

        merged.append(text)
        total += len(text)

    return "\n\n".join(merged)
