import json
import os

import config

from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from main import clean_html


def build_vector_store():
    docs = []
    for filename in os.listdir(config.HTML_PATH):
    
        if filename.endswith(".html"):
            path = os.path.join(config.HTML_PATH, filename)
            with open(path, 'r', encoding='utf-8') as file:
                html = file.read()
            
            doc_id = str(filename)[:-5]
            text = clean_html(html)
            chunks = [text[i:i + config.RAG_CHUNK_SIZE]
                      for i in range(0, len(text), config.RAG_CHUNK_SIZE)]
            for i, chunk in enumerate(chunks):
                docs.append(Document(page_content=chunk, metadata={
                            "docID": str(doc_id), "chunk_id": i}))

    embeddings = HuggingFaceEmbeddings(model_name=config.RAG_MODEL)
    db = FAISS.from_documents(docs, embedding=embeddings)
    db.save_local(config.VECTOR_DB_DIR)


if __name__ == "__main__":
    build_vector_store()
