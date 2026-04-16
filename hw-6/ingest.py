"""
Knowledge ingestion pipeline.

Loads documents from data/ directory, splits into chunks,
generates embeddings, and saves the index to disk.

Usage: python ingest.py
"""

import json
import logging
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings


def _load_documents(data_dir: str):
    docs = []
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        if fname.endswith(".pdf"):
            loader = PyPDFLoader(fpath)
        elif fname.endswith(".txt") or fname.endswith(".md"):
            loader = TextLoader(fpath, encoding="utf-8")
        else:
            continue
        docs.extend(loader.load())
    return docs


def ingest():
    data_dir = settings.data_dir
    index_dir = settings.index_dir

    print(f"Loading documents from {data_dir}/...")
    docs = _load_documents(data_dir)
    if not docs:
        print("No documents found. Add PDF or TXT files to the data/ directory.")
        return

    print(f"Loaded {len(docs)} pages. Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    print("Loading embedding model (first run will download ~470MB)...")
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)

    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(index_dir, exist_ok=True)
    vectorstore.save_local(index_dir)
    print(f"FAISS index saved to {index_dir}/")

    chunks_path = os.path.join(index_dir, "bm25_chunks.json")
    chunks_data = [
        {"page_content": c.page_content, "metadata": c.metadata} for c in chunks
    ]
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False)
    print(f"BM25 chunks saved to {chunks_path}")

    print(f"\nIndexed {len(chunks)} chunks from {len(docs)} pages. Ready for search.")


if __name__ == "__main__":
    ingest()
