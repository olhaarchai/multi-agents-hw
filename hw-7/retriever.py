"""
Hybrid retrieval module.

Combines semantic search (vector DB) + BM25 (lexical) + cross-encoder reranking.
"""

import json
import logging
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import settings


def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def get_retriever():
    index_dir = settings.index_dir
    chunks_path = os.path.join(index_dir, "bm25_chunks.json")

    if not os.path.exists(index_dir):
        raise FileNotFoundError(
            f"Index not found at '{index_dir}'. Run `python ingest.py` first."
        )

    embeddings = _get_embeddings()

    # 1. Load FAISS vector store
    vectorstore = FAISS.load_local(
        index_dir, embeddings, allow_dangerous_deserialization=True
    )

    # 2. Semantic retriever
    semantic_retriever = vectorstore.as_retriever(
        search_kwargs={"k": settings.retrieval_top_k}
    )

    # 3. BM25 retriever from saved chunks
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    chunks = [
        Document(page_content=c["page_content"], metadata=c["metadata"])
        for c in chunks_data
    ]
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = settings.retrieval_top_k

    # 4. Ensemble: semantic + BM25
    ensemble = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )

    # 5. Cross-encoder reranker
    cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=settings.rerank_top_n)

    # 6. Contextual compression with reranking on top of ensemble
    retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=ensemble,
    )

    return retriever
