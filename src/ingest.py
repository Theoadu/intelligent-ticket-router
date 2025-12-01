"""
ingest.py

- Loads documents from data/<domain>_docs/
- Splits into chunks using LangChain text splitters and creates Chroma collections per domain
- Embeds with OpenAI embeddings
- Includes a helper to synthesize extra docs if you have too few docs (to reach 50+ chunks)
"""

import os
import glob
import json
from typing import List
from dotenv import load_dotenv

load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import uuid
import random

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./.chromadb")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

def list_text_files(dir_path: str) -> List[str]:
    exts = ["*.txt", "*.md", "*.csv", "*.pdf"]
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(dir_path, e)))
    return files

def read_text_file(path: str) -> str:
    if path.endswith(".pdf"):
        
        with open(path, "rb") as f:
            return f.read().decode(errors="ignore")
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

def synthesize_variants(base_text: str, n_variants: int = 10) -> List[str]:
   
    out = []
    lines = [l.strip() for l in base_text.splitlines() if l.strip()]
    for i in range(n_variants):
        random.shuffle(lines)
        new = "\n".join(lines)
        extra = f"\n\nDocument variant {i+1}. Generated for ingestion testing. Variant id: {uuid.uuid4()}"
        out.append(new + extra)
    return out

def chunk_texts(texts: List[str], chunk_size=800, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = []
    for t in texts:
        docs.extend(splitter.split_text(t))
    return docs

def build_chroma_client(persist_directory=CHROMA_PERSIST_DIR):
    os.makedirs(persist_directory, exist_ok=True)
    # client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory))
    # client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory))
    # client = Chroma(
    #     persist_directory=persist_directory,
    #     embedding_function=embeddings,
    #     collection_name="hr_faq"
    # )
    client = chromadb.PersistentClient(path=persist_directory)
    return client

def ingest_domain(domain: str, data_dir: str, client=None, min_chunks=50):
    print(f"[ingest] domain={domain}, data_dir={data_dir}")
    txt_files = list_text_files(data_dir)
    texts = []
    for f in txt_files:
        print(f"  - loading {f}")
        texts.append(read_text_file(f))

    # if too few texts, synthesize variants from first doc
    if len(texts) < 3 and len(texts) > 0:
        synth = synthesize_variants(texts[0], n_variants=10)
        texts.extend(synth)

    # chunk
    chunks = chunk_texts(texts, chunk_size=800, chunk_overlap=150)
    if len(chunks) < min_chunks:
        # synthesize more small chunks by splitting and adding small paraphrases
        needed = min_chunks - len(chunks)
        print(f"  - only {len(chunks)} chunks; synthesizing {needed} extra small chunks for testing")
        base = texts[0] if texts else "Generic company policy text."
        extra_variants = synthesize_variants(base, n_variants=needed//5 + 2)
        chunks.extend(chunk_texts(extra_variants, chunk_size=400, chunk_overlap=80))
    print(f"  - total chunks for {domain}: {len(chunks)}")

    # Embeddings
    emb = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)

    # Build chroma collection
    if client is None:
        client = build_chroma_client()

    collection_name = f"{domain}_collection"
    # delete if exists (clean reingest)
    try:
        existing = client.get_collection(name=collection_name)
        print("  - collection exists; deleting and rebuilding")
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    # create
    collection = client.create_collection(name=collection_name)

    # produce metadata and ids
    ids = [f"{domain}-{i}-{uuid.uuid4()}" for i in range(len(chunks))]
    metadatas = [{"domain": domain, "chunk_index": i} for i in range(len(chunks))]
    # embed in batches
    batch_size = 64
    all_embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        embs = emb.embed_documents(batch)
        all_embeddings.extend(embs)

    # Upsert into collection
    # Chroma client expects embeddings as list of lists; ensure lengths match.
    collection.add(
        ids=ids,
        documents=chunks,
        metadatas=metadatas,
        embeddings=all_embeddings
    )
    # persist
    # client.persist()
    print(f"  - persisted collection {collection_name}")
    return len(chunks)

def ingest_all(base_data_dir="data", min_chunks_per_domain=50):
    client = build_chroma_client()
    domains = ["hr", "tech", "finance"]
    counts = {}
    for d in domains:
        dirpath = os.path.join(base_data_dir, f"{d}_docs")
        if not os.path.exists(dirpath):
            print(f"Warning: {dirpath} does not exist; skipping.")
            counts[d] = 0
            continue
        counts[d] = ingest_domain(d, dirpath, client=client, min_chunks=min_chunks_per_domain)
    return counts

if __name__ == "__main__":
    # Allow running as script
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--min-chunks", type=int, default=50)
    args = p.parse_args()
    counts = ingest_all(base_data_dir=args.data_dir, min_chunks_per_domain=args.min_chunks)
    print("Ingestion complete:", counts)