"""
Base agent utilities and common Retriever creation
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import ConversationalRetrievalChain

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./.chromadb")
OPENAI_COMPLETION_MODEL = os.getenv("OPENAI_COMPLETION_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

def get_retriever_for_domain(domain: str, k: int = 4):
    """
    Returns a LangChain retriever object from Chroma for the given domain.
    """
    persist_directory = CHROMA_PERSIST_DIR
    collection_name = f"{domain}_collection"

    embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
    vectordb = Chroma(collection_name=collection_name, persist_directory=persist_directory, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    return retriever