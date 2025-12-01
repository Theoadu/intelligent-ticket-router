from langchain_openai import ChatOpenAI
from langchain_classic.chains import ConversationalRetrievalChain
from .base_agent import get_retriever_for_domain, OPENAI_COMPLETION_MODEL
import os

MODEL = os.getenv("OPENAI_COMPLETION_MODEL", "gpt-4o-mini")

class HRAgent:
    def __init__(self):
        self.retriever = get_retriever_for_domain("hr", k=6)
        self.llm = ChatOpenAI(model_name=MODEL, temperature=0.0)

        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": None}
        )

    def run(self, query: str, chat_history=None):
        chat_history = chat_history or []
        result = self.chain({"question": query, "chat_history": chat_history})
        return result