"""
Orchestrator: classifies incoming query and routes to domain agent.
Also integrates Langfuse tracing if available.
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

from langchain_classic.chains import LLMChain
from langchain_classic import PromptTemplate
from langchain_openai import ChatOpenAI
# from langchain_core.callbacks.manager import CallbackManager
# from langchain_core.callbacks.base import CallbackManager as CM

from agents.hr_agent import HRAgent
from agents.tech_agent import TechAgent
from agents.finance_agent import FinanceAgent

# Optional Langfuse tracer integration
try:
    from langfuse import Langfuse
    from langfuse.tracing.langchain import LangfuseTracer

    LANGFUSE_AVAILABLE = True
except Exception:
    Langfuse = None
    LangfuseTracer = None
    LANGFUSE_AVAILABLE = False

OPENAI_COMPLETION_MODEL = os.getenv("OPENAI_COMPLETION_MODEL", "gpt-4o-mini")

CLASSIFIER_PROMPT = PromptTemplate(
    input_variables=["query"],
    template=(
        "You are a classification assistant. Given the user support query, classify it into one of these categories:\n"
        "- HR\n- Tech\n- Finance\n- Other\n\n"
        "Return a JSON object with fields:\n"
        "  - category: one of [HR, Tech, Finance, Other]\n"
        "  - confidence: a float between 0 and 1\n"
        "  - reasons: short justification (1-2 sentences)\n\n"
        "User query:\n"
        "{query}\n\n"
        "Only return the JSON. Do not include extra text.\n"
    ),
)


class Orchestrator:
    def __init__(self, enable_langfuse: bool = False):
        # Model for classification
        self.llm = ChatOpenAI(model_name=OPENAI_COMPLETION_MODEL, temperature=0.0)
        self.chain = LLMChain(llm=self.llm, prompt=CLASSIFIER_PROMPT)
        # instantiate agents
        self.hr_agent = HRAgent()
        self.tech_agent = TechAgent()
        self.finance_agent = FinanceAgent()
        self.enable_langfuse = enable_langfuse and LANGFUSE_AVAILABLE
        if self.enable_langfuse:
            # Initialize Langfuse client using env variables if not already initialized
            lf_public = os.getenv("LANGFUSE_PUBLIC_KEY")
            lf_secret = os.getenv("LANGFUSE_SECRET_KEY")
            lf_host = os.getenv("LANGFUSE_HOST")
            # safe init
            Langfuse(
                api_key=lf_secret, tracker_url=lf_host
            )  # depending on SDK; this is best-effort
            self.tracer = LangfuseTracer()
        else:
            self.tracer = None

    def classify(self, query: str):
        # run classification chain
        try:
            resp = self.chain.run({"query": query})
            # The response should be a JSON object. Try to parse.
            parsed = json.loads(resp)
            return parsed
        except Exception as e:
            # fallback simple keyword classification
            q = query.lower()
            cat = "Other"
            if any(
                w in q for w in ["leave", "vacation", "paternity", "hr", "employee"]
            ):
                cat = "HR"
            elif any(
                w in q
                for w in [
                    "laptop",
                    "password",
                    "vpn",
                    "deploy",
                    "server",
                    "build",
                    "log",
                ]
            ):
                cat = "Tech"
            elif any(
                w in q
                for w in ["expense", "receipt", "bank", "payroll", "budget", "invoice"]
            ):
                cat = "Finance"
            return {
                "category": cat,
                "confidence": 0.6,
                "reasons": f"Fallback keyword rule matched: {str(e)}",
            }

    def route(self, query: str, chat_history=None):
        classification = self.classify(query)
        cat = classification.get("category", "Other")
        confidence = classification.get("confidence", 0.0)
        reasons = classification.get("reasons", "")
        trace_info = {"query": query, "classification": classification}

        # Tracing start - if Langfuse tracer exists, register a run (best effort)
        if self.tracer:
            # Use tracer as a callback in LangChain; for brevity, we only note this place.
            pass

        # route to correct agent
        if cat == "HR":
            result = self.hr_agent.run(query, chat_history)
        elif cat == "Tech":
            result = self.tech_agent.run(query, chat_history)
        elif cat == "Finance":
            result = self.finance_agent.run(query, chat_history)
        else:
            # fallback: use combined small retrieval across all domains (naive)
            # choose tech retriever by default to return something
            result = {
                "answer": "Sorry, I couldn't determine the correct department. Please provide more context.",
                "source_documents": [],
            }

        # Build final response object
        response_obj = {
            "query": query,
            "classification": classification,
            "answer": result.get("answer") or result.get("result"),
            "raw_result": result,
            "sources": [
                {"source": getattr(d, "metadata", None) or getattr(d, "source", "")}
                for d in result.get("source_documents", [])
            ],
        }
        return response_obj
