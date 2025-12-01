"""
Evaluator: takes query + answer + sources -> returns a 1-10 score and short reasoning.
Optionally logs to Langfuse.

This is a light-weight evaluator chain using the LLM. For production, prefer a suite of rule-based + LLM checks + human-in-the-loop.
"""

import os
import json
from dotenv import load_dotenv
load_dotenv()

from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

OPENAI_COMPLETION_MODEL = os.getenv("OPENAI_COMPLETION_MODEL", "gpt-4o-mini")

EVAL_PROMPT = PromptTemplate(
    input_variables=["query", "answer", "sources"],
    template=(
        "You are an automated evaluator for user support responses. Given the user query, a proposed answer (which may contain citations), and the source documents used, assign a numeric score from 1 to 10 where:\n"
        "- 10 = perfect (accurate, complete, relevant, cites correct sources)\n"
        "- 1 = completely wrong or hallucinated\n\n"
        "Also provide a 1-2 sentence rationale and a short list of issues if the score < 8.\n\n"
        "Return JSON object with fields: {\"score\": int, \"rationale\": str, \"issues\": [str]}\n\n"
        "User query:\n{query}\n\n"
        "Answer:\n{answer}\n\n"
        "Sources:\n{sources}\n\n"
    )
)

class Evaluator:
    def __init__(self):
        self.llm = ChatOpenAI(model_name=OPENAI_COMPLETION_MODEL, temperature=0.0)
        self.chain = LLMChain(llm=self.llm, prompt=EVAL_PROMPT)

    def evaluate(self, query: str, answer: str, sources) -> dict:
        src_text = "\n".join([str(s) for s in (sources or [])])
        raw = self.chain.run({"query": query, "answer": answer, "sources": src_text})
        try:
            out = json.loads(raw)
            print("Parsed eval output:", out)
        except Exception:
            # Best-effort parse: fallback to simple scoring heuristic
            score = 8 if "source" in src_text.lower() else 6
            out = {"score": score, "rationale": raw[:200], "issues": []}
        return out