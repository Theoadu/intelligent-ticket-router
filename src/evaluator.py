"""
Evaluator: takes query + answer + sources -> returns a 1-10 score and short reasoning.
Optionally logs to Langfuse.

This is a light-weight evaluator chain using the LLM. For production, prefer a suite of rule-based + LLM checks + human-in-the-loop.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):
    score: int = Field(..., description="Numeric score from 1 to 10")
    rationale: str = Field(..., description="1-2 sentence rationale for the score")
    issues: list[str] = Field([], description="List of issues if score < 8")


OPENAI_COMPLETION_MODEL = os.getenv("OPENAI_COMPLETION_MODEL", "gpt-4o-mini")

system_prompt = """
You are an automated evaluator for user support responses. 
Given the user query, a proposed answer (which may contain citations), 
and the source documents used, assign a numeric score from 1 to 10 where:
- 10 = perfect (accurate, complete, relevant, cites correct sources)
- 1 = completely wrong or hallucinated
Also provide a 1-2 sentence rationale and a short list of issues if the score < 8.
"""
agent = create_agent(
    system_prompt=system_prompt,
    model=OPENAI_COMPLETION_MODEL,
    response_format=EvaluationResult,
)


class Evaluator:

    def evaluate(self, query: str, answer: str, sources) -> dict:
        src_text = "\n".join([str(s) for s in (sources or [])])
        result = agent.invoke(
            {"messages": f"Query: {query}\nAnswer: {answer}\nSources: {src_text}"}
        )

        return result["structured_response"].model_dump()
