"""
CLI runner: ingest, build vector DBs, instantiate orchestrator, run test queries and optionally evaluate and print results.
"""

import argparse
import json
import os
from dotenv import load_dotenv
from rich import print

load_dotenv()

from ingest import ingest_all
from orchestrator import Orchestrator
from evaluator import Evaluator

from langfuse import get_client
 
langfuse = get_client()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ingest", action="store_true", help="Run ingestion to build vector stores"
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Alias for ingest (kept for CLI compatibility)",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run test queries through the orchestrator",
    )
    parser.add_argument(
        "--test-file", default="test_queries.json", help="Path to test queries json"
    )
    parser.add_argument(
        "--langfuse", action="store_true", help="Enable Langfuse traces if configured"
    )
    args = parser.parse_args()

    if args.ingest or args.build:
        print("Starting ingestion...")
        counts = ingest_all(min_chunks_per_domain=50)
        print("Ingestion finished:", counts)

    if args.run_tests:
        # load tests
        with open(args.test_file, "r") as f:
            tests = json.load(f)
        with langfuse.start_as_current_observation(as_type="span", name="intelligent-ticket-router") as span:
            orchestrator = Orchestrator(enable_langfuse=args.langfuse)
            evaluator = Evaluator()
            results = []
            print(tests )
            print("************", type(tests))
            for t in tests["queries"]:
                q = t["query"]
                expected = t.get("expected")
                print("\n---")
                print("Query:", q)
                resp = orchestrator.route(q)
                answer = resp.get("answer") or resp.get("raw_result", {}).get("result", "")
                print("Answer:", answer)
                print("Classification:", resp.get("classification"))
                # Evaluate
                eval_res = evaluator.evaluate(q, answer, resp.get("sources", []))
                print("\nEvaluation:")
                print(eval_res)
                print("Eval score:", eval_res.get("score"))
                print("Eval rationale:", eval_res.get("rationale"))
                results.append(
                    {
                        "query": q,
                        "expected": expected,
                        "classification": resp.get("classification"),
                        "answer": answer,
                        "eval": eval_res,
                    }
                )
            # Save a results file
            out_path = "run_results.json"
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved results to {out_path}")
        langfuse.flush()


if __name__ == "__main__":
    main()
