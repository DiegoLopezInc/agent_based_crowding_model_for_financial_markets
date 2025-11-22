#!/usr/bin/env python3
"""Script to query the ETF research agent"""

import argparse
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.agent.agent import get_agent
from backend.agent.evaluation import get_evaluator
from backend.utils.logger import setup_logger
from backend.utils.cost_monitor import get_cost_monitor

logger = setup_logger("query_script")


def main():
    parser = argparse.ArgumentParser(description="Query the ETF research agent")
    parser.add_argument(
        "question",
        type=str,
        help="Question to ask the agent"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["agent", "rag"],
        default="rag",
        help="Query mode: 'agent' for full agentic RAG, 'rag' for simple RAG (faster, cheaper)"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the response quality"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )

    args = parser.parse_args()

    # Check budget
    cost_monitor = get_cost_monitor()
    if not cost_monitor.check_budget("agent"):
        logger.error("Agent budget exceeded")
        print("ERROR: Daily agent budget exceeded. Please try again tomorrow.")
        sys.exit(1)

    # Initialize agent
    logger.info("Initializing agent...")
    agent = get_agent()

    # Query
    logger.info(f"Processing query: {args.question}")

    if args.mode == "agent":
        response = agent.query(args.question, return_metadata=True)
    else:
        response = agent.simple_rag_query(args.question)

    # Evaluate if requested
    if args.evaluate and "answer" in response:
        evaluator = get_evaluator()
        documents = response.get("documents", [])

        scores = evaluator.calculate_reward(
            question=args.question,
            answer=response["answer"],
            retrieved_documents=documents,
        )

        response["evaluation"] = scores

    # Output
    if args.json:
        print(json.dumps(response, indent=2, default=str))
    else:
        print(f"\nQuestion: {args.question}\n")
        print(f"Answer:\n{response.get('answer', 'No answer generated')}\n")

        if "metadata" in response:
            metadata = response["metadata"]
            print(f"Metadata:")
            print(f"  Duration: {metadata.get('duration_ms', 0):.0f}ms")
            print(f"  Cost: ${metadata.get('cost_usd', 0):.4f}")

            if "results_count" in metadata:
                print(f"  Documents retrieved: {metadata['results_count']}")

        if "evaluation" in response:
            print(f"\nEvaluation Scores:")
            for key, value in response["evaluation"].items():
                print(f"  {key}: {value:.3f}")

        # Show cost summary
        costs = cost_monitor.get_daily_costs()
        print(f"\nToday's Total Costs:")
        print(f"  Embedding: ${costs['embedding']:.4f}")
        print(f"  Agent: ${costs['agent']:.4f}")
        print(f"  Total: ${costs['total']:.4f}")


if __name__ == "__main__":
    main()
