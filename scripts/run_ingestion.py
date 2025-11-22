#!/usr/bin/env python3
"""Script to run data ingestion pipeline"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.database.connection import init_database
from backend.data_pipeline.ingestion import get_ingestion_pipeline
from backend.utils.logger import setup_logger
from backend.utils.cost_monitor import get_cost_monitor

logger = setup_logger("ingestion_script")


def main():
    parser = argparse.ArgumentParser(description="Run ETF data ingestion pipeline")
    parser.add_argument(
        "--ticker",
        type=str,
        help="Ingest data for a specific ticker"
    )
    parser.add_argument(
        "--all-ai-stocks",
        action="store_true",
        help="Ingest data for all AI stocks"
    )
    parser.add_argument(
        "--market-news",
        action="store_true",
        help="Ingest general market news"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Limit for news articles per feed"
    )

    args = parser.parse_args()

    # Initialize database
    logger.info("Initializing database...")
    db_manager = init_database()

    # Check database connection
    if not db_manager.check_connection():
        logger.error("Database connection failed")
        sys.exit(1)

    # Initialize pipeline
    logger.info("Initializing ingestion pipeline...")
    pipeline = get_ingestion_pipeline()

    # Check budget
    cost_monitor = get_cost_monitor()
    if not cost_monitor.check_budget():
        logger.error("Daily budget exceeded. Cannot proceed with ingestion.")
        sys.exit(1)

    # Run ingestion based on arguments
    if args.ticker:
        logger.info(f"Ingesting data for ticker: {args.ticker}")
        stats = pipeline.ingest_stock(args.ticker.upper())
        logger.info(f"Ingestion complete: {stats}")

    elif args.all_ai_stocks:
        logger.info("Ingesting data for all AI stocks...")
        all_stats = pipeline.ingest_all_ai_stocks()
        logger.info(f"Completed ingestion for {len(all_stats)} stocks")

        # Print summary
        total_docs = sum(s["documents_created"] for s in all_stats)
        total_embeddings = sum(s["embeddings_created"] for s in all_stats)
        total_errors = sum(s["errors"] for s in all_stats)

        print(f"\nIngestion Summary:")
        print(f"  Total documents: {total_docs}")
        print(f"  Total embeddings: {total_embeddings}")
        print(f"  Total errors: {total_errors}")

    elif args.market_news:
        logger.info("Ingesting market news...")
        stats = pipeline.ingest_market_news(limit_per_feed=args.limit)
        logger.info(f"Market news ingestion complete: {stats}")

    else:
        parser.print_help()
        sys.exit(1)

    # Print cost summary
    costs = cost_monitor.get_daily_costs()
    print(f"\nToday's Costs:")
    print(f"  Embedding: ${costs['embedding']:.4f}")
    print(f"  Agent: ${costs['agent']:.4f}")
    print(f"  Total: ${costs['total']:.4f}")

    logger.info("Ingestion script completed successfully")


if __name__ == "__main__":
    main()
