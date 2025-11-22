#!/usr/bin/env python3
"""Script to analyze ETF exposures and comparisons"""

import argparse
import sys
import json
from pathlib import Path
from datetime import date, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.database.connection import init_database
from backend.exposure.holdings_fetcher import get_holdings_fetcher
from backend.exposure.matrix_calculator import get_matrix_calculator
from backend.exposure.comparison_engine import get_comparison_engine
from backend.utils.logger import setup_logger

logger = setup_logger("exposure_script")


def main():
    parser = argparse.ArgumentParser(description="Analyze ETF exposures")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Compute command
    compute_parser = subparsers.add_parser("compute", help="Compute exposure matrices")
    compute_parser.add_argument("etf", help="ETF ticker")
    compute_parser.add_argument("--lookback", type=int, default=252, help="Lookback days")
    compute_parser.add_argument("--factors", type=int, default=5, help="Number of PCA factors")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two ETFs")
    compare_parser.add_argument("etf1", help="First ETF")
    compare_parser.add_argument("etf2", help="Second ETF")
    compare_parser.add_argument("--matrix-type", default="correlation", help="Matrix type")

    # Pairwise command
    pairwise_parser = subparsers.add_parser("pairwise", help="Get pairwise exposure")
    pairwise_parser.add_argument("etf", help="ETF")
    pairwise_parser.add_argument("ticker1", help="First ticker")
    pairwise_parser.add_argument("ticker2", help="Second ticker")

    # Similar command
    similar_parser = subparsers.add_parser("similar", help="Find similar stocks")
    similar_parser.add_argument("etf", help="ETF")
    similar_parser.add_argument("ticker", help="Target ticker")
    similar_parser.add_argument("--metric", default="correlation", help="Similarity metric")
    similar_parser.add_argument("--top", type=int, default=10, help="Number of results")

    # View command
    view_parser = subparsers.add_parser("view", help="View exposure matrix")
    view_parser.add_argument("etf", help="ETF")
    view_parser.add_argument("--matrix-type", default="correlation", help="Matrix type")
    view_parser.add_argument("--save", help="Save to file (JSON)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Initialize database
    logger.info("Initializing database...")
    db_manager = init_database()

    if not db_manager.check_connection():
        logger.error("Database connection failed")
        sys.exit(1)

    # Execute command
    if args.command == "compute":
        compute_exposure(args)
    elif args.command == "compare":
        compare_etfs_cmd(args)
    elif args.command == "pairwise":
        get_pairwise_cmd(args)
    elif args.command == "similar":
        find_similar_cmd(args)
    elif args.command == "view":
        view_matrix_cmd(args)


def compute_exposure(args):
    """Compute exposure matrices"""
    print(f"\n{'='*60}")
    print(f"Computing Exposure Matrices for {args.etf}")
    print(f"{'='*60}\n")

    fetcher = get_holdings_fetcher()
    calculator = get_matrix_calculator()

    # Get holdings
    print(f"Fetching holdings for {args.etf}...")
    tickers = fetcher.get_holdings_tickers(args.etf)

    if not tickers:
        print(f"❌ No holdings found for {args.etf}")
        sys.exit(1)

    print(f"✓ Found {len(tickers)} holdings: {', '.join(tickers[:10])}{', ...' if len(tickers) > 10 else ''}")

    # Fetch prices
    print(f"\nFetching price history ({args.lookback} days)...")
    processed = fetcher.fetch_and_save_etf_prices(args.etf, args.lookback)
    print(f"✓ Fetched prices for {processed}/{len(tickers)} stocks")

    # Compute matrices
    print(f"\nComputing exposure matrices...")
    end_date = date.today()
    start_date = end_date - timedelta(days=args.lookback)

    result = calculator.compute_and_save_matrices(
        args.etf, tickers, start_date, end_date, args.factors
    )

    if result['success']:
        print(f"\n✓ Computation successful!")
        print(f"\n  ETF: {result['etf']}")
        print(f"  Stocks: {result['n_stocks']}")
        print(f"  Date range: {result['date_range'][0]} to {result['date_range'][1]}")
        print(f"  Factors: {result['n_factors']}")
        print(f"  Explained variance: {sum(result['explained_variance']):.2%}")
        print(f"\n  Matrices computed:")
        print(f"    - Correlation matrix")
        print(f"    - Covariance matrix")
        print(f"    - Idiosyncratic matrix")
        print(f"\n  Pairwise exposures saved: {result['n_stocks'] * (result['n_stocks'] - 1) // 2} pairs")
    else:
        print(f"❌ Computation failed: {result.get('error')}")


def compare_etfs_cmd(args):
    """Compare two ETFs"""
    print(f"\n{'='*60}")
    print(f"Comparing {args.etf1} vs {args.etf2}")
    print(f"{'='*60}\n")

    engine = get_comparison_engine()

    result = engine.compare_etfs(args.etf1, args.etf2, args.matrix_type)

    if not result.get('success'):
        print(f"❌ Comparison failed: {result.get('error')}")
        sys.exit(1)

    print(f"Matrix type: {result['matrix_type']}")
    print(f"\nOverlap Analysis:")
    print(f"  Common stocks: {result['stats']['n_common']}")
    print(f"  {args.etf1} only: {result['stats']['n_etf1_only']}")
    print(f"  {args.etf2} only: {result['stats']['n_etf2_only']}")
    print(f"  Overlap ratio: {result['stats']['overlap_ratio']:.2%}")

    print(f"\nSimilarity Score: {result['similarity_score']:.4f}")

    if result['common_tickers']:
        print(f"\nCommon stocks ({len(result['common_tickers'])}): ")
        print(f"  {', '.join(result['common_tickers'][:15])}{', ...' if len(result['common_tickers']) > 15 else ''}")

    comp_matrix = result['comparison_matrix']
    print(f"\nMatrix Statistics:")
    print(f"  Pattern correlation: {comp_matrix.get('pattern_correlation', 0):.4f}")
    print(f"  Mean difference: {comp_matrix.get('mean_difference', 0):.4f}")
    print(f"  Max difference: {comp_matrix.get('max_difference', 0):.4f}")


def get_pairwise_cmd(args):
    """Get pairwise exposure"""
    print(f"\n{'='*60}")
    print(f"Pairwise Exposure: {args.ticker1} vs {args.ticker2} (in {args.etf})")
    print(f"{'='*60}\n")

    engine = get_comparison_engine()

    result = engine.get_pairwise_comparison(args.etf, args.ticker1, args.ticker2)

    if not result:
        print(f"❌ No data found for {args.ticker1}-{args.ticker2} in {args.etf}")
        print(f"\nTry running: python scripts/analyze_exposures.py compute {args.etf}")
        sys.exit(1)

    print(f"Correlation: {result['correlation']:+.4f}")
    print(f"Covariance: {result['covariance']:+.6f}")
    print(f"\nIdiosyncratic Score: {result['idiosyncratic_score']:.4f}")
    print(f"  (Higher = more different in stock-specific behavior)")
    print(f"\nCommon Factor Exposure: {result['common_factor_exposure']:.4f}")
    print(f"  (Shared market factor exposure)")
    print(f"\nSpecific Exposure: {result['specific_exposure']:.4f}")
    print(f"  (Stock-specific correlation)")
    print(f"\nComputation date: {result['computation_date']}")


def find_similar_cmd(args):
    """Find similar stocks"""
    print(f"\n{'='*60}")
    print(f"Stocks Similar to {args.ticker} (in {args.etf})")
    print(f"{'='*60}\n")

    engine = get_comparison_engine()

    similar = engine.get_stock_similarity_ranking(args.etf, args.ticker, args.metric, args.top)

    if not similar:
        print(f"❌ No data found for {args.ticker} in {args.etf}")
        sys.exit(1)

    print(f"Ranked by: {args.metric}\n")
    print(f"{'Rank':<6}{'Ticker':<10}{'Score':>10}")
    print(f"{'-'*26}")

    for i, item in enumerate(similar, 1):
        print(f"{i:<6}{item['ticker']:<10}{item['score']:>10.4f}")


def view_matrix_cmd(args):
    """View exposure matrix"""
    calculator = get_matrix_calculator()

    matrix_data = calculator.get_cached_matrix(args.etf, args.matrix_type)

    if not matrix_data:
        print(f"❌ Matrix not found for {args.etf}")
        print(f"\nTry running: python scripts/analyze_exposures.py compute {args.etf}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"{args.matrix_type.title()} Matrix for {args.etf}")
    print(f"{'='*60}\n")

    print(f"Stocks: {len(matrix_data['tickers'])}")
    print(f"Date range: {matrix_data['start_date']} to {matrix_data['end_date']}")
    print(f"Created: {matrix_data['created_at']}")

    print(f"\nTickers:")
    tickers = matrix_data['tickers']
    for i in range(0, len(tickers), 10):
        print(f"  {', '.join(tickers[i:i+10])}")

    if args.save:
        output = {
            'etf': args.etf,
            'matrix_type': args.matrix_type,
            'tickers': matrix_data['tickers'],
            'matrix': matrix_data['matrix'].tolist(),
            'params': matrix_data['params'],
            'date_range': {
                'start': matrix_data['start_date'].isoformat(),
                'end': matrix_data['end_date'].isoformat()
            }
        }

        with open(args.save, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\n✓ Saved to {args.save}")
    else:
        print(f"\nUse --save to export matrix to JSON")


if __name__ == "__main__":
    main()
