#!/usr/bin/env python3
"""Script to fine-tune embeddings with contrastive learning"""

import argparse
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.database.connection import init_database
from backend.embeddings.fine_tuning.data_preparation import get_data_preparer
from backend.embeddings.fine_tuning.trainer import get_trainer
from backend.embeddings.fine_tuning.evaluation import compare_embeddings
from backend.utils.logger import setup_logger
from backend.utils.config import get_config
from sentence_transformers import SentenceTransformer

logger = setup_logger("fine_tune_script")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune embeddings with contrastive learning"
    )
    parser.add_argument(
        "--prepare-data-only",
        action="store_true",
        help="Only prepare data without training"
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only train (assumes data is already prepared)"
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Only evaluate existing models"
    )
    parser.add_argument(
        "--save-data",
        type=str,
        help="Save prepared data to file for inspection"
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.2,
        help="Ratio of data to use for evaluation (default: 0.2)"
    )

    args = parser.parse_args()

    config = get_config()

    # Check if database is accessible
    logger.info("Initializing database...")
    db_manager = init_database()

    if not db_manager.check_connection():
        logger.error("Database connection failed")
        sys.exit(1)

    # Check if we have data
    stats = db_manager.get_stats()
    logger.info(f"Database stats: {stats}")

    if stats["documents"] == 0:
        logger.error(
            "No documents in database. Please run ingestion first:\n"
            "  python scripts/run_ingestion.py --all-ai-stocks\n"
            "  python scripts/run_ingestion.py --ticker JNJ  # and other opposite ETF stocks"
        )
        sys.exit(1)

    # Step 1: Prepare data
    if not args.train_only and not args.evaluate_only:
        logger.info("=" * 60)
        logger.info("STEP 1: Preparing training data")
        logger.info("=" * 60)

        preparer = get_data_preparer()

        # Check if we have opposite ETF data
        opposite_stocks = preparer.get_opposite_stocks()
        logger.info(f"Opposite ETF stocks: {opposite_stocks}")

        # Prepare training pairs
        all_pairs = preparer.prepare_training_data(include_hard_negatives=True)

        if len(all_pairs) == 0:
            logger.error(
                "No training pairs created. Make sure you have ingested data for both:\n"
                "  1. AI stocks (QQQ): python scripts/run_ingestion.py --all-ai-stocks\n"
                "  2. Opposite ETF stocks (VYM): python scripts/run_ingestion.py --ticker JNJ"
            )
            sys.exit(1)

        # Split into train and eval
        train_pairs, eval_pairs = preparer.split_train_eval(
            all_pairs,
            eval_ratio=args.eval_ratio
        )

        logger.info(f"Training pairs: {len(train_pairs)}")
        logger.info(f"Evaluation pairs: {len(eval_pairs)}")

        # Save if requested
        if args.save_data:
            logger.info(f"Saving data to {args.save_data}")
            preparer.save_pairs_to_file(all_pairs, args.save_data)

        if args.prepare_data_only:
            logger.info("Data preparation complete (--prepare-data-only)")
            sys.exit(0)

    # Step 2: Train
    if not args.evaluate_only:
        logger.info("=" * 60)
        logger.info("STEP 2: Fine-tuning embeddings")
        logger.info("=" * 60)

        trainer = get_trainer()

        if args.train_only:
            logger.error("--train-only requires pre-prepared data (not yet implemented)")
            sys.exit(1)

        # Train model
        fine_tuned_model = trainer.train(
            train_pairs=train_pairs,
            eval_pairs=eval_pairs
        )

        logger.info("Fine-tuning complete")

    # Step 3: Evaluate
    logger.info("=" * 60)
    logger.info("STEP 3: Evaluating models")
    logger.info("=" * 60)

    # Load models
    base_model_name = config.embedding.model_name
    fine_tuned_model_path = config.embedding.fine_tuning.fine_tuned_model_path

    if not Path(fine_tuned_model_path).exists():
        logger.error(f"Fine-tuned model not found at {fine_tuned_model_path}")
        logger.error("Please run training first")
        sys.exit(1)

    logger.info(f"Loading base model: {base_model_name}")
    base_model = SentenceTransformer(base_model_name)

    logger.info(f"Loading fine-tuned model: {fine_tuned_model_path}")
    fine_tuned_model = SentenceTransformer(fine_tuned_model_path)

    # Get test pairs
    if args.evaluate_only:
        # Re-prepare data
        preparer = get_data_preparer()
        all_pairs = preparer.prepare_training_data(include_hard_negatives=True)
        _, eval_pairs = preparer.split_train_eval(all_pairs, eval_ratio=args.eval_ratio)

    # Compare models
    comparison = compare_embeddings(base_model, fine_tuned_model, eval_pairs)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print("\nBase Model:")
    print(f"  Threshold: {comparison['base_model']['threshold']:.2f}")
    print(f"  F1 Score: {comparison['base_model']['results']['f1_score']:.4f}")
    print(f"  Accuracy: {comparison['base_model']['results']['accuracy']:.4f}")
    print(f"  AUC-ROC: {comparison['base_model']['results']['auc_roc']:.4f}")
    print(f"  Separation: {comparison['base_model']['results']['separation']:.4f}")

    print("\nFine-Tuned Model:")
    print(f"  Threshold: {comparison['fine_tuned_model']['threshold']:.2f}")
    print(f"  F1 Score: {comparison['fine_tuned_model']['results']['f1_score']:.4f}")
    print(f"  Accuracy: {comparison['fine_tuned_model']['results']['accuracy']:.4f}")
    print(f"  AUC-ROC: {comparison['fine_tuned_model']['results']['auc_roc']:.4f}")
    print(f"  Separation: {comparison['fine_tuned_model']['results']['separation']:.4f}")

    print("\nImprovements:")
    for metric, improvement in comparison['improvements'].items():
        print(f"  {metric}: {improvement['absolute']:+.4f} ({improvement['percentage']:+.2f}%)")

    # Save results
    results_file = "logs/fine_tuning_results.json"
    Path(results_file).parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(comparison, f, indent=2)

    logger.info(f"Results saved to {results_file}")

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("\nTo use the fine-tuned model:")
    print("1. Edit config.yaml and set:")
    print("   embedding:")
    print("     fine_tuning:")
    print("       enabled: true")
    print("\n2. Restart your application")
    print("\n3. The fine-tuned embeddings will be used automatically")
    print("=" * 60)


if __name__ == "__main__":
    main()
