"""Data preparation for fine-tuning with contrastive learning"""

from typing import List, Dict, Any, Tuple
import random
from datetime import datetime
import numpy as np
from sqlalchemy import and_

from backend.database.connection import get_db_manager
from backend.database.models import Document, ETFHolding
from backend.utils.config import get_config
from backend.utils.logger import setup_logger

logger = setup_logger("fine_tuning_data")


class ContrastiveDataPreparer:
    """Prepare training data for contrastive learning"""

    def __init__(self):
        self.config = get_config()
        self.db_manager = get_db_manager()
        self.etf_config = self.config.etfs
        self.ft_config = self.config.embedding.fine_tuning.training

    def get_opposite_stocks(self) -> List[str]:
        """Get list of opposite ETF stocks"""
        return self.etf_config.opposite_etf_stocks

    def get_ai_stocks(self) -> List[str]:
        """Get list of AI stocks"""
        return self.etf_config.qqq_ai_stocks

    def create_positive_pairs(
        self,
        ticker: str,
        num_pairs: int = 5,
    ) -> List[Tuple[str, str, int]]:
        """Create positive pairs from the same stock

        Args:
            ticker: Stock ticker symbol
            num_pairs: Number of pairs to create

        Returns:
            List of (text1, text2, label) tuples where label=1 for positive
        """
        logger.info(f"Creating {num_pairs} positive pairs for {ticker}")

        with self.db_manager.get_session() as session:
            # Get all documents for this ticker
            docs = (
                session.query(Document)
                .filter(Document.ticker == ticker)
                .all()
            )

            if len(docs) < 2:
                logger.warning(f"Not enough documents for {ticker}, found {len(docs)}")
                return []

            pairs = []
            for _ in range(num_pairs):
                if len(docs) >= 2:
                    # Randomly sample two different documents
                    doc1, doc2 = random.sample(docs, 2)
                    pairs.append((doc1.content, doc2.content, 1))

            logger.info(f"Created {len(pairs)} positive pairs for {ticker}")
            return pairs

    def create_negative_pairs(
        self,
        ai_ticker: str,
        num_pairs: int = 5,
    ) -> List[Tuple[str, str, int]]:
        """Create negative pairs between AI stock and opposite ETF stocks

        Args:
            ai_ticker: AI stock ticker symbol
            num_pairs: Number of pairs to create

        Returns:
            List of (text1, text2, label) tuples where label=0 for negative
        """
        logger.info(f"Creating {num_pairs} negative pairs for {ai_ticker}")

        opposite_stocks = self.get_opposite_stocks()

        with self.db_manager.get_session() as session:
            # Get documents for AI ticker
            ai_docs = (
                session.query(Document)
                .filter(Document.ticker == ai_ticker)
                .all()
            )

            if not ai_docs:
                logger.warning(f"No documents found for AI ticker {ai_ticker}")
                return []

            # Get documents for opposite stocks
            opposite_docs = (
                session.query(Document)
                .filter(Document.ticker.in_(opposite_stocks))
                .all()
            )

            if not opposite_docs:
                logger.warning("No documents found for opposite stocks")
                return []

            pairs = []
            for _ in range(num_pairs):
                # Sample one AI doc and one opposite doc
                ai_doc = random.choice(ai_docs)
                opp_doc = random.choice(opposite_docs)
                pairs.append((ai_doc.content, opp_doc.content, 0))

            logger.info(f"Created {len(pairs)} negative pairs for {ai_ticker}")
            return pairs

    def create_hard_negative_pairs(
        self,
        ticker: str,
        num_pairs: int = 3,
    ) -> List[Tuple[str, str, int]]:
        """Create hard negative pairs (similar but different stocks)

        Hard negatives are from different AI stocks (same sector, different companies)

        Args:
            ticker: Source ticker
            num_pairs: Number of pairs to create

        Returns:
            List of (text1, text2, label) tuples where label=0 for negative
        """
        logger.info(f"Creating {num_pairs} hard negative pairs for {ticker}")

        ai_stocks = self.get_ai_stocks()
        other_ai_stocks = [t for t in ai_stocks if t != ticker]

        if not other_ai_stocks:
            logger.warning("No other AI stocks available for hard negatives")
            return []

        with self.db_manager.get_session() as session:
            # Get documents for source ticker
            source_docs = (
                session.query(Document)
                .filter(Document.ticker == ticker)
                .all()
            )

            if not source_docs:
                return []

            # Get documents for other AI stocks
            other_docs = (
                session.query(Document)
                .filter(Document.ticker.in_(other_ai_stocks))
                .all()
            )

            if not other_docs:
                return []

            pairs = []
            for _ in range(num_pairs):
                source_doc = random.choice(source_docs)
                other_doc = random.choice(other_docs)
                pairs.append((source_doc.content, other_doc.content, 0))

            logger.info(f"Created {len(pairs)} hard negative pairs for {ticker}")
            return pairs

    def prepare_training_data(
        self,
        include_hard_negatives: bool = True,
    ) -> List[Tuple[str, str, int]]:
        """Prepare complete training dataset

        Args:
            include_hard_negatives: Whether to include hard negative examples

        Returns:
            List of (text1, text2, label) tuples
        """
        logger.info("Preparing training data for contrastive learning")

        all_pairs = []
        ai_stocks = self.get_ai_stocks()

        for ticker in ai_stocks:
            # Positive pairs (same stock)
            pos_pairs = self.create_positive_pairs(
                ticker,
                num_pairs=self.ft_config.positive_samples_per_stock
            )
            all_pairs.extend(pos_pairs)

            # Negative pairs (AI vs opposite ETF)
            neg_pairs = self.create_negative_pairs(
                ticker,
                num_pairs=self.ft_config.negative_samples_per_stock
            )
            all_pairs.extend(neg_pairs)

            # Hard negative pairs (AI vs AI)
            if include_hard_negatives:
                hard_neg_pairs = self.create_hard_negative_pairs(
                    ticker,
                    num_pairs=int(
                        self.ft_config.negative_samples_per_stock *
                        self.ft_config.hard_negative_ratio
                    )
                )
                all_pairs.extend(hard_neg_pairs)

        # Shuffle pairs
        random.shuffle(all_pairs)

        logger.info(f"Prepared {len(all_pairs)} training pairs")

        # Log statistics
        positive_count = sum(1 for _, _, label in all_pairs if label == 1)
        negative_count = len(all_pairs) - positive_count

        logger.info(f"Positive pairs: {positive_count}")
        logger.info(f"Negative pairs: {negative_count}")
        logger.info(f"Ratio: {positive_count/len(all_pairs):.2%} positive")

        return all_pairs

    def split_train_eval(
        self,
        pairs: List[Tuple[str, str, int]],
        eval_ratio: float = 0.2,
    ) -> Tuple[List[Tuple[str, str, int]], List[Tuple[str, str, int]]]:
        """Split data into train and evaluation sets

        Args:
            pairs: All pairs
            eval_ratio: Ratio of data to use for evaluation

        Returns:
            (train_pairs, eval_pairs)
        """
        random.shuffle(pairs)
        split_idx = int(len(pairs) * (1 - eval_ratio))

        train_pairs = pairs[:split_idx]
        eval_pairs = pairs[split_idx:]

        logger.info(f"Split: {len(train_pairs)} train, {len(eval_pairs)} eval")

        return train_pairs, eval_pairs

    def save_pairs_to_file(
        self,
        pairs: List[Tuple[str, str, int]],
        filepath: str
    ):
        """Save pairs to a file for inspection

        Args:
            pairs: List of pairs
            filepath: Output file path
        """
        import json

        with open(filepath, 'w') as f:
            for text1, text2, label in pairs:
                f.write(json.dumps({
                    "text1": text1,
                    "text2": text2,
                    "label": label
                }) + '\n')

        logger.info(f"Saved {len(pairs)} pairs to {filepath}")


def get_data_preparer() -> ContrastiveDataPreparer:
    """Get data preparer instance"""
    return ContrastiveDataPreparer()
