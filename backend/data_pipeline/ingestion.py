"""Data ingestion pipeline - orchestrates fetching, chunking, embedding, and storing"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from tqdm import tqdm

from backend.data_pipeline.data_sources import get_data_pipeline
from backend.data_pipeline.chunker import get_chunker
from backend.embeddings.embedding_model import get_embedding_model
from backend.database.connection import get_db_manager
from backend.database.repository import DocumentRepository, EmbeddingRepository, ETFHoldingRepository
from backend.utils.logger import setup_logger
from backend.utils.cost_monitor import get_cost_monitor

logger = setup_logger("ingestion")


class IngestionPipeline:
    """Main ingestion pipeline for ETF research data"""

    def __init__(self):
        self.data_pipeline = get_data_pipeline()
        self.chunker = get_chunker()
        self.embedding_model = get_embedding_model()
        self.db_manager = get_db_manager()
        self.cost_monitor = get_cost_monitor()

    def ingest_stock(self, ticker: str) -> Dict[str, int]:
        """Ingest data for a single stock

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with ingestion statistics
        """
        logger.info(f"Starting ingestion for {ticker}")
        stats = {
            "ticker": ticker,
            "documents_created": 0,
            "chunks_created": 0,
            "embeddings_created": 0,
            "errors": 0,
        }

        try:
            # Check budget before proceeding
            if not self.cost_monitor.check_budget("embedding"):
                logger.warning("Embedding budget exceeded, skipping ingestion")
                return stats

            # Fetch stock data
            stock_data = self.data_pipeline.fetch_stock_data(ticker)

            # Convert to documents
            documents = self.data_pipeline.create_documents_from_stock_data(stock_data)
            logger.info(f"Created {len(documents)} documents for {ticker}")

            # Chunk documents
            chunked_docs = self.chunker.chunk_documents(documents)
            logger.info(f"Created {len(chunked_docs)} chunks for {ticker}")

            # Batch process chunks
            batch_size = 32
            with self.db_manager.get_session() as session:
                for i in range(0, len(chunked_docs), batch_size):
                    batch = chunked_docs[i:i + batch_size]

                    # Extract texts for embedding
                    texts = [doc["content"] for doc in batch]

                    # Generate embeddings
                    embeddings = self.embedding_model.embed_documents(
                        texts,
                        show_progress=False
                    )

                    # Store documents and embeddings
                    for doc, embedding in zip(batch, embeddings):
                        try:
                            # Create document
                            db_doc = DocumentRepository.create_document(
                                session=session,
                                content=doc["content"],
                                metadata=doc.get("metadata", {}),
                                source=doc.get("source", "unknown"),
                                ticker=doc.get("ticker"),
                                etf=doc.get("etf"),
                                document_type=doc.get("document_type"),
                                published_date=doc.get("published_date"),
                            )
                            stats["documents_created"] += 1

                            # Create embedding
                            EmbeddingRepository.create_embedding(
                                session=session,
                                document_id=db_doc.id,
                                embedding=embedding,
                                model=self.embedding_model.config.model_name,
                            )
                            stats["embeddings_created"] += 1

                        except Exception as e:
                            logger.error(f"Failed to store document: {e}")
                            stats["errors"] += 1
                            continue

                    session.commit()
                    logger.info(
                        f"Processed batch {i // batch_size + 1} "
                        f"({min(i + batch_size, len(chunked_docs))}/{len(chunked_docs)})"
                    )

            stats["chunks_created"] = len(chunked_docs)
            logger.info(f"Completed ingestion for {ticker}: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Failed to ingest {ticker}: {e}")
            stats["errors"] += 1
            return stats

    def ingest_all_ai_stocks(self) -> List[Dict[str, int]]:
        """Ingest data for all AI stocks

        Returns:
            List of ingestion statistics per stock
        """
        ai_stocks = self.data_pipeline.get_ai_stocks()
        logger.info(f"Starting ingestion for {len(ai_stocks)} AI stocks")

        all_stats = []

        for ticker in tqdm(ai_stocks, desc="Ingesting stocks"):
            # Check budget before each stock
            if not self.cost_monitor.check_budget():
                logger.warning("Budget exceeded, stopping ingestion")
                break

            stats = self.ingest_stock(ticker)
            all_stats.append(stats)

        # Summary
        total_docs = sum(s["documents_created"] for s in all_stats)
        total_embeddings = sum(s["embeddings_created"] for s in all_stats)
        total_errors = sum(s["errors"] for s in all_stats)

        logger.info(
            f"Ingestion complete: "
            f"{total_docs} documents, "
            f"{total_embeddings} embeddings, "
            f"{total_errors} errors"
        )

        return all_stats

    def ingest_market_news(self, limit_per_feed: int = 20) -> Dict[str, int]:
        """Ingest general market news

        Args:
            limit_per_feed: Maximum articles per feed

        Returns:
            Ingestion statistics
        """
        logger.info("Starting market news ingestion")
        stats = {
            "documents_created": 0,
            "embeddings_created": 0,
            "errors": 0,
        }

        try:
            # Fetch news
            articles = self.data_pipeline.fetch_market_news(limit=limit_per_feed)
            logger.info(f"Fetched {len(articles)} news articles")

            if not articles:
                return stats

            # Convert to documents
            documents = []
            for article in articles:
                documents.append({
                    "content": f"{article['title']}\n\n{article.get('summary', '')}",
                    "metadata": {
                        "link": article.get("link"),
                        "source_name": article.get("source"),
                    },
                    "source": "financial_news",
                    "ticker": None,
                    "etf": None,
                    "document_type": "market_news",
                    "published_date": article.get("published"),
                })

            # Chunk documents
            chunked_docs = self.chunker.chunk_documents(documents)

            # Generate embeddings and store
            texts = [doc["content"] for doc in chunked_docs]
            embeddings = self.embedding_model.embed_documents(texts)

            with self.db_manager.get_session() as session:
                for doc, embedding in zip(chunked_docs, embeddings):
                    try:
                        db_doc = DocumentRepository.create_document(
                            session=session,
                            content=doc["content"],
                            metadata=doc.get("metadata", {}),
                            source=doc.get("source"),
                            ticker=doc.get("ticker"),
                            etf=doc.get("etf"),
                            document_type=doc.get("document_type"),
                            published_date=doc.get("published_date"),
                        )
                        stats["documents_created"] += 1

                        EmbeddingRepository.create_embedding(
                            session=session,
                            document_id=db_doc.id,
                            embedding=embedding,
                            model=self.embedding_model.config.model_name,
                        )
                        stats["embeddings_created"] += 1

                    except Exception as e:
                        logger.error(f"Failed to store news article: {e}")
                        stats["errors"] += 1

            logger.info(f"Market news ingestion complete: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Failed to ingest market news: {e}")
            stats["errors"] += 1
            return stats

    def update_etf_holdings(self) -> int:
        """Update ETF holdings in database

        Returns:
            Number of holdings added
        """
        logger.info("Updating ETF holdings")
        count = 0

        ai_stocks = self.data_pipeline.get_ai_stocks()

        with self.db_manager.get_session() as session:
            for ticker in ai_stocks:
                try:
                    # Check if holding already exists
                    existing = (
                        session.query(ETFHoldingRepository)
                        .filter_by(etf="QQQ", ticker=ticker)
                        .first()
                    )

                    if not existing:
                        ETFHoldingRepository.add_holding(
                            session=session,
                            etf="QQQ",
                            ticker=ticker,
                            is_ai_related=True,
                        )
                        count += 1

                except Exception as e:
                    logger.error(f"Failed to add holding for {ticker}: {e}")

        logger.info(f"Added {count} new ETF holdings")
        return count


def get_ingestion_pipeline() -> IngestionPipeline:
    """Get ingestion pipeline instance"""
    return IngestionPipeline()
