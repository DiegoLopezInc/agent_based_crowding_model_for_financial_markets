"""Agent tools for ETF research"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from backend.database.connection import get_db_manager
from backend.database.repository import (
    DocumentRepository,
    EmbeddingRepository,
    ETFHoldingRepository,
)
from backend.embeddings.embedding_model import get_embedding_model
from backend.utils.logger import setup_logger
from backend.utils.config import get_config

logger = setup_logger("agent_tools")


class ETFResearchTools:
    """Tools for ETF research agent"""

    def __init__(self):
        self.db_manager = get_db_manager()
        self.embedding_model = get_embedding_model()
        self.config = get_config().rag

    def search_documents(
        self,
        query: str,
        top_k: Optional[int] = None,
        ticker_filter: Optional[str] = None,
        etf_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for relevant documents using semantic search

        Args:
            query: Search query
            top_k: Number of results to return
            ticker_filter: Optional ticker filter
            etf_filter: Optional ETF filter

        Returns:
            List of relevant documents with metadata
        """
        if top_k is None:
            top_k = self.config.top_k

        logger.info(f"Searching documents: '{query}' (top_k={top_k})")

        try:
            # Generate query embedding
            query_embedding = self.embedding_model.embed_query(query)

            # Search database
            with self.db_manager.get_session() as session:
                results = EmbeddingRepository.search_similar(
                    session=session,
                    query_embedding=query_embedding,
                    limit=top_k,
                    similarity_threshold=self.config.similarity_threshold,
                    etf_filter=etf_filter,
                    ticker_filter=ticker_filter,
                )

            logger.info(f"Found {len(results)} documents")
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_stock_documents(
        self,
        ticker: str,
        document_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get documents for a specific stock

        Args:
            ticker: Stock ticker symbol
            document_type: Optional document type filter
            limit: Maximum number of documents

        Returns:
            List of documents
        """
        logger.info(f"Getting documents for {ticker}")

        try:
            with self.db_manager.get_session() as session:
                docs = DocumentRepository.get_documents_by_ticker(
                    session=session,
                    ticker=ticker.upper(),
                    limit=limit,
                )

                results = []
                for doc in docs:
                    if document_type is None or doc.document_type == document_type:
                        results.append({
                            "id": doc.id,
                            "content": doc.content,
                            "ticker": doc.ticker,
                            "etf": doc.etf,
                            "document_type": doc.document_type,
                            "source": doc.source,
                            "published_date": doc.published_date.isoformat() if doc.published_date else None,
                            "metadata": doc.metadata,
                        })

                logger.info(f"Found {len(results)} documents for {ticker}")
                return results

        except Exception as e:
            logger.error(f"Failed to get documents for {ticker}: {e}")
            return []

    def get_ai_stocks_list(self) -> List[str]:
        """Get list of AI-related stocks

        Returns:
            List of ticker symbols
        """
        logger.info("Getting AI stocks list")

        try:
            with self.db_manager.get_session() as session:
                stocks = ETFHoldingRepository.get_ai_stocks(session)

                logger.info(f"Found {len(stocks)} AI stocks")
                return stocks

        except Exception as e:
            logger.error(f"Failed to get AI stocks: {e}")
            return []

    def get_etf_holdings(
        self,
        etf: str,
        ai_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """Get holdings for an ETF

        Args:
            etf: ETF ticker symbol
            ai_only: Only return AI-related stocks

        Returns:
            List of holdings
        """
        logger.info(f"Getting holdings for {etf}")

        try:
            with self.db_manager.get_session() as session:
                holdings = ETFHoldingRepository.get_holdings_by_etf(
                    session=session,
                    etf=etf.upper(),
                    ai_only=ai_only,
                )

                results = []
                for holding in holdings:
                    results.append({
                        "ticker": holding.ticker,
                        "etf": holding.etf,
                        "weight": holding.weight,
                        "sector": holding.sector,
                        "is_ai_related": holding.is_ai_related,
                    })

                logger.info(f"Found {len(results)} holdings for {etf}")
                return results

        except Exception as e:
            logger.error(f"Failed to get holdings for {etf}: {e}")
            return []

    def compare_stocks(
        self,
        tickers: List[str],
        aspect: str = "general",
    ) -> Dict[str, Any]:
        """Compare multiple stocks based on available data

        Args:
            tickers: List of ticker symbols to compare
            aspect: Aspect to compare (general, news, earnings)

        Returns:
            Comparison data
        """
        logger.info(f"Comparing stocks: {tickers}")

        comparison = {
            "tickers": tickers,
            "aspect": aspect,
            "data": {},
        }

        try:
            for ticker in tickers:
                docs = self.get_stock_documents(
                    ticker=ticker,
                    document_type=aspect if aspect != "general" else None,
                    limit=5,
                )
                comparison["data"][ticker] = docs

            return comparison

        except Exception as e:
            logger.error(f"Failed to compare stocks: {e}")
            return comparison


def get_research_tools() -> ETFResearchTools:
    """Get research tools instance"""
    return ETFResearchTools()
