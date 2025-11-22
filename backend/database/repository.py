"""Database repository for CRUD operations"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy import text, func
from sqlalchemy.orm import Session

from backend.database.models import Document, Embedding, ETFHolding, QueryHistory, AgentEvaluation
from backend.database.connection import get_db_manager
from backend.utils.logger import setup_logger

logger = setup_logger("repository")


class DocumentRepository:
    """Repository for document operations"""

    @staticmethod
    def create_document(
        session: Session,
        content: str,
        metadata: Dict[str, Any],
        source: str,
        ticker: Optional[str] = None,
        etf: Optional[str] = None,
        document_type: Optional[str] = None,
        published_date: Optional[datetime] = None,
    ) -> Document:
        """Create a new document"""
        doc = Document(
            content=content,
            metadata=metadata,
            source=source,
            ticker=ticker,
            etf=etf,
            document_type=document_type,
            published_date=published_date,
        )
        session.add(doc)
        session.flush()
        return doc

    @staticmethod
    def get_document(session: Session, document_id: int) -> Optional[Document]:
        """Get a document by ID"""
        return session.query(Document).filter(Document.id == document_id).first()

    @staticmethod
    def get_documents_by_ticker(
        session: Session,
        ticker: str,
        limit: int = 100
    ) -> List[Document]:
        """Get documents for a specific ticker"""
        return (
            session.query(Document)
            .filter(Document.ticker == ticker)
            .order_by(Document.published_date.desc())
            .limit(limit)
            .all()
        )

    @staticmethod
    def get_documents_by_etf(
        session: Session,
        etf: str,
        limit: int = 100
    ) -> List[Document]:
        """Get documents for a specific ETF"""
        return (
            session.query(Document)
            .filter(Document.etf == etf)
            .order_by(Document.published_date.desc())
            .limit(limit)
            .all()
        )

    @staticmethod
    def delete_document(session: Session, document_id: int):
        """Delete a document (embeddings are cascade deleted)"""
        doc = session.query(Document).filter(Document.id == document_id).first()
        if doc:
            session.delete(doc)


class EmbeddingRepository:
    """Repository for embedding operations"""

    @staticmethod
    def create_embedding(
        session: Session,
        document_id: int,
        embedding: List[float],
        model: str,
    ) -> Embedding:
        """Create a new embedding"""
        emb = Embedding(
            document_id=document_id,
            embedding=embedding,
            model=model,
        )
        session.add(emb)
        session.flush()
        return emb

    @staticmethod
    def search_similar(
        session: Session,
        query_embedding: List[float],
        limit: int = 5,
        similarity_threshold: float = 0.7,
        etf_filter: Optional[str] = None,
        ticker_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar documents using vector similarity

        Args:
            session: Database session
            query_embedding: Query embedding vector
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score (0-1)
            etf_filter: Optional ETF filter
            ticker_filter: Optional ticker filter

        Returns:
            List of dictionaries with document info and similarity scores
        """
        # Use the custom PostgreSQL function
        sql = text("""
            SELECT * FROM etf_research.search_similar_documents(
                :query_embedding::vector,
                :threshold,
                :limit,
                :etf_filter,
                :ticker_filter
            )
        """)

        result = session.execute(
            sql,
            {
                "query_embedding": str(query_embedding),
                "threshold": similarity_threshold,
                "limit": limit,
                "etf_filter": etf_filter,
                "ticker_filter": ticker_filter,
            }
        )

        results = []
        for row in result:
            results.append({
                "id": row.id,
                "content": row.content,
                "metadata": row.metadata,
                "ticker": row.ticker,
                "etf": row.etf,
                "similarity": float(row.similarity),
            })

        return results

    @staticmethod
    def get_embedding_by_document(
        session: Session,
        document_id: int,
        model: str
    ) -> Optional[Embedding]:
        """Get embedding for a document and model"""
        return (
            session.query(Embedding)
            .filter(
                Embedding.document_id == document_id,
                Embedding.model == model
            )
            .first()
        )


class ETFHoldingRepository:
    """Repository for ETF holding operations"""

    @staticmethod
    def add_holding(
        session: Session,
        etf: str,
        ticker: str,
        weight: Optional[float] = None,
        sector: Optional[str] = None,
        is_ai_related: bool = False,
    ) -> ETFHolding:
        """Add a new ETF holding"""
        holding = ETFHolding(
            etf=etf,
            ticker=ticker,
            weight=weight,
            sector=sector,
            is_ai_related=is_ai_related,
        )
        session.add(holding)
        session.flush()
        return holding

    @staticmethod
    def get_holdings_by_etf(
        session: Session,
        etf: str,
        ai_only: bool = False
    ) -> List[ETFHolding]:
        """Get all holdings for an ETF"""
        query = session.query(ETFHolding).filter(ETFHolding.etf == etf)

        if ai_only:
            query = query.filter(ETFHolding.is_ai_related == True)

        return query.all()

    @staticmethod
    def get_ai_stocks(session: Session) -> List[str]:
        """Get all AI-related stock tickers"""
        holdings = (
            session.query(ETFHolding.ticker)
            .filter(ETFHolding.is_ai_related == True)
            .distinct()
            .all()
        )
        return [h.ticker for h in holdings]


class QueryHistoryRepository:
    """Repository for query history operations"""

    @staticmethod
    def create_query(
        session: Session,
        query: str,
        results_count: int,
        top_tickers: List[str],
        agent_used: bool = False,
        response_time_ms: Optional[int] = None,
        cost_usd: Optional[float] = None,
    ) -> QueryHistory:
        """Create a new query history entry"""
        query_history = QueryHistory(
            query=query,
            results_count=results_count,
            top_tickers=top_tickers,
            agent_used=agent_used,
            response_time_ms=response_time_ms,
            cost_usd=cost_usd,
        )
        session.add(query_history)
        session.flush()
        return query_history

    @staticmethod
    def get_recent_queries(
        session: Session,
        limit: int = 10
    ) -> List[QueryHistory]:
        """Get recent queries"""
        return (
            session.query(QueryHistory)
            .order_by(QueryHistory.created_at.desc())
            .limit(limit)
            .all()
        )


class AgentEvaluationRepository:
    """Repository for agent evaluation operations"""

    @staticmethod
    def create_evaluation(
        session: Session,
        query_id: int,
        relevance_score: float,
        accuracy_score: float,
        completeness_score: float,
        reward_score: float,
        feedback: Optional[str] = None,
    ) -> AgentEvaluation:
        """Create a new agent evaluation"""
        evaluation = AgentEvaluation(
            query_id=query_id,
            relevance_score=relevance_score,
            accuracy_score=accuracy_score,
            completeness_score=completeness_score,
            reward_score=reward_score,
            feedback=feedback,
        )
        session.add(evaluation)
        session.flush()
        return evaluation

    @staticmethod
    def get_average_scores(session: Session) -> Dict[str, float]:
        """Get average evaluation scores"""
        result = session.query(
            func.avg(AgentEvaluation.relevance_score).label("avg_relevance"),
            func.avg(AgentEvaluation.accuracy_score).label("avg_accuracy"),
            func.avg(AgentEvaluation.completeness_score).label("avg_completeness"),
            func.avg(AgentEvaluation.reward_score).label("avg_reward"),
        ).first()

        return {
            "avg_relevance": float(result.avg_relevance or 0),
            "avg_accuracy": float(result.avg_accuracy or 0),
            "avg_completeness": float(result.avg_completeness or 0),
            "avg_reward": float(result.avg_reward or 0),
        }
