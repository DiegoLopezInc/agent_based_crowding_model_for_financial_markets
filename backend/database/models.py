"""SQLAlchemy models for the ETF Research database"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    Column, Integer, String, Text, Float, Boolean,
    DateTime, ForeignKey, ARRAY, func, Index
)
from sqlalchemy.dialects.postgresql import JSONB, VECTOR
from sqlalchemy.orm import declarative_base, relationship
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class Document(Base):
    """Document model for storing text chunks"""

    __tablename__ = "documents"
    __table_args__ = {"schema": "etf_research"}

    id = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)
    metadata = Column(JSONB, default={})
    source = Column(String(255))
    ticker = Column(String(10), index=True)
    etf = Column(String(10), index=True)
    document_type = Column(String(50), index=True)
    published_date = Column(DateTime, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    embeddings = relationship("Embedding", back_populates="document", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Document(id={self.id}, ticker={self.ticker}, type={self.document_type})>"


class Embedding(Base):
    """Embedding model for storing vector embeddings"""

    __tablename__ = "embeddings"
    __table_args__ = {"schema": "etf_research"}

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("etf_research.documents.id", ondelete="CASCADE"), index=True)
    embedding = Column(Vector(384))  # Default dimension for all-MiniLM-L6-v2
    model = Column(String(255), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="embeddings")

    def __repr__(self):
        return f"<Embedding(id={self.id}, document_id={self.document_id}, model={self.model})>"


class ETFHolding(Base):
    """ETF holdings model for tracking stocks in ETFs"""

    __tablename__ = "etf_holdings"
    __table_args__ = {"schema": "etf_research"}

    id = Column(Integer, primary_key=True)
    etf = Column(String(10), nullable=False, index=True)
    ticker = Column(String(10), nullable=False, index=True)
    weight = Column(Float)
    sector = Column(String(100))
    is_ai_related = Column(Boolean, default=False, index=True)
    added_date = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<ETFHolding(etf={self.etf}, ticker={self.ticker})>"


class QueryHistory(Base):
    """Query history for tracking user queries"""

    __tablename__ = "query_history"
    __table_args__ = {"schema": "etf_research"}

    id = Column(Integer, primary_key=True)
    query = Column(Text, nullable=False)
    results_count = Column(Integer)
    top_tickers = Column(ARRAY(String))
    agent_used = Column(Boolean, default=False)
    response_time_ms = Column(Integer)
    cost_usd = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    evaluations = relationship("AgentEvaluation", back_populates="query")

    def __repr__(self):
        return f"<QueryHistory(id={self.id}, query={self.query[:50]}...)>"


class AgentEvaluation(Base):
    """Agent evaluation metrics"""

    __tablename__ = "agent_evaluations"
    __table_args__ = {"schema": "etf_research"}

    id = Column(Integer, primary_key=True)
    query_id = Column(Integer, ForeignKey("etf_research.query_history.id"))
    relevance_score = Column(Float)
    accuracy_score = Column(Float)
    completeness_score = Column(Float)
    reward_score = Column(Float)
    feedback = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    query = relationship("QueryHistory", back_populates="evaluations")

    def __repr__(self):
        return f"<AgentEvaluation(id={self.id}, reward={self.reward_score})>"
