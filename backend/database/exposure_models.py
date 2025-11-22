"""SQLAlchemy models for exposure analysis"""

from datetime import date, datetime
from typing import Optional, List
from sqlalchemy import (
    Column, Integer, String, Float, Date, BigInteger,
    DateTime, ARRAY, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base

from backend.database.models import Base


class StockPrice(Base):
    """Stock price history for exposure calculations"""

    __tablename__ = "stock_prices"
    __table_args__ = (
        UniqueConstraint('ticker', 'date', name='uq_stock_price_ticker_date'),
        {"schema": "etf_research"}
    )

    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    adjusted_close = Column(Float)
    volume = Column(BigInteger)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<StockPrice(ticker={self.ticker}, date={self.date}, close={self.close})>"


class ETFCompositionSnapshot(Base):
    """ETF composition at a point in time"""

    __tablename__ = "etf_composition_snapshots"
    __table_args__ = (
        UniqueConstraint('etf', 'snapshot_date', name='uq_etf_snapshot'),
        {"schema": "etf_research"}
    )

    id = Column(Integer, primary_key=True)
    etf = Column(String(10), nullable=False, index=True)
    snapshot_date = Column(Date, nullable=False, index=True)
    holdings = Column(JSONB, nullable=False)  # [{ticker, weight, shares}, ...]
    total_holdings = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<ETFCompositionSnapshot(etf={self.etf}, date={self.snapshot_date}, holdings={self.total_holdings})>"


class ExposureMatrix(Base):
    """Cached exposure matrices"""

    __tablename__ = "exposure_matrices"
    __table_args__ = (
        UniqueConstraint('etf', 'matrix_type', 'start_date', 'end_date', name='uq_exposure_matrix'),
        {"schema": "etf_research"}
    )

    id = Column(Integer, primary_key=True)
    etf = Column(String(10), nullable=False, index=True)
    matrix_type = Column(String(50), nullable=False, index=True)  # correlation, covariance, idiosyncratic
    tickers = Column(ARRAY(String), nullable=False)  # Ordered list
    matrix_data = Column(JSONB, nullable=False)  # Serialized matrix
    computation_params = Column(JSONB)
    start_date = Column(Date, index=True)
    end_date = Column(Date, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<ExposureMatrix(etf={self.etf}, type={self.matrix_type}, tickers={len(self.tickers)})>"


class PairwiseExposure(Base):
    """Pairwise exposure between two stocks"""

    __tablename__ = "pairwise_exposures"
    __table_args__ = (
        UniqueConstraint('etf', 'ticker1', 'ticker2', 'computation_date', name='uq_pairwise_exposure'),
        {"schema": "etf_research"}
    )

    id = Column(Integer, primary_key=True)
    etf = Column(String(10), nullable=False, index=True)
    ticker1 = Column(String(10), nullable=False, index=True)
    ticker2 = Column(String(10), nullable=False, index=True)
    correlation = Column(Float)
    covariance = Column(Float)
    idiosyncratic_score = Column(Float)
    common_factor_exposure = Column(Float)
    specific_exposure = Column(Float)
    computation_date = Column(Date, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<PairwiseExposure({self.ticker1}-{self.ticker2}, corr={self.correlation:.3f})>"


class ETFComparison(Base):
    """Comparison between two ETFs"""

    __tablename__ = "etf_comparisons"
    __table_args__ = (
        UniqueConstraint('etf1', 'etf2', 'computation_date', name='uq_etf_comparison'),
        {"schema": "etf_research"}
    )

    id = Column(Integer, primary_key=True)
    etf1 = Column(String(10), nullable=False, index=True)
    etf2 = Column(String(10), nullable=False, index=True)
    comparison_matrix = Column(JSONB, nullable=False)
    common_tickers = Column(ARRAY(String))
    etf1_only_tickers = Column(ARRAY(String))
    etf2_only_tickers = Column(ARRAY(String))
    similarity_score = Column(Float)
    computation_date = Column(Date, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<ETFComparison({self.etf1}-{self.etf2}, similarity={self.similarity_score:.3f})>"
