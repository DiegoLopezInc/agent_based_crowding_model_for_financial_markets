"""Database connection manager"""

from typing import Optional, Generator
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from backend.utils.config import get_config
from backend.utils.logger import setup_logger
from backend.database.models import Base

logger = setup_logger("database")


class DatabaseManager:
    """Manages database connections and sessions"""

    def __init__(self):
        self.config = get_config().database
        self.engine = None
        self.session_factory = None
        self._initialize_engine()

    def _initialize_engine(self):
        """Initialize SQLAlchemy engine"""
        logger.info(f"Connecting to database: {self.config.database}")

        self.engine = create_engine(
            self.config.connection_string,
            poolclass=QueuePool,
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
            pool_pre_ping=True,  # Verify connections before using
            echo=False,  # Set to True for SQL debugging
        )

        self.session_factory = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
        )

        logger.info("Database connection established")

    def create_tables(self):
        """Create all tables if they don't exist"""
        logger.info("Creating database tables...")

        # First enable vector extension
        with self.engine.connect() as conn:
            try:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.execute(text("CREATE SCHEMA IF NOT EXISTS etf_research"))
                conn.commit()
                logger.info("Vector extension and schema created")
            except Exception as e:
                logger.warning(f"Extension/schema creation warning: {e}")

        # Create tables
        Base.metadata.create_all(self.engine)
        logger.info("Database tables created successfully")

    def drop_tables(self):
        """Drop all tables (use with caution!)"""
        logger.warning("Dropping all database tables...")
        Base.metadata.drop_all(self.engine)
        logger.info("All tables dropped")

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Context manager for database sessions

        Usage:
            with db_manager.get_session() as session:
                # Use session
                pass
        """
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()

    def execute_raw_sql(self, sql: str, params: Optional[dict] = None):
        """Execute raw SQL query

        Args:
            sql: SQL query string
            params: Optional parameters dictionary

        Returns:
            Query result
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            conn.commit()
            return result

    def check_connection(self) -> bool:
        """Check if database connection is healthy

        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection is healthy")
            return True
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            return False

    def get_stats(self) -> dict:
        """Get database statistics

        Returns:
            Dictionary with database stats
        """
        with self.get_session() as session:
            stats = {}

            # Count documents
            stats["documents"] = session.execute(
                text("SELECT COUNT(*) FROM etf_research.documents")
            ).scalar()

            # Count embeddings
            stats["embeddings"] = session.execute(
                text("SELECT COUNT(*) FROM etf_research.embeddings")
            ).scalar()

            # Count ETF holdings
            stats["etf_holdings"] = session.execute(
                text("SELECT COUNT(*) FROM etf_research.etf_holdings")
            ).scalar()

            # Count queries
            stats["queries"] = session.execute(
                text("SELECT COUNT(*) FROM etf_research.query_history")
            ).scalar()

            # Get unique tickers
            stats["unique_tickers"] = session.execute(
                text("SELECT COUNT(DISTINCT ticker) FROM etf_research.documents")
            ).scalar()

            # Get unique ETFs
            stats["unique_etfs"] = session.execute(
                text("SELECT COUNT(DISTINCT etf) FROM etf_research.etf_holdings")
            ).scalar()

            logger.info(f"Database stats: {stats}")
            return stats

    def close(self):
        """Close database connections"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connections closed")


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def init_database():
    """Initialize the database (create tables, etc.)"""
    db_manager = get_db_manager()
    db_manager.create_tables()
    return db_manager
