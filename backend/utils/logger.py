"""Logging utilities for the ETF Research RAG system"""

import sys
import json
from pathlib import Path
from typing import Optional
from loguru import logger
from datetime import datetime

from backend.utils.config import get_config


def setup_logger(name: Optional[str] = None) -> logger:
    """Set up and configure the logger

    Args:
        name: Optional logger name for contextualization

    Returns:
        Configured logger instance
    """
    config = get_config()
    log_config = config.logging

    # Remove default handler
    logger.remove()

    # Create logs directory
    log_dir = Path(log_config.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Console handler
    if log_config.enable_console:
        if log_config.format == "json":
            logger.add(
                sys.stdout,
                format="{message}",
                level=log_config.level,
                serialize=True,
            )
        else:
            logger.add(
                sys.stdout,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                level=log_config.level,
            )

    # File handler
    if log_config.enable_file:
        log_file = log_dir / f"etf_rag_{datetime.now().strftime('%Y%m%d')}.log"

        if log_config.format == "json":
            logger.add(
                log_file,
                format="{message}",
                level=log_config.level,
                rotation="1 day",
                retention=f"{log_config.retention_days} days",
                serialize=True,
            )
        else:
            logger.add(
                log_file,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                level=log_config.level,
                rotation="1 day",
                retention=f"{log_config.retention_days} days",
            )

    # Add context if name is provided
    if name:
        return logger.bind(module=name)

    return logger


def log_execution_time(func):
    """Decorator to log function execution time"""
    def wrapper(*args, **kwargs):
        start = datetime.now()
        result = func(*args, **kwargs)
        duration = (datetime.now() - start).total_seconds()
        logger.info(f"{func.__name__} executed in {duration:.2f}s")
        return result
    return wrapper


class StructuredLogger:
    """Structured logger for consistent logging format"""

    def __init__(self, name: str):
        self.logger = setup_logger(name)
        self.name = name

    def log_event(
        self,
        event_type: str,
        message: str,
        level: str = "INFO",
        **kwargs
    ):
        """Log a structured event

        Args:
            event_type: Type of event (e.g., "data_pipeline", "embedding", "agent_call")
            message: Log message
            level: Log level
            **kwargs: Additional structured data
        """
        log_data = {
            "event_type": event_type,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "module": self.name,
            **kwargs
        }

        log_method = getattr(self.logger, level.lower())
        log_method(json.dumps(log_data))

    def log_api_call(
        self,
        provider: str,
        endpoint: str,
        duration: float,
        cost: Optional[float] = None,
        tokens: Optional[int] = None,
        **kwargs
    ):
        """Log an API call with cost tracking"""
        self.log_event(
            "api_call",
            f"API call to {provider}/{endpoint}",
            provider=provider,
            endpoint=endpoint,
            duration_seconds=duration,
            cost_usd=cost,
            tokens=tokens,
            **kwargs
        )

    def log_embedding(
        self,
        num_texts: int,
        model: str,
        duration: float,
        cost: Optional[float] = None,
    ):
        """Log embedding generation"""
        self.log_event(
            "embedding",
            f"Generated embeddings for {num_texts} texts",
            num_texts=num_texts,
            model=model,
            duration_seconds=duration,
            cost_usd=cost,
        )

    def log_query(
        self,
        query: str,
        results_count: int,
        duration: float,
        **kwargs
    ):
        """Log a RAG query"""
        self.log_event(
            "rag_query",
            f"Query returned {results_count} results",
            query=query,
            results_count=results_count,
            duration_seconds=duration,
            **kwargs
        )
