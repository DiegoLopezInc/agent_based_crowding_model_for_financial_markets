"""Cost monitoring utilities for tracking API and embedding costs"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional
from dataclasses import dataclass, asdict
import threading

from backend.utils.config import get_config
from backend.utils.logger import setup_logger


logger = setup_logger("cost_monitor")


@dataclass
class CostEntry:
    """Single cost entry"""
    timestamp: str
    service: str  # "embedding" or "agent"
    provider: str
    model: str
    operation: str
    tokens: Optional[int] = None
    cost_usd: float = 0.0
    metadata: Optional[Dict] = None


class CostMonitor:
    """Monitor and track costs for embeddings and agent API calls"""

    # Pricing (USD per 1000 tokens or per call)
    PRICING = {
        "anthropic": {
            "claude-sonnet-4-5-20250929": {
                "input": 0.003,   # $3 per million input tokens
                "output": 0.015,  # $15 per million output tokens
            },
            "claude-sonnet-3-5-20241022": {
                "input": 0.003,
                "output": 0.015,
            },
        },
        "openai": {
            "gpt-4-turbo": {
                "input": 0.01,
                "output": 0.03,
            },
            "gpt-3.5-turbo": {
                "input": 0.0005,
                "output": 0.0015,
            },
        },
        "huggingface": {
            # HuggingFace embeddings are free for local inference
            "sentence-transformers/all-MiniLM-L6-v2": {
                "embedding": 0.0,
            },
            "BAAI/bge-large-en-v1.5": {
                "embedding": 0.0,
            },
        },
    }

    def __init__(self):
        self.config = get_config()
        self.cost_config = self.config.cost_monitoring

        if not self.cost_config.enabled:
            logger.info("Cost monitoring is disabled")
            return

        # Create logs directory
        log_path = Path(self.cost_config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = log_path

        # Thread-safe lock for writing
        self._lock = threading.Lock()

        # In-memory cache of today's costs
        self._daily_costs: Dict[str, float] = {
            "embedding": 0.0,
            "agent": 0.0,
            "total": 0.0,
        }
        self._last_reset = datetime.now().date()

        # Load today's costs
        self._load_daily_costs()

    def _load_daily_costs(self):
        """Load costs from today's entries"""
        if not self.log_file.exists():
            return

        today = datetime.now().date()
        with open(self.log_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    entry_date = datetime.fromisoformat(entry["timestamp"]).date()

                    if entry_date == today:
                        service = entry["service"]
                        cost = entry["cost_usd"]

                        if service in self._daily_costs:
                            self._daily_costs[service] += cost
                            self._daily_costs["total"] += cost
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.warning(f"Failed to parse cost entry: {e}")

    def _reset_if_new_day(self):
        """Reset daily costs if it's a new day"""
        today = datetime.now().date()
        if today != self._last_reset:
            self._daily_costs = {
                "embedding": 0.0,
                "agent": 0.0,
                "total": 0.0,
            }
            self._last_reset = today

    def _write_entry(self, entry: CostEntry):
        """Write a cost entry to the log file"""
        if not self.cost_config.enabled:
            return

        with self._lock:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(asdict(entry)) + '\n')

    def _calculate_cost(
        self,
        provider: str,
        model: str,
        operation: str,
        tokens: Optional[int] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ) -> float:
        """Calculate cost based on provider, model, and token usage"""

        if provider not in self.PRICING:
            logger.warning(f"Unknown provider: {provider}, cost set to 0")
            return 0.0

        if model not in self.PRICING[provider]:
            logger.warning(f"Unknown model: {model} for provider {provider}, cost set to 0")
            return 0.0

        pricing = self.PRICING[provider][model]

        # For embeddings
        if operation == "embedding":
            if "embedding" in pricing:
                return pricing["embedding"]
            # Some providers charge by tokens for embeddings
            if tokens and "input" in pricing:
                return (tokens / 1000) * pricing["input"]
            return 0.0

        # For agent calls (chat completions)
        cost = 0.0
        if input_tokens and "input" in pricing:
            cost += (input_tokens / 1000) * pricing["input"]
        if output_tokens and "output" in pricing:
            cost += (output_tokens / 1000) * pricing["output"]

        return cost

    def log_embedding_cost(
        self,
        provider: str,
        model: str,
        num_texts: int,
        tokens: Optional[int] = None,
    ) -> float:
        """Log embedding cost

        Args:
            provider: Provider name (e.g., "huggingface")
            model: Model name
            num_texts: Number of texts embedded
            tokens: Total tokens processed

        Returns:
            Cost in USD
        """
        if not self.cost_config.enabled:
            return 0.0

        self._reset_if_new_day()

        cost = self._calculate_cost(
            provider=provider,
            model=model,
            operation="embedding",
            tokens=tokens,
        )

        entry = CostEntry(
            timestamp=datetime.now().isoformat(),
            service="embedding",
            provider=provider,
            model=model,
            operation="embedding",
            tokens=tokens,
            cost_usd=cost,
            metadata={"num_texts": num_texts}
        )

        self._write_entry(entry)

        # Update daily costs
        with self._lock:
            self._daily_costs["embedding"] += cost
            self._daily_costs["total"] += cost

        self._check_budget_alerts()

        logger.info(f"Embedding cost: ${cost:.4f} ({num_texts} texts)")
        return cost

    def log_agent_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        operation: str = "chat_completion",
    ) -> float:
        """Log agent API call cost

        Args:
            provider: Provider name (e.g., "anthropic", "openai")
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            operation: Operation type

        Returns:
            Cost in USD
        """
        if not self.cost_config.enabled:
            return 0.0

        self._reset_if_new_day()

        cost = self._calculate_cost(
            provider=provider,
            model=model,
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        entry = CostEntry(
            timestamp=datetime.now().isoformat(),
            service="agent",
            provider=provider,
            model=model,
            operation=operation,
            tokens=input_tokens + output_tokens,
            cost_usd=cost,
            metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        )

        self._write_entry(entry)

        # Update daily costs
        with self._lock:
            self._daily_costs["agent"] += cost
            self._daily_costs["total"] += cost

        self._check_budget_alerts()

        logger.info(
            f"Agent cost: ${cost:.4f} "
            f"(in: {input_tokens}, out: {output_tokens})"
        )
        return cost

    def _check_budget_alerts(self):
        """Check if costs exceed alert thresholds"""
        threshold = self.cost_config.alert_threshold

        # Check embedding budget
        embedding_limit = self.cost_config.max_embedding_cost
        embedding_cost = self._daily_costs["embedding"]
        if embedding_cost >= embedding_limit * threshold:
            logger.warning(
                f"Embedding cost alert: ${embedding_cost:.2f} "
                f"(limit: ${embedding_limit:.2f})"
            )

        # Check agent budget
        agent_limit = self.cost_config.max_agent_cost
        agent_cost = self._daily_costs["agent"]
        if agent_cost >= agent_limit * threshold:
            logger.warning(
                f"Agent cost alert: ${agent_cost:.2f} "
                f"(limit: ${agent_limit:.2f})"
            )

        # Check total budget
        total_limit = self.cost_config.max_daily_cost
        total_cost = self._daily_costs["total"]
        if total_cost >= total_limit * threshold:
            logger.warning(
                f"Total cost alert: ${total_cost:.2f} "
                f"(limit: ${total_limit:.2f})"
            )

    def check_budget(self, service: Optional[str] = None) -> bool:
        """Check if we're within budget

        Args:
            service: Check specific service ("embedding" or "agent") or total

        Returns:
            True if within budget, False otherwise
        """
        if not self.cost_config.enabled:
            return True

        self._reset_if_new_day()

        if service == "embedding":
            return self._daily_costs["embedding"] < self.cost_config.max_embedding_cost
        elif service == "agent":
            return self._daily_costs["agent"] < self.cost_config.max_agent_cost
        else:
            return self._daily_costs["total"] < self.cost_config.max_daily_cost

    def get_daily_costs(self) -> Dict[str, float]:
        """Get current daily costs"""
        self._reset_if_new_day()
        return self._daily_costs.copy()

    def get_cost_summary(self, days: int = 7) -> Dict[str, any]:
        """Get cost summary for the last N days

        Args:
            days: Number of days to summarize

        Returns:
            Summary dictionary with daily breakdown
        """
        if not self.log_file.exists():
            return {"daily": [], "total": 0.0}

        cutoff_date = datetime.now() - timedelta(days=days)
        daily_costs = {}

        with open(self.log_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    entry_date = datetime.fromisoformat(entry["timestamp"])

                    if entry_date >= cutoff_date:
                        date_key = entry_date.date().isoformat()
                        if date_key not in daily_costs:
                            daily_costs[date_key] = {
                                "embedding": 0.0,
                                "agent": 0.0,
                                "total": 0.0,
                            }

                        service = entry["service"]
                        cost = entry["cost_usd"]

                        if service in daily_costs[date_key]:
                            daily_costs[date_key][service] += cost
                            daily_costs[date_key]["total"] += cost

                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    continue

        total = sum(day["total"] for day in daily_costs.values())

        return {
            "daily": [{"date": k, **v} for k, v in sorted(daily_costs.items())],
            "total": total,
        }


# Global cost monitor instance
_cost_monitor: Optional[CostMonitor] = None


def get_cost_monitor() -> CostMonitor:
    """Get the global cost monitor instance"""
    global _cost_monitor
    if _cost_monitor is None:
        _cost_monitor = CostMonitor()
    return _cost_monitor
