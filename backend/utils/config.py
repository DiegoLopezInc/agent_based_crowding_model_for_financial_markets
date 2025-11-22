"""Configuration management for the ETF Research RAG system"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


# Load environment variables
load_dotenv()


class FineTuningTrainingConfig(BaseModel):
    """Fine-tuning training configuration"""
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    evaluation_steps: int = 500
    save_steps: int = 500
    margin: float = 0.5
    distance_metric: str = "cosine"
    positive_samples_per_stock: int = 5
    negative_samples_per_stock: int = 5
    hard_negative_ratio: float = 0.3
    output_dir: str = "models/fine_tuning_checkpoints"
    logging_dir: str = "logs/fine_tuning"


class FineTuningConfig(BaseModel):
    """Fine-tuning configuration"""
    enabled: bool = False
    fine_tuned_model_path: str = "models/fine_tuned_embeddings"
    training: FineTuningTrainingConfig = Field(default_factory=FineTuningTrainingConfig)


class EmbeddingConfig(BaseModel):
    """Embedding model configuration"""
    provider: str = "huggingface"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    batch_size: int = 32
    max_length: int = 512
    fine_tuning: FineTuningConfig = Field(default_factory=FineTuningConfig)


class AgentConfig(BaseModel):
    """Agent configuration"""
    provider: str = "anthropic"
    model_name: str = "claude-sonnet-4-5-20250929"
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: int = 60


class DatabaseConfig(BaseModel):
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "etf_research"
    user: str = "postgres"
    password: str = Field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", "postgres"))
    pool_size: int = 10
    max_overflow: int = 20

    @property
    def connection_string(self) -> str:
        """Get SQLAlchemy connection string"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    @property
    def async_connection_string(self) -> str:
        """Get async SQLAlchemy connection string"""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class ETFConfig(BaseModel):
    """ETF configuration"""
    primary: str = "QQQ"
    focus_sector: str = "AI"
    opposite_etf: Optional[str] = "VYM"
    opposite_etf_stocks: List[str] = []
    additional_etfs: List[str] = []
    qqq_ai_stocks: List[str] = []


class DataSourceConfig(BaseModel):
    """Data source configuration"""
    name: str
    enabled: bool = True
    api_key: Optional[str] = None
    rss_feeds: Optional[List[str]] = None


class DataPipelineConfig(BaseModel):
    """Data pipeline configuration"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    sources: List[DataSourceConfig] = []
    update_frequency: str = "daily"


class CostMonitoringConfig(BaseModel):
    """Cost monitoring configuration"""
    enabled: bool = True
    max_daily_cost: float = 10.0
    max_embedding_cost: float = 5.0
    max_agent_cost: float = 5.0
    alert_threshold: float = 0.8
    log_file: str = "logs/cost_tracking.jsonl"


class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = "INFO"
    format: str = "json"
    output_dir: str = "logs"
    retention_days: int = 30
    enable_console: bool = True
    enable_file: bool = True


class RAGConfig(BaseModel):
    """RAG configuration"""
    top_k: int = 5
    similarity_threshold: float = 0.7
    rerank: bool = True
    rerank_top_k: int = 3
    use_hyde: bool = False


class RewardFunctionConfig(BaseModel):
    """Reward function weights for agent evaluation"""
    relevance_weight: float = 0.5
    accuracy_weight: float = 0.3
    completeness_weight: float = 0.2


class EvaluationConfig(BaseModel):
    """Evaluation configuration"""
    enabled: bool = True
    metrics: List[str] = ["relevance", "accuracy", "completeness"]
    reward_function: RewardFunctionConfig = Field(default_factory=RewardFunctionConfig)
    validation_budget: float = 2.0


class MCPServerConfig(BaseModel):
    """MCP Server configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    log_level: str = "info"


class Config(BaseModel):
    """Main configuration class"""
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    etfs: ETFConfig = Field(default_factory=ETFConfig)
    data_pipeline: DataPipelineConfig = Field(default_factory=DataPipelineConfig)
    cost_monitoring: CostMonitoringConfig = Field(default_factory=CostMonitoringConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    mcp_server: MCPServerConfig = Field(default_factory=MCPServerConfig)

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "Config":
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Handle data sources specially
        if "data_pipeline" in config_dict and "sources" in config_dict["data_pipeline"]:
            sources = []
            for source in config_dict["data_pipeline"]["sources"]:
                # Resolve environment variables for API keys
                if "api_key" in source and isinstance(source["api_key"], str):
                    if source["api_key"].startswith("${") and source["api_key"].endswith("}"):
                        env_var = source["api_key"][2:-1]
                        source["api_key"] = os.getenv(env_var)
                sources.append(DataSourceConfig(**source))
            config_dict["data_pipeline"]["sources"] = sources

        return cls(**config_dict)

    def save_yaml(self, config_path: str | Path) -> None:
        """Save configuration to YAML file"""
        with open(config_path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)


# Global config instance
_config: Optional[Config] = None


def get_config(config_path: Optional[str | Path] = None) -> Config:
    """Get the global configuration instance"""
    global _config

    if _config is None:
        if config_path is None:
            # Look for config.yaml in project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config.yaml"

        if not Path(config_path).exists():
            # Return default config
            _config = Config()
        else:
            _config = Config.from_yaml(config_path)

    return _config


def reload_config(config_path: Optional[str | Path] = None) -> Config:
    """Reload the configuration"""
    global _config
    _config = None
    return get_config(config_path)
