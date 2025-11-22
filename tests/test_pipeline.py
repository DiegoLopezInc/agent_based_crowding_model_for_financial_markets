"""Integration tests for the complete pipeline"""

import pytest
from backend.database.connection import get_db_manager, init_database
from backend.embeddings.embedding_model import get_embedding_model
from backend.data_pipeline.chunker import get_chunker
from backend.data_pipeline.ingestion import get_ingestion_pipeline
from backend.agent.agent import get_agent
from backend.agent.evaluation import get_evaluator
from backend.utils.config import get_config


class TestDatabaseConnection:
    """Test database connectivity"""

    def test_database_connection(self):
        """Test database connection is healthy"""
        db_manager = get_db_manager()
        assert db_manager.check_connection(), "Database connection failed"

    def test_database_stats(self):
        """Test getting database statistics"""
        db_manager = get_db_manager()
        stats = db_manager.get_stats()

        assert isinstance(stats, dict), "Stats should be a dictionary"
        assert "documents" in stats, "Stats should include document count"
        assert "embeddings" in stats, "Stats should include embedding count"


class TestEmbeddingModel:
    """Test embedding model"""

    def test_model_loading(self):
        """Test embedding model loads correctly"""
        model = get_embedding_model()
        assert model is not None, "Embedding model should load"

        info = model.get_model_info()
        assert "dimension" in info, "Model info should include dimension"
        assert info["dimension"] > 0, "Dimension should be positive"

    def test_single_embedding(self):
        """Test generating single embedding"""
        model = get_embedding_model()
        text = "Apple Inc. is a technology company focusing on AI and consumer electronics."

        embedding = model.embed(text)

        assert isinstance(embedding, list), "Embedding should be a list"
        assert len(embedding) == model.get_dimension(), "Embedding dimension should match model"
        assert all(isinstance(x, float) for x in embedding), "All values should be floats"

    def test_batch_embedding(self):
        """Test generating batch embeddings"""
        model = get_embedding_model()
        texts = [
            "NVIDIA is a leader in AI chip technology.",
            "Microsoft develops cloud computing and AI solutions.",
            "Tesla manufactures electric vehicles and autonomous driving technology.",
        ]

        embeddings = model.embed(texts)

        assert isinstance(embeddings, list), "Should return list of embeddings"
        assert len(embeddings) == len(texts), "Should have one embedding per text"
        assert all(len(emb) == model.get_dimension() for emb in embeddings), "All embeddings should have correct dimension"


class TestChunking:
    """Test text chunking"""

    def test_chunk_text(self):
        """Test basic text chunking"""
        chunker = get_chunker()
        text = "This is a test. " * 200  # Create long text

        chunks = chunker.chunk_text(text)

        assert isinstance(chunks, list), "Should return list of chunks"
        assert len(chunks) > 1, "Long text should be split into multiple chunks"
        assert all(isinstance(chunk, str) for chunk in chunks), "All chunks should be strings"

    def test_chunk_document(self):
        """Test document chunking with metadata preservation"""
        chunker = get_chunker()
        document = {
            "content": "Financial analysis of technology stocks. " * 100,
            "metadata": {"ticker": "NVDA", "type": "analysis"},
            "source": "test",
        }

        chunked_docs = chunker.chunk_document(document)

        assert len(chunked_docs) > 0, "Should create chunked documents"
        assert all("metadata" in doc for doc in chunked_docs), "Metadata should be preserved"
        assert all(doc["source"] == "test" for doc in chunked_docs), "Source should be preserved"


class TestIngestionPipeline:
    """Test data ingestion pipeline"""

    @pytest.mark.skipif(
        not get_config().data_pipeline.sources,
        reason="No data sources configured"
    )
    def test_fetch_stock_data(self):
        """Test fetching stock data"""
        pipeline = get_ingestion_pipeline()

        # Test with a well-known ticker
        data = pipeline.data_pipeline.fetch_stock_data("AAPL")

        assert "ticker" in data, "Should include ticker"
        assert data["ticker"] == "AAPL", "Ticker should match"
        assert "info" in data, "Should include stock info"


class TestAgentTools:
    """Test agent tools"""

    def test_search_documents(self):
        """Test document search"""
        from backend.agent.tools import get_research_tools

        tools = get_research_tools()

        # This will only work if database has data
        # In a real test, we'd seed test data first
        results = tools.search_documents("AI technology", top_k=5)

        assert isinstance(results, list), "Should return list of results"

    def test_get_ai_stocks(self):
        """Test getting AI stocks list"""
        from backend.agent.tools import get_research_tools

        tools = get_research_tools()
        stocks = tools.get_ai_stocks_list()

        assert isinstance(stocks, list), "Should return list of stocks"


class TestAgentEvaluation:
    """Test agent evaluation system"""

    def test_relevance_score(self):
        """Test relevance score calculation"""
        evaluator = get_evaluator()

        question = "What is NVDA's recent performance?"
        answer = "NVDA has shown strong performance with 40% growth in AI chip sales."
        documents = [{"content": "NVDA AI chip sales grew 40% this quarter", "ticker": "NVDA"}]

        score = evaluator.calculate_relevance_score(question, answer, documents)

        assert 0 <= score <= 1, "Score should be between 0 and 1"
        assert score > 0.5, "Should have decent relevance score"

    def test_accuracy_score(self):
        """Test accuracy score calculation"""
        evaluator = get_evaluator()

        answer = "NVDA stock is trading at $495.32, up 3.2% this week with Q3 EPS of $4.02"
        documents = [{"content": "NVDA earnings report shows $4.02 EPS", "ticker": "NVDA"}]

        score = evaluator.calculate_accuracy_score(answer, documents)

        assert 0 <= score <= 1, "Score should be between 0 and 1"
        assert score > 0.3, "Should detect numerical data usage"

    def test_reward_calculation(self):
        """Test overall reward calculation"""
        evaluator = get_evaluator()

        question = "How is NVDA performing?"
        answer = "NVDA is performing well with $495.32 stock price, up 3.2% this week. Q3 2024 earnings showed EPS of $4.02."
        documents = [
            {"content": "NVDA stock price $495.32 as of Dec 2024", "ticker": "NVDA"},
            {"content": "NVDA Q3 2024 earnings: $4.02 EPS", "ticker": "NVDA"},
        ]

        scores = evaluator.calculate_reward(question, answer, documents)

        assert "reward_score" in scores, "Should include reward score"
        assert "relevance_score" in scores, "Should include relevance score"
        assert "accuracy_score" in scores, "Should include accuracy score"
        assert "completeness_score" in scores, "Should include completeness score"

        assert 0 <= scores["reward_score"] <= 1, "Reward should be between 0 and 1"


class TestCostMonitoring:
    """Test cost monitoring"""

    def test_cost_tracking(self):
        """Test cost tracking functionality"""
        from backend.utils.cost_monitor import get_cost_monitor

        monitor = get_cost_monitor()

        # Log a test cost
        cost = monitor.log_embedding_cost(
            provider="huggingface",
            model="test-model",
            num_texts=10,
            tokens=1000,
        )

        assert isinstance(cost, float), "Should return float cost"
        assert cost >= 0, "Cost should be non-negative"

    def test_budget_check(self):
        """Test budget checking"""
        from backend.utils.cost_monitor import get_cost_monitor

        monitor = get_cost_monitor()

        within_budget = monitor.check_budget()
        assert isinstance(within_budget, bool), "Should return boolean"


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
