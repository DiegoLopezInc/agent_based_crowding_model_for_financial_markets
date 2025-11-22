# Agentic RAG for ETF Stock Research

A comprehensive agentic Retrieval-Augmented Generation (RAG) system for researching AI-correlated stocks in ETFs, with a focus on the QQQ ETF. This system combines vector search with an intelligent agent to provide data-driven financial insights.

## 🎯 Features

### Backend
- **PostgreSQL Vector Database**: Powered by pgvector for semantic search
- **HuggingFace Embeddings**: Free, local embedding generation
- **Data Pipeline**: Automated ingestion of stock data, news, and earnings
- **MCP Server**: FastAPI-based server for database interaction
- **Agentic RAG**: LangChain-powered agent with specialized financial tools
- **Cost Monitoring**: Track and limit API costs for embeddings and agent calls
- **Quantitative Evaluation**: Automated agent performance evaluation with reward functions
- **Comprehensive Logging**: JSON-structured logging for all operations

### Data Sources
- Yahoo Finance (stock info, news, earnings)
- Financial news RSS feeds (Reuters, CNBC)
- Configurable to add more sources

### Agent Capabilities
- Semantic search across financial documents
- Stock-specific document retrieval
- ETF holdings analysis
- Multi-stock comparisons
- Data-driven insights with quantitative backing

## 📋 Prerequisites

- Python 3.9+
- Docker & Docker Compose (for PostgreSQL)
- API Keys:
  - Anthropic API key (for Claude agent)
  - Optional: OpenAI API key, Yahoo Finance API key

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd agent_based_crowding_model_for_financial_markets

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API keys
nano .env  # or use your preferred editor
```

Required in `.env`:
```bash
POSTGRES_PASSWORD=your_secure_password
ANTHROPIC_API_KEY=your_anthropic_key
```

### 3. Start PostgreSQL Database

```bash
# Start PostgreSQL with pgvector
docker-compose up -d postgres

# Verify database is running
docker-compose ps

# Initialize database (creates tables and schema)
python -c "from backend.database.connection import init_database; init_database()"
```

### 4. Configure Settings

Edit `config.yaml` to customize:
- Embedding model
- Agent model
- ETF list and AI stocks
- Cost limits
- Data sources

### 5. Ingest Data

```bash
# Ingest all AI stocks from QQQ
python scripts/run_ingestion.py --all-ai-stocks

# Or ingest a specific stock
python scripts/run_ingestion.py --ticker NVDA

# Ingest market news
python scripts/run_ingestion.py --market-news
```

### 6. Start MCP Server

```bash
# In one terminal, start the MCP server
python backend/mcp_server/server.py

# Or use uvicorn directly
uvicorn backend.mcp_server.server:app --host 0.0.0.0 --port 8000
```

### 7. Query the Agent

```bash
# Simple RAG query (faster, cheaper)
python scripts/query_agent.py "What is NVDA's recent performance?" --mode rag

# Full agentic query (uses tools)
python scripts/query_agent.py "Compare NVDA and AMD in AI capabilities" --mode agent

# With evaluation
python scripts/query_agent.py "What AI stocks should I research?" --mode rag --evaluate
```

## 📊 Architecture

```
┌─────────────────────────────────────────────────────┐
│                   User Interface                     │
│            (CLI / API / Future Frontend)             │
└───────────────────┬─────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────┐
│                  Agent Layer                         │
│  ┌──────────────────────────────────────────────┐   │
│  │  LangChain Agent (Claude Sonnet 4.5)        │   │
│  │  - Tool calling                              │   │
│  │  - Multi-step reasoning                      │   │
│  │  - Cost monitoring                           │   │
│  └──────────────────────────────────────────────┘   │
└───────────────────┬─────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────┐
│                  MCP Server                          │
│  ┌──────────────────────────────────────────────┐   │
│  │  FastAPI Endpoints                           │   │
│  │  - /search (semantic search)                 │   │
│  │  - /ticker/{ticker}/documents                │   │
│  │  - /etf/{etf}/holdings                       │   │
│  │  - /cost (cost tracking)                     │   │
│  └──────────────────────────────────────────────┘   │
└───────────────────┬─────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────┐
│              Vector Database Layer                   │
│  ┌──────────────────────────────────────────────┐   │
│  │  PostgreSQL + pgvector                       │   │
│  │  - Documents & Embeddings                    │   │
│  │  - ETF Holdings                              │   │
│  │  - Query History & Evaluations               │   │
│  │  - HNSW Index for fast similarity search    │   │
│  └──────────────────────────────────────────────┘   │
└───────────────────┬─────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────┐
│               Data Pipeline Layer                    │
│  ┌──────────────────────────────────────────────┐   │
│  │  Data Sources                                │   │
│  │  - Yahoo Finance                             │   │
│  │  - Financial News RSS                        │   │
│  │                                              │   │
│  │  Processing                                  │   │
│  │  - Text chunking                             │   │
│  │  - Embedding generation (HuggingFace)        │   │
│  │  - Cost tracking                             │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

## 🗂️ Project Structure

```
.
├── backend/
│   ├── agent/              # Agent implementation
│   │   ├── agent.py       # Main agent class
│   │   ├── tools.py       # Agent tools
│   │   ├── prompts.py     # Optimized prompts
│   │   └── evaluation.py  # Reward function & evaluation
│   ├── database/          # Database layer
│   │   ├── connection.py  # Connection manager
│   │   ├── models.py      # SQLAlchemy models
│   │   ├── repository.py  # Data access layer
│   │   └── init.sql       # Database initialization
│   ├── embeddings/        # Embedding models
│   │   └── embedding_model.py
│   ├── data_pipeline/     # Data ingestion
│   │   ├── data_sources.py  # Yahoo Finance, RSS feeds
│   │   ├── chunker.py       # Text chunking
│   │   └── ingestion.py     # Main pipeline
│   ├── mcp_server/        # MCP API server
│   │   └── server.py
│   └── utils/             # Utilities
│       ├── config.py      # Configuration management
│       ├── logger.py      # Logging
│       └── cost_monitor.py # Cost tracking
├── scripts/               # Utility scripts
│   ├── run_ingestion.py  # Data ingestion script
│   └── query_agent.py    # Query script
├── tests/                 # Test suite
│   └── test_pipeline.py  # Integration tests
├── config.yaml           # Main configuration
├── docker-compose.yml    # Docker services
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## ⚙️ Configuration

### Agent Configuration

In `config.yaml`:

```yaml
agent:
  provider: "anthropic"  # or "openai"
  model_name: "claude-sonnet-4-5-20250929"
  temperature: 0.1
  max_tokens: 4096
```

### Embedding Configuration

```yaml
embedding:
  provider: "huggingface"
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  embedding_dimension: 384
  batch_size: 32
```

Options:
- `all-MiniLM-L6-v2` - Fast, good quality (384 dims)
- `BAAI/bge-large-en-v1.5` - Better quality, slower (1024 dims)

### ETF Configuration

```yaml
etfs:
  primary: "QQQ"
  focus_sector: "AI"
  qqq_ai_stocks:
    - "NVDA"
    - "MSFT"
    - "GOOGL"
    # ... add more
```

### Cost Limits

```yaml
cost_monitoring:
  enabled: true
  max_daily_cost: 10.0
  max_embedding_cost: 5.0
  max_agent_cost: 5.0
  alert_threshold: 0.8
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_pipeline.py::TestDatabaseConnection

# Run with coverage
pytest --cov=backend tests/

# Run only fast tests (skip integration)
pytest -m "not slow" tests/
```

## 📈 Cost Monitoring

The system tracks costs for:
- Embedding generation (free for local HuggingFace models)
- Agent API calls (Claude/OpenAI)

View costs:
```bash
# Via API
curl http://localhost:8000/cost

# Via database stats
python -c "from backend.utils.cost_monitor import get_cost_monitor; print(get_cost_monitor().get_cost_summary())"
```

## 🎯 Evaluation & Reward Function

The system includes automated evaluation with quantitative metrics:

### Metrics
1. **Relevance** (0-1): How relevant is the answer to the question?
2. **Accuracy** (0-1): Factual correctness and data usage
3. **Completeness** (0-1): Thoroughness of the answer
4. **Data Usage** (0-1): Use of quantitative financial data

### Reward Function

```
Reward = (Relevance × 0.5) + (Accuracy × 0.3) + (Completeness × 0.2)
```

Weights are configurable in `config.yaml`.

### View Evaluation Results

```python
from backend.agent.evaluation import get_evaluator

evaluator = get_evaluator()
avg_scores = evaluator.get_average_scores()
print(avg_scores)
```

## 🔧 API Endpoints

### MCP Server API

- `GET /health` - Health check
- `GET /stats` - Database statistics
- `POST /search` - Semantic search
  ```json
  {
    "query": "What is NVDA's recent performance?",
    "top_k": 5,
    "similarity_threshold": 0.7,
    "ticker_filter": "NVDA"
  }
  ```
- `GET /etf/{etf}/holdings` - Get ETF holdings
- `GET /ticker/{ticker}/documents` - Get stock documents
- `GET /ai-stocks` - List AI stocks
- `GET /cost` - Cost summary

API docs available at: `http://localhost:8000/docs`

## 🎨 Frontend Integration

The backend is ready for frontend integration via the MCP Server API. A future enhancement will include a modified version of searchthearxiv with:
- PostgreSQL instead of Pinecone
- Agentic query interface
- Real-time cost tracking
- Evaluation feedback

## 📝 Usage Examples

### Example 1: Simple Query

```bash
python scripts/query_agent.py "What is NVDA's latest earnings report?" --mode rag
```

### Example 2: Comparison

```bash
python scripts/query_agent.py "Compare NVDA, AMD, and INTC for AI capabilities" --mode agent
```

### Example 3: ETF Analysis

```bash
python scripts/query_agent.py "What are the top AI stocks in QQQ?" --mode rag
```

### Example 4: Programmatic Use

```python
from backend.agent.agent import get_agent

agent = get_agent()
response = agent.simple_rag_query("What is MSFT's AI strategy?")
print(response["answer"])
```

## 🚦 Validation with Tinker

For validation using Tinker (budget: $2):

1. Create test queries in a notebook
2. Run queries through the system
3. Collect evaluation metrics
4. Analyze performance vs. cost

```python
from backend.agent.agent import get_agent
from backend.agent.evaluation import get_evaluator

# Test queries
queries = [
    "What is NVDA's market position in AI?",
    "Compare MSFT and GOOGL cloud AI offerings",
    # ... more queries
]

agent = get_agent()
evaluator = get_evaluator()

results = []
for query in queries:
    response = agent.simple_rag_query(query)
    scores = evaluator.calculate_reward(
        query,
        response["answer"],
        response.get("documents", [])
    )
    results.append({
        "query": query,
        "answer": response["answer"],
        "scores": scores,
        "cost": response["metadata"]["cost_usd"]
    })

# Analyze results
total_cost = sum(r["cost"] for r in results)
avg_reward = sum(r["scores"]["reward_score"] for r in results) / len(results)

print(f"Total cost: ${total_cost:.2f}")
print(f"Average reward: {avg_reward:.3f}")
```

## 🔒 Security Notes

- Never commit `.env` file
- Use strong PostgreSQL passwords
- Restrict MCP server access in production
- Set appropriate CORS policies
- Monitor API usage and costs

## 🐛 Troubleshooting

### Database Connection Issues
```bash
# Check if PostgreSQL is running
docker-compose ps

# View logs
docker-compose logs postgres

# Restart database
docker-compose restart postgres
```

### Embedding Model Issues
```bash
# Clear cache and reload
rm -rf ~/.cache/huggingface/
python -c "from backend.embeddings.embedding_model import get_embedding_model; get_embedding_model()"
```

### Cost Exceeded
```bash
# Check current costs
python -c "from backend.utils.cost_monitor import get_cost_monitor; print(get_cost_monitor().get_daily_costs())"

# Reset (new day) or adjust limits in config.yaml
```

## 📚 Adding New ETFs

1. Update `config.yaml`:
```yaml
etfs:
  additional_etfs:
    - "SPY"
    - "ARKK"
```

2. Add holdings to database:
```python
from backend.database.connection import get_db_manager
from backend.database.repository import ETFHoldingRepository

with get_db_manager().get_session() as session:
    ETFHoldingRepository.add_holding(
        session=session,
        etf="SPY",
        ticker="AAPL",
        is_ai_related=True
    )
```

3. Ingest data:
```bash
python scripts/run_ingestion.py --ticker AAPL
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure tests pass
5. Submit a pull request

## 📄 License

MIT License

## 🙏 Acknowledgments

- PostgreSQL pgvector team
- HuggingFace for open-source models
- Anthropic for Claude
- LangChain framework
- Yahoo Finance for data access

## 📧 Support

For issues and questions:
- Open a GitHub issue
- Check logs in `logs/` directory
- Review cost tracking in database

---

**Built with ❤️ for quantitative financial research**
