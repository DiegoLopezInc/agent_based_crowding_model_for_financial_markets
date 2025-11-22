# Quick Start Guide

Get up and running in 5 minutes!

## Prerequisites Check

```bash
# Check Python version (needs 3.9+)
python --version

# Check Docker
docker --version
docker-compose --version
```

## 1. Initial Setup (2 minutes)

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

Edit `.env` and add your Anthropic API key:
```bash
ANTHROPIC_API_KEY=sk-ant-...
POSTGRES_PASSWORD=mypassword123
```

## 2. Start Database (1 minute)

```bash
# Start PostgreSQL
docker-compose up -d postgres

# Wait 10 seconds for PostgreSQL to start
sleep 10

# Initialize database
python -c "from backend.database.connection import init_database; init_database()"
```

## 3. Ingest Sample Data (1-2 minutes)

```bash
# Ingest NVDA as a test
python scripts/run_ingestion.py --ticker NVDA

# This will:
# - Fetch NVDA stock info from Yahoo Finance
# - Get recent news and earnings
# - Generate embeddings
# - Store in PostgreSQL
```

## 4. Test Query (30 seconds)

```bash
# Ask a question
python scripts/query_agent.py "What is NVDA's recent performance?" --mode rag
```

You should get a response based on the ingested data!

## 5. Optional: Start MCP Server

```bash
# In a separate terminal
python backend/mcp_server/server.py

# Then visit http://localhost:8000/docs for API documentation
```

## Next Steps

### Ingest More Data

```bash
# Ingest all AI stocks (takes 10-20 minutes)
python scripts/run_ingestion.py --all-ai-stocks

# Ingest market news
python scripts/run_ingestion.py --market-news
```

### Query Examples

```bash
# Simple questions
python scripts/query_agent.py "What AI stocks are in the database?" --mode rag

# Comparisons
python scripts/query_agent.py "Compare NVDA and AMD" --mode agent

# With evaluation
python scripts/query_agent.py "What is MSFT's AI strategy?" --mode rag --evaluate
```

### Check Database Stats

```bash
python -c "from backend.database.connection import get_db_manager; print(get_db_manager().get_stats())"
```

### Monitor Costs

```bash
python -c "from backend.utils.cost_monitor import get_cost_monitor; print(get_cost_monitor().get_daily_costs())"
```

## Troubleshooting

### Database won't start
```bash
docker-compose logs postgres
# Look for errors

# Try restarting
docker-compose restart postgres
```

### Import errors
```bash
# Make sure you're in the project root
pwd

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### No data returned
```bash
# Check if data was ingested
python -c "from backend.database.connection import get_db_manager; print(get_db_manager().get_stats())"

# If documents=0, run ingestion again
python scripts/run_ingestion.py --ticker NVDA
```

## Configuration

Edit `config.yaml` to change:
- Agent model (default: Claude Sonnet 4.5)
- Embedding model (default: all-MiniLM-L6-v2)
- Cost limits (default: $10/day)
- ETF list and AI stocks

## Common Commands

```bash
# Ingest specific ticker
python scripts/run_ingestion.py --ticker <TICKER>

# Query with RAG (fast, cheap)
python scripts/query_agent.py "<question>" --mode rag

# Query with agent (slower, uses tools)
python scripts/query_agent.py "<question>" --mode agent

# Start MCP server
python backend/mcp_server/server.py

# Run tests
pytest tests/

# Check database stats
python -c "from backend.database.connection import get_db_manager; print(get_db_manager().get_stats())"

# View costs
curl http://localhost:8000/cost
```

## Development Workflow

1. **Data Ingestion** (once daily or as needed)
   ```bash
   python scripts/run_ingestion.py --all-ai-stocks
   python scripts/run_ingestion.py --market-news
   ```

2. **Start MCP Server** (keep running)
   ```bash
   python backend/mcp_server/server.py
   ```

3. **Query** (as needed)
   ```bash
   python scripts/query_agent.py "your question" --mode rag
   ```

4. **Monitor Costs**
   ```bash
   curl http://localhost:8000/cost
   ```

## Ready for Production?

1. Set strong passwords in `.env`
2. Configure CORS in `backend/mcp_server/server.py`
3. Set up SSL/TLS for MCP server
4. Use environment-specific configs
5. Set up monitoring and alerting
6. Configure database backups

---

**You're all set! 🚀**

Start querying your ETF research agent and explore the financial data.
