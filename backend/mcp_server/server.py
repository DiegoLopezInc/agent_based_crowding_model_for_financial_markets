"""MCP Server implementation for ETF Research database interaction"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.database.connection import get_db_manager, init_database
from backend.database.repository import (
    DocumentRepository,
    EmbeddingRepository,
    ETFHoldingRepository,
    QueryHistoryRepository,
)
from backend.embeddings.embedding_model import get_embedding_model
from backend.utils.config import get_config
from backend.utils.logger import setup_logger
from backend.utils.cost_monitor import get_cost_monitor

logger = setup_logger("mcp_server")

# Initialize FastAPI app
app = FastAPI(
    title="ETF Research RAG MCP Server",
    description="MCP Server for ETF stock research with agentic RAG capabilities",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class SearchRequest(BaseModel):
    """Search request model"""
    query: str
    top_k: int = 5
    similarity_threshold: float = 0.7
    etf_filter: Optional[str] = None
    ticker_filter: Optional[str] = None


class SearchResult(BaseModel):
    """Search result model"""
    id: int
    content: str
    metadata: Dict[str, Any]
    ticker: Optional[str]
    etf: Optional[str]
    similarity: float


class SearchResponse(BaseModel):
    """Search response model"""
    query: str
    results: List[SearchResult]
    count: int
    query_embedding_time_ms: float
    search_time_ms: float


class StatsResponse(BaseModel):
    """Database statistics response"""
    documents: int
    embeddings: int
    etf_holdings: int
    queries: int
    unique_tickers: int
    unique_etfs: int


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    database_connected: bool
    embedding_model_loaded: bool
    timestamp: str


# Initialize components
db_manager = None
embedding_model = None
cost_monitor = None


@app.on_event("startup")
async def startup_event():
    """Initialize server components on startup"""
    global db_manager, embedding_model, cost_monitor

    logger.info("Starting MCP Server...")

    # Initialize database
    try:
        db_manager = init_database()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

    # Initialize embedding model
    try:
        embedding_model = get_embedding_model()
        logger.info("Embedding model loaded")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise

    # Initialize cost monitor
    cost_monitor = get_cost_monitor()

    logger.info("MCP Server started successfully")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    db_connected = db_manager is not None and db_manager.check_connection()
    embedding_loaded = embedding_model is not None

    return HealthResponse(
        status="healthy" if (db_connected and embedding_loaded) else "degraded",
        database_connected=db_connected,
        embedding_model_loaded=embedding_loaded,
        timestamp=datetime.now().isoformat(),
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get database statistics"""
    try:
        stats = db_manager.get_stats()
        return StatsResponse(**stats)
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Semantic search endpoint

    Args:
        request: Search request with query and parameters

    Returns:
        Search results with similarity scores
    """
    try:
        # Time the embedding generation
        start_time = datetime.now()
        query_embedding = embedding_model.embed_query(request.query)
        embedding_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Time the database search
        start_time = datetime.now()
        with db_manager.get_session() as session:
            results = EmbeddingRepository.search_similar(
                session=session,
                query_embedding=query_embedding,
                limit=request.top_k,
                similarity_threshold=request.similarity_threshold,
                etf_filter=request.etf_filter,
                ticker_filter=request.ticker_filter,
            )

            # Log query
            top_tickers = list(set([r["ticker"] for r in results if r["ticker"]]))
            QueryHistoryRepository.create_query(
                session=session,
                query=request.query,
                results_count=len(results),
                top_tickers=top_tickers,
                agent_used=False,
                response_time_ms=int((datetime.now() - start_time).total_seconds() * 1000),
            )

        search_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Convert to response model
        search_results = [SearchResult(**r) for r in results]

        logger.info(
            f"Search completed: '{request.query}' returned {len(results)} results "
            f"(embedding: {embedding_time_ms:.2f}ms, search: {search_time_ms:.2f}ms)"
        )

        return SearchResponse(
            query=request.query,
            results=search_results,
            count=len(search_results),
            query_embedding_time_ms=embedding_time_ms,
            search_time_ms=search_time_ms,
        )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/etf/{etf}/holdings")
async def get_etf_holdings(
    etf: str,
    ai_only: bool = Query(False, description="Only AI-related stocks")
):
    """Get holdings for an ETF

    Args:
        etf: ETF ticker symbol
        ai_only: Filter for AI-related stocks only

    Returns:
        List of holdings
    """
    try:
        with db_manager.get_session() as session:
            holdings = ETFHoldingRepository.get_holdings_by_etf(
                session=session,
                etf=etf.upper(),
                ai_only=ai_only,
            )

            return {
                "etf": etf.upper(),
                "count": len(holdings),
                "holdings": [
                    {
                        "ticker": h.ticker,
                        "weight": h.weight,
                        "sector": h.sector,
                        "is_ai_related": h.is_ai_related,
                    }
                    for h in holdings
                ],
            }

    except Exception as e:
        logger.error(f"Failed to get holdings for {etf}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ticker/{ticker}/documents")
async def get_ticker_documents(
    ticker: str,
    limit: int = Query(10, ge=1, le=100)
):
    """Get documents for a specific ticker

    Args:
        ticker: Stock ticker symbol
        limit: Maximum number of documents

    Returns:
        List of documents
    """
    try:
        with db_manager.get_session() as session:
            documents = DocumentRepository.get_documents_by_ticker(
                session=session,
                ticker=ticker.upper(),
                limit=limit,
            )

            return {
                "ticker": ticker.upper(),
                "count": len(documents),
                "documents": [
                    {
                        "id": d.id,
                        "content": d.content[:500] + "..." if len(d.content) > 500 else d.content,
                        "document_type": d.document_type,
                        "source": d.source,
                        "published_date": d.published_date.isoformat() if d.published_date else None,
                        "metadata": d.metadata,
                    }
                    for d in documents
                ],
            }

    except Exception as e:
        logger.error(f"Failed to get documents for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ai-stocks")
async def get_ai_stocks():
    """Get list of AI-related stocks

    Returns:
        List of AI stock tickers
    """
    try:
        with db_manager.get_session() as session:
            stocks = ETFHoldingRepository.get_ai_stocks(session)

            return {
                "count": len(stocks),
                "stocks": stocks,
            }

    except Exception as e:
        logger.error(f"Failed to get AI stocks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cost")
async def get_cost_summary(days: int = Query(7, ge=1, le=90)):
    """Get cost summary

    Args:
        days: Number of days to summarize

    Returns:
        Cost summary
    """
    try:
        summary = cost_monitor.get_cost_summary(days=days)
        daily_costs = cost_monitor.get_daily_costs()

        return {
            "period_days": days,
            "total_cost": summary["total"],
            "daily_breakdown": summary["daily"],
            "today": daily_costs,
            "budget": {
                "max_daily": cost_monitor.cost_config.max_daily_cost,
                "max_embedding": cost_monitor.cost_config.max_embedding_cost,
                "max_agent": cost_monitor.cost_config.max_agent_cost,
            },
        }

    except Exception as e:
        logger.error(f"Failed to get cost summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    config = get_config()

    uvicorn.run(
        "backend.mcp_server.server:app",
        host=config.mcp_server.host,
        port=config.mcp_server.port,
        workers=config.mcp_server.workers,
        reload=config.mcp_server.reload,
        log_level=config.mcp_server.log_level,
    )
