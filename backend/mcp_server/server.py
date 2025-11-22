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


@app.post("/exposure/compute")
async def compute_exposure_matrices(
    etf: str = Query(..., description="ETF ticker"),
    lookback_days: int = Query(252, description="Lookback period in days"),
    n_factors: int = Query(5, description="Number of factors for PCA")
):
    """Compute exposure matrices for an ETF

    Args:
        etf: ETF ticker
        lookback_days: Number of days to look back
        n_factors: Number of factors for idiosyncratic analysis

    Returns:
        Computation results
    """
    try:
        from backend.exposure.holdings_fetcher import get_holdings_fetcher
        from backend.exposure.matrix_calculator import get_matrix_calculator
        from datetime import date, timedelta

        # Fetch holdings and prices
        fetcher = get_holdings_fetcher()
        tickers = fetcher.get_holdings_tickers(etf)

        if not tickers:
            raise HTTPException(status_code=404, detail=f"No holdings found for {etf}")

        # Fetch prices
        fetcher.fetch_and_save_etf_prices(etf, lookback_days)

        # Compute matrices
        calculator = get_matrix_calculator()
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)

        result = calculator.compute_and_save_matrices(
            etf, tickers, start_date, end_date, n_factors
        )

        return result

    except Exception as e:
        logger.error(f"Failed to compute exposure matrices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/exposure/matrix/{etf}")
async def get_exposure_matrix(
    etf: str,
    matrix_type: str = Query("correlation", description="Matrix type: correlation, covariance, idiosyncratic"),
    max_age_days: int = Query(7, description="Max age of cached matrix in days")
):
    """Get computed exposure matrix for an ETF

    Args:
        etf: ETF ticker
        matrix_type: Type of matrix
        max_age_days: Maximum age of cached data

    Returns:
        Matrix data
    """
    try:
        from backend.exposure.matrix_calculator import get_matrix_calculator

        calculator = get_matrix_calculator()
        matrix_data = calculator.get_cached_matrix(etf, matrix_type, max_age_days)

        if not matrix_data:
            raise HTTPException(
                status_code=404,
                detail=f"Matrix not found. Run /exposure/compute first."
            )

        return {
            "etf": etf,
            "matrix_type": matrix_type,
            "tickers": matrix_data['tickers'],
            "matrix": matrix_data['matrix'].tolist(),
            "params": matrix_data['params'],
            "date_range": {
                "start": matrix_data['start_date'].isoformat(),
                "end": matrix_data['end_date'].isoformat()
            },
            "created_at": matrix_data['created_at'].isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get exposure matrix: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/exposure/compare/{etf1}/{etf2}")
async def compare_etfs(
    etf1: str,
    etf2: str,
    matrix_type: str = Query("correlation", description="Matrix type to use for comparison"),
    max_age_days: int = Query(7, description="Max age of cached matrices")
):
    """Compare two ETFs using exposure matrices

    Args:
        etf1: First ETF
        etf2: Second ETF
        matrix_type: Matrix type to use
        max_age_days: Max age of cached data

    Returns:
        Comparison results with similarity score and comparison matrix
    """
    try:
        from backend.exposure.comparison_engine import get_comparison_engine

        engine = get_comparison_engine()
        result = engine.compare_etfs(etf1, etf2, matrix_type, max_age_days)

        if not result.get('success'):
            raise HTTPException(status_code=404, detail=result.get('error'))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to compare ETFs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/exposure/pairwise/{etf}/{ticker1}/{ticker2}")
async def get_pairwise_exposure(
    etf: str,
    ticker1: str,
    ticker2: str,
    max_age_days: int = Query(7, description="Max age of cached data")
):
    """Get pairwise exposure between two stocks

    Args:
        etf: ETF containing both stocks
        ticker1: First ticker
        ticker2: Second ticker
        max_age_days: Max age of cached data

    Returns:
        Pairwise exposure metrics
    """
    try:
        from backend.exposure.comparison_engine import get_comparison_engine

        engine = get_comparison_engine()
        result = engine.get_pairwise_comparison(etf, ticker1, ticker2, max_age_days)

        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Pairwise exposure not found for {ticker1}-{ticker2} in {etf}"
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get pairwise exposure: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/exposure/similar/{etf}/{ticker}")
async def get_similar_stocks(
    etf: str,
    ticker: str,
    metric: str = Query("correlation", description="Metric to use: correlation, idiosyncratic_score, etc."),
    top_n: int = Query(10, ge=1, le=50, description="Number of results")
):
    """Get stocks most similar to a given stock

    Args:
        etf: ETF
        ticker: Target ticker
        metric: Metric to rank by
        top_n: Number of results

    Returns:
        List of similar stocks with scores
    """
    try:
        from backend.exposure.comparison_engine import get_comparison_engine

        engine = get_comparison_engine()
        similar = engine.get_stock_similarity_ranking(etf, ticker, metric, top_n)

        return {
            "etf": etf,
            "ticker": ticker,
            "metric": metric,
            "similar_stocks": similar
        }

    except Exception as e:
        logger.error(f"Failed to get similar stocks: {e}")
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
