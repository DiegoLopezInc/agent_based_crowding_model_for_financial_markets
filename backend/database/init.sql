-- Initialize PostgreSQL database with pgvector extension
-- This script runs automatically when the Docker container starts

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create schema
CREATE SCHEMA IF NOT EXISTS etf_research;

-- Set search path
SET search_path TO etf_research, public;

-- Documents table: stores the original documents/chunks
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    source VARCHAR(255),
    ticker VARCHAR(10),
    etf VARCHAR(10),
    document_type VARCHAR(50), -- 'news', 'financial_report', 'sec_filing', etc.
    published_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Embeddings table: stores vector embeddings
CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    embedding vector(384), -- Default dimension for all-MiniLM-L6-v2
    model VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ETF holdings table: tracks which stocks belong to which ETFs
CREATE TABLE IF NOT EXISTS etf_holdings (
    id SERIAL PRIMARY KEY,
    etf VARCHAR(10) NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    weight FLOAT,
    sector VARCHAR(100),
    is_ai_related BOOLEAN DEFAULT FALSE,
    added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(etf, ticker)
);

-- Query history table: tracks user queries for analytics
CREATE TABLE IF NOT EXISTS query_history (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    results_count INTEGER,
    top_tickers TEXT[],
    agent_used BOOLEAN DEFAULT FALSE,
    response_time_ms INTEGER,
    cost_usd FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Agent evaluations table: stores evaluation metrics
CREATE TABLE IF NOT EXISTS agent_evaluations (
    id SERIAL PRIMARY KEY,
    query_id INTEGER REFERENCES query_history(id),
    relevance_score FLOAT,
    accuracy_score FLOAT,
    completeness_score FLOAT,
    reward_score FLOAT,
    feedback TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_documents_ticker ON documents(ticker);
CREATE INDEX IF NOT EXISTS idx_documents_etf ON documents(etf);
CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(document_type);
CREATE INDEX IF NOT EXISTS idx_documents_published ON documents(published_date DESC);
CREATE INDEX IF NOT EXISTS idx_documents_metadata ON documents USING gin(metadata);

CREATE INDEX IF NOT EXISTS idx_embeddings_document ON embeddings(document_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model);

-- Vector similarity index using HNSW for fast approximate nearest neighbor search
CREATE INDEX IF NOT EXISTS idx_embeddings_vector_hnsw ON embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Alternative: IVFFlat index (uncomment if preferred)
-- CREATE INDEX IF NOT EXISTS idx_embeddings_vector_ivfflat ON embeddings
-- USING ivfflat (embedding vector_cosine_ops)
-- WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_etf_holdings_etf ON etf_holdings(etf);
CREATE INDEX IF NOT EXISTS idx_etf_holdings_ticker ON etf_holdings(ticker);
CREATE INDEX IF NOT EXISTS idx_etf_holdings_ai ON etf_holdings(is_ai_related);

CREATE INDEX IF NOT EXISTS idx_query_history_created ON query_history(created_at DESC);

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to automatically update updated_at
CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Function for similarity search
CREATE OR REPLACE FUNCTION search_similar_documents(
    query_embedding vector(384),
    match_threshold float DEFAULT 0.7,
    match_count int DEFAULT 5,
    filter_etf varchar DEFAULT NULL,
    filter_ticker varchar DEFAULT NULL
)
RETURNS TABLE (
    id integer,
    content text,
    metadata jsonb,
    ticker varchar,
    etf varchar,
    similarity float
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        d.id,
        d.content,
        d.metadata,
        d.ticker,
        d.etf,
        1 - (e.embedding <=> query_embedding) AS similarity
    FROM embeddings e
    JOIN documents d ON e.document_id = d.id
    WHERE
        (filter_etf IS NULL OR d.etf = filter_etf)
        AND (filter_ticker IS NULL OR d.ticker = filter_ticker)
        AND (1 - (e.embedding <=> query_embedding)) >= match_threshold
    ORDER BY e.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- Insert initial QQQ AI stocks
INSERT INTO etf_holdings (etf, ticker, is_ai_related) VALUES
    ('QQQ', 'NVDA', true),
    ('QQQ', 'MSFT', true),
    ('QQQ', 'GOOGL', true),
    ('QQQ', 'META', true),
    ('QQQ', 'AAPL', true),
    ('QQQ', 'AMZN', true),
    ('QQQ', 'TSLA', true),
    ('QQQ', 'AMD', true),
    ('QQQ', 'AVGO', true),
    ('QQQ', 'QCOM', true),
    ('QQQ', 'ADBE', true),
    ('QQQ', 'CRM', true),
    ('QQQ', 'INTC', true),
    ('QQQ', 'ORCL', true),
    ('QQQ', 'CSCO', true)
ON CONFLICT (etf, ticker) DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA etf_research TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA etf_research TO postgres;
