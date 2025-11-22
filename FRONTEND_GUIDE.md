# Frontend Integration Guide

This guide explains how to integrate a frontend with the ETF Research RAG backend.

## Backend API

The MCP Server provides a REST API at `http://localhost:8000` with the following endpoints:

### Core Endpoints

#### Health Check
```bash
GET /health

Response:
{
  "status": "healthy",
  "database_connected": true,
  "embedding_model_loaded": true,
  "timestamp": "2024-12-15T10:30:00"
}
```

#### Semantic Search
```bash
POST /search

Request:
{
  "query": "What is NVDA's performance?",
  "top_k": 5,
  "similarity_threshold": 0.7,
  "ticker_filter": null,
  "etf_filter": null
}

Response:
{
  "query": "What is NVDA's performance?",
  "results": [
    {
      "id": 123,
      "content": "NVDA reported strong Q3 earnings...",
      "metadata": {...},
      "ticker": "NVDA",
      "etf": "QQQ",
      "similarity": 0.89
    }
  ],
  "count": 5,
  "query_embedding_time_ms": 45.2,
  "search_time_ms": 12.8
}
```

#### Get Stock Documents
```bash
GET /ticker/{ticker}/documents?limit=10

Example: GET /ticker/NVDA/documents?limit=5

Response:
{
  "ticker": "NVDA",
  "count": 5,
  "documents": [...]
}
```

#### Get ETF Holdings
```bash
GET /etf/{etf}/holdings?ai_only=true

Example: GET /etf/QQQ/holdings?ai_only=true

Response:
{
  "etf": "QQQ",
  "count": 15,
  "holdings": [
    {
      "ticker": "NVDA",
      "weight": 4.5,
      "sector": "Technology",
      "is_ai_related": true
    }
  ]
}
```

#### Get AI Stocks
```bash
GET /ai-stocks

Response:
{
  "count": 15,
  "stocks": ["NVDA", "MSFT", "GOOGL", ...]
}
```

#### Database Stats
```bash
GET /stats

Response:
{
  "documents": 1250,
  "embeddings": 1250,
  "etf_holdings": 15,
  "queries": 45,
  "unique_tickers": 15,
  "unique_etfs": 1
}
```

#### Cost Tracking
```bash
GET /cost?days=7

Response:
{
  "period_days": 7,
  "total_cost": 2.45,
  "daily_breakdown": [...],
  "today": {
    "embedding": 0.0,
    "agent": 0.15,
    "total": 0.15
  },
  "budget": {
    "max_daily": 10.0,
    "max_embedding": 5.0,
    "max_agent": 5.0
  }
}
```

## Frontend Architecture Recommendations

### Option 1: Modify searchthearxiv (Recommended)

The original project uses Pinecone. To adapt it:

1. **Clone searchthearxiv**
   ```bash
   cd frontend
   git clone https://github.com/augustwester/searchthearxiv.git
   cd searchthearxiv
   ```

2. **Replace Pinecone calls with PostgreSQL API calls**

   Original (Pinecone):
   ```typescript
   const results = await pinecone.query({
     vector: embedding,
     topK: 5
   });
   ```

   New (PostgreSQL):
   ```typescript
   const results = await fetch('http://localhost:8000/search', {
     method: 'POST',
     headers: { 'Content-Type': 'application/json' },
     body: JSON.stringify({
       query: searchQuery,
       top_k: 5,
       similarity_threshold: 0.7
     })
   });
   ```

3. **Add Agentic Features**

   Add buttons for:
   - Simple search (RAG)
   - Agentic search (with tools)
   - Stock comparisons
   - ETF analysis

4. **Update UI Components**

   - Add cost tracking display
   - Add evaluation scores display
   - Add ticker/ETF filters
   - Add document type filters

### Option 2: Build from Scratch with React

```typescript
// Example React component
import React, { useState } from 'react';

function ETFSearch() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState('rag'); // 'rag' or 'agent'

  const search = async () => {
    setLoading(true);

    if (mode === 'rag') {
      // Direct search via MCP server
      const response = await fetch('http://localhost:8000/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query,
          top_k: 5,
          similarity_threshold: 0.7
        })
      });

      const data = await response.json();
      setResults(data.results);
    } else {
      // Agent mode - you would need to add an agent endpoint
      // Or use the Python script and expose via API
    }

    setLoading(false);
  };

  return (
    <div className="etf-search">
      <div className="mode-selector">
        <button onClick={() => setMode('rag')}
                className={mode === 'rag' ? 'active' : ''}>
          Fast Search
        </button>
        <button onClick={() => setMode('agent')}
                className={mode === 'agent' ? 'active' : ''}>
          Agentic Search
        </button>
      </div>

      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Ask about AI stocks..."
      />

      <button onClick={search} disabled={loading}>
        {loading ? 'Searching...' : 'Search'}
      </button>

      <div className="results">
        {results.map((result, idx) => (
          <div key={idx} className="result-card">
            <div className="ticker-badge">{result.ticker}</div>
            <div className="similarity">
              Similarity: {(result.similarity * 100).toFixed(1)}%
            </div>
            <p>{result.content}</p>
            <div className="metadata">
              {result.metadata.published_date && (
                <span>📅 {result.metadata.published_date}</span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default ETFSearch;
```

### Option 3: Streamlit (Quick Prototype)

Create `frontend/app.py`:

```python
import streamlit as st
import requests
import json

st.title("🤖 ETF Research Assistant")

# API base URL
API_URL = "http://localhost:8000"

# Mode selection
mode = st.radio("Search Mode", ["Simple Search", "Agentic Search"])

# Query input
query = st.text_input("Ask about AI stocks:",
                      placeholder="What is NVDA's recent performance?")

# Filters
col1, col2 = st.columns(2)
with col1:
    ticker_filter = st.selectbox("Ticker Filter",
                                 ["All"] + get_ai_stocks())
with col2:
    top_k = st.slider("Results", 1, 20, 5)

if st.button("Search"):
    if mode == "Simple Search":
        # Call MCP server
        response = requests.post(
            f"{API_URL}/search",
            json={
                "query": query,
                "top_k": top_k,
                "ticker_filter": None if ticker_filter == "All" else ticker_filter,
                "similarity_threshold": 0.7
            }
        )

        data = response.json()

        # Display results
        st.write(f"Found {data['count']} results")

        for result in data['results']:
            with st.expander(f"{result['ticker']} - {result['similarity']:.2%} match"):
                st.write(result['content'])
                st.json(result['metadata'])

    else:  # Agentic search
        # Call agent endpoint (would need to add this)
        st.info("Agentic search coming soon!")

# Sidebar: Stats and costs
with st.sidebar:
    st.header("System Status")

    # Health
    health = requests.get(f"{API_URL}/health").json()
    st.metric("Database",
              "✅ Connected" if health['database_connected'] else "❌ Disconnected")

    # Stats
    stats = requests.get(f"{API_URL}/stats").json()
    st.metric("Documents", stats['documents'])
    st.metric("Unique Stocks", stats['unique_tickers'])

    # Costs
    costs = requests.get(f"{API_URL}/cost").json()
    st.metric("Today's Cost", f"${costs['today']['total']:.2f}")
    st.progress(costs['today']['total'] / costs['budget']['max_daily'])

def get_ai_stocks():
    response = requests.get(f"{API_URL}/ai-stocks")
    return response.json()['stocks']
```

Run with:
```bash
streamlit run frontend/app.py
```

## UI Components Needed

### 1. Search Interface
- Query input box
- Mode selector (Simple/Agentic)
- Filters (ticker, ETF, date range)
- Search button

### 2. Results Display
- Result cards with:
  - Ticker badge
  - Similarity score
  - Content excerpt
  - Metadata (date, source, type)
  - "View full" button

### 3. Cost Dashboard
- Real-time cost tracking
- Budget progress bar
- Daily/weekly cost chart
- Cost breakdown by service

### 4. Stock Explorer
- List of available stocks
- ETF breakdown
- Stock cards with quick facts

### 5. Evaluation Display (Optional)
- Show evaluation scores
- Historical performance graphs
- Feedback submission

## Styling Recommendations

Use a financial/professional theme:
- Colors: Blues, grays, accent green/red for gains/losses
- Fonts: Clean sans-serif (Inter, Roboto)
- Cards: Subtle shadows, clean borders
- Charts: Use recharts or Chart.js
- Icons: Use react-icons or heroicons

## State Management

For React, consider:
- **Simple**: React useState/useContext
- **Medium**: Zustand
- **Complex**: Redux Toolkit

## Next Steps

1. Choose your frontend framework
2. Set up API client/SDK
3. Implement core search interface
4. Add filtering and modes
5. Implement cost tracking display
6. Add evaluation feedback
7. Style and polish

## Example API Client

```typescript
// api/etfResearch.ts
export class ETFResearchAPI {
  private baseUrl: string;

  constructor(baseUrl: string = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
  }

  async search(query: string, options?: SearchOptions) {
    const response = await fetch(`${this.baseUrl}/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        top_k: options?.topK || 5,
        similarity_threshold: options?.threshold || 0.7,
        ticker_filter: options?.ticker,
        etf_filter: options?.etf,
      })
    });

    return response.json();
  }

  async getStockDocuments(ticker: string, limit: number = 10) {
    const response = await fetch(
      `${this.baseUrl}/ticker/${ticker}/documents?limit=${limit}`
    );
    return response.json();
  }

  async getETFHoldings(etf: string, aiOnly: boolean = false) {
    const response = await fetch(
      `${this.baseUrl}/etf/${etf}/holdings?ai_only=${aiOnly}`
    );
    return response.json();
  }

  async getStats() {
    const response = await fetch(`${this.baseUrl}/stats`);
    return response.json();
  }

  async getCosts(days: number = 7) {
    const response = await fetch(`${this.baseUrl}/cost?days=${days}`);
    return response.json();
  }
}
```

---

**The backend is ready - choose your frontend and start building! 🚀**
