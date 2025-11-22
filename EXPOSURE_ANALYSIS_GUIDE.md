# ETF Exposure Analysis Guide

This guide explains how to use the exposure analysis system to compare idiosyncratic exposures across ETFs using linear algebra.

## 🎯 Overview

The exposure analysis system allows you to:
1. **Compute exposure matrices** for ETFs using correlation, covariance, and factor decomposition
2. **Compare ETFs** to find similarities and differences in their holdings
3. **Analyze pairwise exposures** between individual stocks
4. **Find similar/dissimilar stocks** based on various metrics
5. **Handle different sized ETFs** using matrix algebra techniques
6. **Cache results** for incremental updates and fast retrieval

### Key Concepts

- **Correlation Matrix**: How stocks move together (standardized)
- **Covariance Matrix**: Joint variability of stock returns
- **Idiosyncratic Matrix**: Stock-specific movements after removing common market factors
- **Factor Decomposition**: Using PCA to extract common factors and isolate stock-specific risks

## 📋 Prerequisites

1. **Price data** for all stocks in the ETF
2. **Holdings data** (automatic from configuration or fetched from Yahoo Finance)
3. **PostgreSQL database** with exposure tables

## 🚀 Quick Start

### Step 1: Compute Exposure Matrices

```bash
# Compute matrices for QQQ
python scripts/analyze_exposures.py compute QQQ

# With custom parameters
python scripts/analyze_exposures.py compute QQQ --lookback 365 --factors 10
```

This will:
1. Fetch holdings for QQQ
2. Download price history (default: 252 trading days = 1 year)
3. Calculate correlation, covariance, and idiosyncratic matrices
4. Decompose using PCA (default: 5 factors)
5. Save all pairwise exposures to database

### Step 2: Compare ETFs

```bash
# Compare QQQ vs VYM
python scripts/analyze_exposures.py compare QQQ VYM

# Using different matrix type
python scripts/analyze_exposures.py compare QQQ VYM --matrix-type idiosyncratic
```

Output includes:
- **Similarity score** (0-1, higher = more similar)
- **Common stocks** vs unique to each ETF
- **Matrix comparison statistics**

### Step 3: Analyze Specific Stocks

```bash
# Get pairwise exposure between NVDA and AMD
python scripts/analyze_exposures.py pairwise QQQ NVDA AMD

# Find stocks similar to NVDA
python scripts/analyze_exposures.py similar QQQ NVDA --top 10

# Find stocks with low idiosyncratic correlation (most different)
python scripts/analyze_exposures.py similar QQQ NVDA --metric idiosyncratic_score --top 5
```

## 📊 Understanding the Matrices

### 1. Correlation Matrix

Shows how stocks move together on a standardized scale (-1 to +1).

```
         NVDA    AMD     MSFT    GOOGL
NVDA     1.00    0.75    0.62    0.58
AMD      0.75    1.00    0.54    0.51
MSFT     0.62    0.54    1.00    0.71
GOOGL    0.58    0.51    0.71    1.00
```

- **1.00**: Perfect positive correlation (stock with itself)
- **0.75**: Strong positive correlation (NVDA-AMD move together)
- **0.62**: Moderate positive correlation

### 2. Covariance Matrix

Similar to correlation but not standardized - shows actual joint variability.

```
         NVDA      AMD       MSFT      GOOGL
NVDA     0.0023    0.0015    0.0008    0.0007
AMD      0.0015    0.0019    0.0006    0.0005
MSFT     0.0008    0.0006    0.0012    0.0009
GOOGL    0.0007    0.0005    0.0009    0.0011
```

Higher values = more volatile together.

### 3. Idiosyncratic Matrix

Correlations after removing common market factors (via PCA).

```
         NVDA    AMD     MSFT    GOOGL
NVDA     1.00    0.45    0.21    0.18
AMD      0.45    1.00    0.19    0.16
MSFT     0.21    0.19    1.00    0.52
GOOGL    0.18    0.16    0.52    1.00
```

Lower values = stocks are driven by different company-specific factors.

**Example interpretation**:
- NVDA-AMD: 0.45 (still correlated even after market factors - both in AI chips)
- NVDA-MSFT: 0.21 (low idiosyncratic correlation - different drivers)

## 🔧 API Endpoints

### Compute Exposure Matrices

```bash
POST /exposure/compute?etf=QQQ&lookback_days=252&n_factors=5
```

Response:
```json
{
  "success": true,
  "etf": "QQQ",
  "n_stocks": 15,
  "tickers": ["NVDA", "MSFT", ...],
  "date_range": ["2023-12-01", "2024-11-22"],
  "n_factors": 5,
  "explained_variance": [0.42, 0.18, 0.11, 0.07, 0.05]
}
```

### Get Exposure Matrix

```bash
GET /exposure/matrix/QQQ?matrix_type=correlation&max_age_days=7
```

Response:
```json
{
  "etf": "QQQ",
  "matrix_type": "correlation",
  "tickers": ["NVDA", "AMD", "MSFT", ...],
  "matrix": [[1.0, 0.75, ...], [0.75, 1.0, ...], ...],
  "params": {"n_stocks": 15},
  "date_range": {"start": "2023-12-01", "end": "2024-11-22"},
  "created_at": "2024-11-22T10:30:00"
}
```

### Compare ETFs

```bash
GET /exposure/compare/QQQ/VYM?matrix_type=correlation
```

Response:
```json
{
  "success": true,
  "etf1": "QQQ",
  "etf2": "VYM",
  "matrix_type": "correlation",
  "common_tickers": [],  // QQQ and VYM have no overlap
  "etf1_only_tickers": ["NVDA", "AMD", ...],
  "etf2_only_tickers": ["JNJ", "PG", ...],
  "similarity_score": 0.15,  // Low similarity (opposite ETFs)
  "stats": {
    "n_common": 0,
    "n_etf1_only": 15,
    "n_etf2_only": 15,
    "overlap_ratio": 0.0
  }
}
```

### Pairwise Exposure

```bash
GET /exposure/pairwise/QQQ/NVDA/AMD
```

Response:
```json
{
  "etf": "QQQ",
  "ticker1": "NVDA",
  "ticker2": "AMD",
  "correlation": 0.752,
  "covariance": 0.00149,
  "idiosyncratic_score": 0.548,  // After removing market factors
  "common_factor_exposure": 0.823,  // Shared market/sector exposure
  "specific_exposure": 0.452,  // Stock-specific correlation
  "computation_date": "2024-11-22"
}
```

### Find Similar Stocks

```bash
GET /exposure/similar/QQQ/NVDA?metric=correlation&top_n=5
```

Response:
```json
{
  "etf": "QQQ",
  "ticker": "NVDA",
  "metric": "correlation",
  "similar_stocks": [
    {"ticker": "AMD", "score": 0.752},
    {"ticker": "AVGO", "score": 0.681},
    {"ticker": "QCOM", "score": 0.623},
    {"ticker": "INTC", "score": 0.587},
    {"ticker": "MSFT", "score": 0.621}
  ]
}
```

## 🎨 Frontend Integration

### Example: Heatmap Visualization

```javascript
// Fetch correlation matrix
const response = await fetch('http://localhost:8000/exposure/matrix/QQQ?matrix_type=correlation');
const data = await response.json();

// data.tickers: ["NVDA", "AMD", "MSFT", ...]
// data.matrix: [[1.0, 0.75, ...], [0.75, 1.0, ...], ...]

// Use a library like d3.js or plotly.js to create a heatmap
```

### Example: ETF Comparison UI

```javascript
// Compare two ETFs
const response = await fetch('http://localhost:8000/exposure/compare/QQQ/VYM');
const comparison = await response.json();

// Display results
console.log(`Similarity: ${comparison.similarity_score.toFixed(2)}`);
console.log(`Common stocks: ${comparison.stats.n_common}`);
console.log(`QQQ only: ${comparison.stats.n_etf1_only}`);
console.log(`VYM only: ${comparison.stats.n_etf2_only}`);
```

### Example: Stock Similarity Finder

```javascript
// Find stocks similar to NVDA
const response = await fetch('http://localhost:8000/exposure/similar/QQQ/NVDA?top_n=10');
const data = await response.json();

// Display as a ranked list
data.similar_stocks.forEach((item, index) => {
  console.log(`${index + 1}. ${item.ticker}: ${item.score.toFixed(3)}`);
});
```

## 📐 Linear Algebra Details

### Correlation Matrix Calculation

```python
# Returns matrix R (n_stocks x n_days)
returns = prices.pct_change()

# Correlation matrix C (n_stocks x n_stocks)
correlation = returns.corr()
```

### Factor Decomposition (PCA)

```python
from sklearn.decomposition import PCA

# Extract k factors
pca = PCA(n_components=5)
factor_returns = pca.fit_transform(returns)

# Factor loadings (how each stock loads on each factor)
loadings = pca.components_.T * sqrt(pca.explained_variance_)

# Reconstruct common returns
common_returns = factor_returns @ pca.components_

# Idiosyncratic returns
idiosyncratic = returns - common_returns

# Idiosyncratic correlation
idio_corr = corrcoef(idiosyncratic.T)
```

### Handling Different Sized ETFs

When comparing ETFs with different numbers of stocks:

1. **Common stocks**: Use standard matrix comparison on overlap
2. **Cross-ETF matrix**: Build rectangular matrix (n1 x n2) using all available data
3. **Similarity score**: Weight by overlap ratio

```python
# For QQQ (15 stocks) vs VYM (15 stocks), no overlap:
# Build 15x15 cross-matrix using historical correlations
cross_matrix = zeros((15, 15))
for i, stock1 in enumerate(qqq_stocks):
    for j, stock2 in enumerate(vym_stocks):
        cross_matrix[i, j] = historical_correlation(stock1, stock2)
```

## 🔄 Incremental Updates

The system caches all results for efficiency:

### Automatic Caching

- **Exposure matrices**: Cached for 7 days (configurable)
- **Pairwise exposures**: Stored in database
- **ETF comparisons**: Cached by computation date

### Update When Holdings Change

```bash
# Holdings changed? Recompute:
python scripts/analyze_exposures.py compute QQQ

# This will:
# 1. Fetch new holdings
# 2. Fetch any new price data
# 3. Recompute all matrices
# 4. Update pairwise exposures
```

### Viewing Cached Data

```bash
# View cached matrix (doesn't recompute)
python scripts/analyze_exposures.py view QQQ --matrix-type correlation

# Save to file for external analysis
python scripts/analyze_exposures.py view QQQ --save qqq_correlation.json
```

## 📈 Use Cases

### 1. Portfolio Diversification

Find stocks with low correlation to reduce risk:

```bash
# Find stocks least correlated with NVDA
python scripts/analyze_exposures.py similar QQQ NVDA --metric correlation --top 5
# Look for lowest scores
```

### 2. Sector Analysis

Compare sector ETFs to understand overlap:

```bash
# Compare tech (QQQ) vs healthcare (XLV)
python scripts/analyze_exposures.py compare QQQ XLV
```

### 3. Risk Factor Exposure

Identify common factors driving returns:

```python
# Via API
response = await fetch('http://localhost:8000/exposure/matrix/QQQ?matrix_type=idiosyncratic');
# Lower idiosyncratic correlation = more independent risks
```

### 4. Pairs Trading

Find highly correlated pairs for statistical arbitrage:

```bash
# Find stocks most correlated with AMD
python scripts/analyze_exposures.py similar QQQ AMD --metric correlation
```

## ⚙️ Configuration

### Matrix Parameters

Adjust in code or via API:

- **lookback_days**: Historical period (default: 252 = 1 year)
- **n_factors**: Number of PCA factors (default: 5)
- **max_age_days**: Cache expiry (default: 7)

### Performance Tuning

For large ETFs (100+ stocks):

- Use longer cache periods (30 days)
- Reduce n_factors (3-4)
- Compute during off-hours

## 🐛 Troubleshooting

### No holdings found

**Problem**: "No holdings found for ETF"

**Solution**:
1. Add to `config.yaml`:
   ```yaml
   etfs:
     qqq_ai_stocks:
       - "NVDA"
       - "AMD"
       # ... add all stocks
   ```

2. Or let system fetch automatically (may not always work)

### Insufficient price data

**Problem**: "No returns data available"

**Solution**:
- Increase lookback period: `--lookback 365`
- Check if stocks are active/have sufficient history
- Manually verify data exists: Check `stock_prices` table

### Matrix computation fails

**Problem**: "Matrix computation failed"

**Solution**:
- Ensure at least 2 stocks have data
- Check for data quality (no all-zero returns)
- Reduce n_factors if you have fewer than 10 stocks

### API returns 404

**Problem**: Matrix not found

**Solution**:
```bash
# Run computation first
python scripts/analyze_exposures.py compute QQQ
```

## 📚 Advanced Topics

### Custom Matrix Metrics

You can extend the system with custom metrics:

1. **Sharpe ratio weighted correlation**
2. **Downside correlation** (only negative returns)
3. **Rolling correlations** (time-varying)

### Alternative Factor Models

Instead of PCA, you could use:

- **Fama-French factors** (market, size, value)
- **Sector factors**
- **Custom factors** based on fundamentals

### Real-Time Updates

For production systems:

1. Schedule daily recomputation (cron job)
2. Stream price updates
3. Incremental matrix updates (only changed stocks)

## 📖 References

- **PCA for Finance**: Jolliffe, I.T. (2002). Principal Component Analysis
- **Portfolio Theory**: Markowitz, H. (1952). Portfolio Selection
- **Risk Models**: Barra Risk Models Documentation

---

**Questions or issues?** Check the logs in `logs/` or run with `--help` for more options.
