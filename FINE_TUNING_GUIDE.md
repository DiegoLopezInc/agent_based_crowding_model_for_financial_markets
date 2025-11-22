# Fine-Tuning Embeddings with Contrastive Learning

This guide explains how to fine-tune your embedding model using contrastive learning with positive and negative pairs from opposite ETFs.

## 🎯 Overview

The fine-tuning system uses **contrastive learning** to improve the embedding model's ability to distinguish between:
- **Positive pairs**: Similar documents (from the same AI/tech stock)
- **Negative pairs**: Dissimilar documents (AI stocks vs opposite ETF stocks)
- **Hard negatives**: Somewhat similar but different (different AI stocks)

### Why Fine-Tune?

1. **Domain Adaptation**: Adapt general-purpose embeddings to financial text
2. **Better Discrimination**: Learn to separate AI/tech stocks from traditional value stocks
3. **Improved Retrieval**: Better semantic search results for stock research

### Opposite ETF Strategy

We use **VYM** (Vanguard High Dividend Yield) as the opposite of **QQQ** (Nasdaq 100 tech/AI):
- QQQ: Growth, technology, AI-focused (NVDA, MSFT, GOOGL)
- VYM: Value, dividends, traditional sectors (JNJ, PG, XOM)

This creates clear contrasts for the model to learn.

## 📋 Prerequisites

1. **Data from both ETFs ingested**:
   ```bash
   # Ingest AI stocks (QQQ)
   python scripts/run_ingestion.py --all-ai-stocks

   # Ingest opposite ETF stocks (VYM)
   python scripts/run_ingestion.py --ticker JNJ
   python scripts/run_ingestion.py --ticker PG
   python scripts/run_ingestion.py --ticker XOM
   # ... or ingest all at once:
   for ticker in JNJ PG JPM XOM CVX KO PEP WMT VZ T MRK PFE UNH BAC WFC; do
       python scripts/run_ingestion.py --ticker $ticker
   done
   ```

2. **Sufficient GPU memory** (recommended but not required):
   - Fine-tuning works on CPU but is slower
   - GPU significantly speeds up training
   - ~4GB VRAM should be sufficient for the default model

## 🚀 Quick Start

### Step 1: Configure Fine-Tuning

Edit `config.yaml`:

```yaml
embedding:
  fine_tuning:
    enabled: false  # Keep false until after training

    training:
      epochs: 3
      batch_size: 16
      learning_rate: 2e-5

      # Sampling settings
      positive_samples_per_stock: 5  # Pairs from same stock
      negative_samples_per_stock: 5  # Pairs from opposite ETF
      hard_negative_ratio: 0.3  # 30% hard negatives

      # Contrastive loss
      margin: 0.5
      distance_metric: "cosine"
```

### Step 2: Run Fine-Tuning

```bash
# Full pipeline: prepare data, train, evaluate
python scripts/fine_tune_embeddings.py

# Or step by step:

# 1. Prepare data only
python scripts/fine_tune_embeddings.py --prepare-data-only

# 2. Save data for inspection
python scripts/fine_tune_embeddings.py --prepare-data-only --save-data data/training_pairs.jsonl

# 3. Train and evaluate
python scripts/fine_tune_embeddings.py
```

### Step 3: Review Results

The script will output:

```
EVALUATION RESULTS
============================================================

Base Model:
  Threshold: 0.50
  F1 Score: 0.7245
  Accuracy: 0.7156
  AUC-ROC: 0.8234
  Separation: 0.3421

Fine-Tuned Model:
  Threshold: 0.55
  F1 Score: 0.8156
  Accuracy: 0.8045
  AUC-ROC: 0.9012
  Separation: 0.4723

Improvements:
  accuracy: +0.0889 (+12.41%)
  f1_score: +0.0911 (+12.57%)
  auc_roc: +0.0778 (+9.45%)
  separation: +0.1302 (+38.05%)
```

### Step 4: Enable Fine-Tuned Model

If results look good, enable the fine-tuned model in `config.yaml`:

```yaml
embedding:
  fine_tuning:
    enabled: true  # Changed from false
```

Restart your application and the fine-tuned model will be used automatically.

## ⚙️ Configuration Reference

### Training Parameters

```yaml
embedding:
  fine_tuning:
    training:
      # Training
      epochs: 3  # Number of training epochs
      batch_size: 16  # Batch size for training
      learning_rate: 2e-5  # Learning rate (2e-5 is standard for BERT)
      warmup_steps: 100  # Linear warmup steps
      evaluation_steps: 500  # Evaluate every N steps
      save_steps: 500  # Save checkpoint every N steps

      # Contrastive Loss
      margin: 0.5  # Margin for contrastive loss (0.5-1.0 typical)
      distance_metric: "cosine"  # "cosine" or "euclidean"

      # Data Sampling
      positive_samples_per_stock: 5  # Positive pairs per stock
      negative_samples_per_stock: 5  # Negative pairs per stock
      hard_negative_ratio: 0.3  # Ratio of hard negatives (0.0-1.0)

      # Output
      output_dir: "models/fine_tuning_checkpoints"
      logging_dir: "logs/fine_tuning"
```

### Parameter Tuning Guide

| Parameter | Low | Medium | High | Notes |
|-----------|-----|--------|------|-------|
| `epochs` | 1-2 | 3-4 | 5+ | More epochs = better fit, risk overfitting |
| `batch_size` | 8 | 16 | 32+ | Limited by GPU memory |
| `learning_rate` | 1e-5 | 2e-5 | 5e-5 | Too high = unstable, too low = slow |
| `margin` | 0.3 | 0.5 | 1.0 | Larger = more separation required |
| `positive_samples` | 3 | 5 | 10+ | More = better representation |
| `negative_samples` | 3 | 5 | 10+ | More = better discrimination |
| `hard_negative_ratio` | 0.1 | 0.3 | 0.5 | Higher = harder training |

## 📊 Understanding Metrics

### Evaluation Metrics

1. **F1 Score** (0-1, higher better)
   - Balance of precision and recall
   - **Target**: > 0.75 is good, > 0.85 is excellent

2. **Accuracy** (0-1, higher better)
   - Percentage of correctly classified pairs
   - **Target**: > 0.70 is acceptable, > 0.80 is good

3. **AUC-ROC** (0-1, higher better)
   - Area under ROC curve
   - **Target**: > 0.80 is good, > 0.90 is excellent

4. **Separation** (0-1, higher better)
   - Difference between mean positive and negative similarities
   - **Target**: > 0.3 is good, > 0.4 is excellent

### What to Look For

✅ **Good fine-tuning**:
- F1 improvement: +5% to +15%
- Separation improvement: +20% to +50%
- AUC-ROC: > 0.85

❌ **Poor fine-tuning**:
- F1 improvement: < 2%
- Separation: < 0.25
- Model might be overfitting or underfitting

## 🔧 Advanced Usage

### Custom Opposite ETF

You can use a different opposite ETF in `config.yaml`:

```yaml
etfs:
  opposite_etf: "XLE"  # Energy sector instead of VYM
  opposite_etf_stocks:
    - "XOM"
    - "CVX"
    - "COP"
    - "EOG"
    # ... more energy stocks
```

Popular alternatives:
- **XLE**: Energy sector (opposite of tech)
- **XLU**: Utilities (stable, non-tech)
- **VTV**: Value stocks (opposite of growth)
- **IWN**: Small cap value

### Evaluation Only

To re-evaluate without retraining:

```bash
python scripts/fine_tune_embeddings.py --evaluate-only
```

### Inspect Training Data

To see what pairs are being created:

```bash
python scripts/fine_tune_embeddings.py --save-data training_pairs.jsonl --prepare-data-only

# Inspect the file
head -5 training_pairs.jsonl | jq .
```

Example output:
```json
{
  "text1": "NVDA announced new AI chip with 40% performance gain...",
  "text2": "NVDA Q3 earnings beat expectations with strong datacenter growth...",
  "label": 1
}
{
  "text1": "Microsoft Azure AI services expanded to new regions...",
  "text2": "Procter & Gamble reported steady dividend growth...",
  "label": 0
}
```

## 🎯 Best Practices

### 1. Data Quality

✅ **Do**:
- Ensure balanced data from both ETFs
- Have at least 100+ documents per stock
- Include diverse document types (news, earnings, company info)

❌ **Don't**:
- Use only one document type
- Have imbalanced data (much more AI than opposite)
- Fine-tune with < 500 total pairs

### 2. Training

✅ **Do**:
- Start with default parameters
- Monitor evaluation metrics during training
- Save training data for reproducibility

❌ **Don't**:
- Use very high learning rates (> 5e-5)
- Train for too many epochs (> 10)
- Skip evaluation

### 3. Deployment

✅ **Do**:
- Test on a sample of queries first
- Compare retrieval quality before/after
- Keep the base model as backup

❌ **Don't**:
- Enable fine-tuned model without testing
- Delete base model
- Skip evaluation step

## 🐛 Troubleshooting

### No Training Pairs Created

**Problem**: "No training pairs created"

**Solutions**:
```bash
# Check database has data
python -c "from backend.database.connection import get_db_manager; print(get_db_manager().get_stats())"

# Verify opposite ETF stocks are ingested
python -c "
from backend.database.connection import get_db_manager
from backend.database.models import Document
with get_db_manager().get_session() as session:
    tickers = session.query(Document.ticker).distinct().all()
    print([t[0] for t in tickers])
"

# Ingest missing stocks
python scripts/run_ingestion.py --ticker JNJ
```

### Out of Memory

**Problem**: CUDA out of memory during training

**Solutions**:
1. Reduce batch size in `config.yaml`:
   ```yaml
   batch_size: 8  # or even 4
   ```

2. Use CPU training (slower but works):
   - Training automatically uses CPU if CUDA unavailable

### Poor Results

**Problem**: Metrics don't improve or get worse

**Solutions**:

1. **Check data quality**:
   ```bash
   python scripts/fine_tune_embeddings.py --save-data check.jsonl --prepare-data-only
   head -20 check.jsonl | jq .
   ```

2. **Adjust parameters**:
   - Increase `epochs` (try 5)
   - Increase `positive_samples_per_stock` (try 10)
   - Adjust `margin` (try 0.7)

3. **More diverse opposite stocks**:
   - Ingest more stocks from opposite ETF
   - Try different opposite ETF

### Model Not Loading

**Problem**: Fine-tuned model not being used

**Solutions**:
```bash
# Check if model exists
ls -la models/fine_tuned_embeddings/

# Check config
grep -A5 "fine_tuning" config.yaml

# Check logs
tail -50 logs/etf_rag_*.log | grep -i "fine"
```

## 📈 Expected Performance

Based on typical results:

| Metric | Base Model | After Fine-Tuning | Improvement |
|--------|-----------|-------------------|-------------|
| F1 Score | 0.68-0.75 | 0.78-0.88 | +10-18% |
| Accuracy | 0.65-0.72 | 0.75-0.85 | +10-15% |
| AUC-ROC | 0.78-0.85 | 0.88-0.95 | +8-12% |
| Separation | 0.28-0.35 | 0.40-0.52 | +30-50% |

## 🔄 Iterative Improvement

1. **First run**: Use default parameters, evaluate results
2. **Analyze**: Look at which pairs are misclassified
3. **Adjust**:
   - If overfitting: reduce epochs, increase regularization
   - If underfitting: increase epochs, samples per stock
4. **Re-train**: With new parameters
5. **Compare**: Against previous best model

## 📚 Next Steps

After fine-tuning:

1. **A/B Test**: Compare search quality
   ```bash
   # Base model
   python scripts/query_agent.py "What is NVDA's AI strategy?" --mode rag

   # Enable fine-tuned model in config.yaml
   python scripts/query_agent.py "What is NVDA's AI strategy?" --mode rag
   ```

2. **Monitor Performance**: Track retrieval quality over time

3. **Re-fine-tune**: As you add more data
   - Re-run fine-tuning monthly or when adding significant data
   - Use more epochs (5-7) for incremental improvements

4. **Experiment**: Try different opposite ETFs for different use cases

---

**Questions or issues?** Check the logs in `logs/fine_tuning/` for detailed training information.
