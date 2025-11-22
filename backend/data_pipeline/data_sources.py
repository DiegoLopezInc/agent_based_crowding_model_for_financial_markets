"""Data sources for fetching financial information"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import yfinance as yf
import feedparser
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.utils.config import get_config
from backend.utils.logger import setup_logger

logger = setup_logger("data_sources")


class YahooFinanceSource:
    """Fetch data from Yahoo Finance"""

    def __init__(self):
        self.config = get_config()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_stock_info(self, ticker: str) -> Dict[str, Any]:
        """Get stock information

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with stock info
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            return {
                "ticker": ticker,
                "name": info.get("longName", ticker),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "description": info.get("longBusinessSummary"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "price": info.get("currentPrice"),
                "52_week_high": info.get("fiftyTwoWeekHigh"),
                "52_week_low": info.get("fiftyTwoWeekLow"),
                "volume": info.get("volume"),
                "avg_volume": info.get("averageVolume"),
                "website": info.get("website"),
            }
        except Exception as e:
            logger.error(f"Failed to fetch info for {ticker}: {e}")
            return {"ticker": ticker, "error": str(e)}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_stock_news(self, ticker: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent news for a stock

        Args:
            ticker: Stock ticker symbol
            limit: Maximum number of news items

        Returns:
            List of news items
        """
        try:
            stock = yf.Ticker(ticker)
            news = stock.news[:limit] if hasattr(stock, 'news') else []

            processed_news = []
            for item in news:
                processed_news.append({
                    "ticker": ticker,
                    "title": item.get("title"),
                    "publisher": item.get("publisher"),
                    "link": item.get("link"),
                    "published": datetime.fromtimestamp(item.get("providerPublishTime", 0)),
                    "summary": item.get("summary", ""),
                    "type": "news",
                })

            logger.info(f"Fetched {len(processed_news)} news items for {ticker}")
            return processed_news

        except Exception as e:
            logger.error(f"Failed to fetch news for {ticker}: {e}")
            return []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_earnings_history(self, ticker: str) -> List[Dict[str, Any]]:
        """Get earnings history

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of earnings reports
        """
        try:
            stock = yf.Ticker(ticker)
            earnings = stock.earnings_history

            if earnings is None or earnings.empty:
                return []

            processed_earnings = []
            for _, row in earnings.iterrows():
                processed_earnings.append({
                    "ticker": ticker,
                    "date": row.get("Earnings Date"),
                    "eps_estimate": row.get("EPS Estimate"),
                    "eps_actual": row.get("EPS Actual"),
                    "surprise": row.get("Surprise(%)"),
                    "type": "earnings",
                })

            logger.info(f"Fetched {len(processed_earnings)} earnings reports for {ticker}")
            return processed_earnings

        except Exception as e:
            logger.error(f"Failed to fetch earnings for {ticker}: {e}")
            return []


class FinancialNewsSource:
    """Fetch financial news from RSS feeds"""

    def __init__(self):
        self.config = get_config()
        self.feeds = []

        # Get RSS feeds from config
        for source in self.config.data_pipeline.sources:
            if source.name == "financial_news" and source.enabled and source.rss_feeds:
                self.feeds.extend(source.rss_feeds)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def fetch_feed_news(
        self,
        feed_url: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Fetch news from an RSS feed

        Args:
            feed_url: RSS feed URL
            limit: Maximum number of articles

        Returns:
            List of news articles
        """
        try:
            feed = feedparser.parse(feed_url)
            articles = []

            for entry in feed.entries[:limit]:
                published = entry.get("published_parsed")
                if published:
                    published_date = datetime(*published[:6])
                else:
                    published_date = datetime.now()

                articles.append({
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", ""),
                    "link": entry.get("link", ""),
                    "published": published_date,
                    "source": feed.feed.get("title", "Unknown"),
                    "type": "financial_news",
                })

            logger.info(f"Fetched {len(articles)} articles from {feed_url}")
            return articles

        except Exception as e:
            logger.error(f"Failed to fetch feed {feed_url}: {e}")
            return []

    def fetch_all_feeds(self, limit_per_feed: int = 20) -> List[Dict[str, Any]]:
        """Fetch news from all configured feeds

        Args:
            limit_per_feed: Maximum articles per feed

        Returns:
            List of all news articles
        """
        all_articles = []

        for feed_url in self.feeds:
            articles = self.fetch_feed_news(feed_url, limit=limit_per_feed)
            all_articles.extend(articles)

        logger.info(f"Fetched total of {len(all_articles)} articles from all feeds")
        return all_articles


class ETFDataPipeline:
    """Main data pipeline for ETF research"""

    def __init__(self):
        self.config = get_config()
        self.yahoo = YahooFinanceSource()
        self.news = FinancialNewsSource()

    def get_ai_stocks(self) -> List[str]:
        """Get list of AI-related stocks from config

        Returns:
            List of ticker symbols
        """
        return self.config.etfs.qqq_ai_stocks

    def fetch_stock_data(self, ticker: str) -> Dict[str, Any]:
        """Fetch comprehensive data for a stock

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with all stock data
        """
        logger.info(f"Fetching data for {ticker}")

        data = {
            "ticker": ticker,
            "info": self.yahoo.get_stock_info(ticker),
            "news": self.yahoo.get_stock_news(ticker, limit=10),
            "earnings": self.yahoo.get_earnings_history(ticker),
        }

        return data

    def fetch_all_ai_stocks_data(self) -> List[Dict[str, Any]]:
        """Fetch data for all AI stocks

        Returns:
            List of stock data dictionaries
        """
        ai_stocks = self.get_ai_stocks()
        logger.info(f"Fetching data for {len(ai_stocks)} AI stocks")

        all_data = []
        for ticker in ai_stocks:
            try:
                stock_data = self.fetch_stock_data(ticker)
                all_data.append(stock_data)
            except Exception as e:
                logger.error(f"Failed to fetch data for {ticker}: {e}")
                continue

        logger.info(f"Successfully fetched data for {len(all_data)} stocks")
        return all_data

    def fetch_market_news(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch general market news

        Args:
            limit: Maximum number of articles per feed

        Returns:
            List of news articles
        """
        return self.news.fetch_all_feeds(limit_per_feed=limit)

    def create_documents_from_stock_data(
        self,
        stock_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Convert stock data into document format for embedding

        Args:
            stock_data: Stock data dictionary

        Returns:
            List of documents
        """
        documents = []
        ticker = stock_data["ticker"]

        # Create document from stock info
        info = stock_data.get("info", {})
        if info and "description" in info and info["description"]:
            documents.append({
                "content": f"{ticker} Company Information:\n{info['description']}",
                "metadata": {
                    "name": info.get("name"),
                    "sector": info.get("sector"),
                    "industry": info.get("industry"),
                    "market_cap": info.get("market_cap"),
                },
                "source": "yahoo_finance",
                "ticker": ticker,
                "etf": "QQQ",  # Default to QQQ
                "document_type": "company_info",
                "published_date": datetime.now(),
            })

        # Create documents from news
        for news_item in stock_data.get("news", []):
            content = f"{news_item['title']}\n\n{news_item.get('summary', '')}"
            documents.append({
                "content": content,
                "metadata": {
                    "publisher": news_item.get("publisher"),
                    "link": news_item.get("link"),
                },
                "source": "yahoo_finance_news",
                "ticker": ticker,
                "etf": "QQQ",
                "document_type": "news",
                "published_date": news_item.get("published", datetime.now()),
            })

        # Create documents from earnings
        for earnings in stock_data.get("earnings", []):
            if earnings.get("eps_actual") and earnings.get("eps_estimate"):
                surprise = ((earnings["eps_actual"] - earnings["eps_estimate"]) /
                           earnings["eps_estimate"] * 100)
                content = (
                    f"{ticker} Earnings Report:\n"
                    f"Date: {earnings.get('date')}\n"
                    f"EPS Estimate: ${earnings['eps_estimate']:.2f}\n"
                    f"EPS Actual: ${earnings['eps_actual']:.2f}\n"
                    f"Surprise: {surprise:.2f}%"
                )
                documents.append({
                    "content": content,
                    "metadata": earnings,
                    "source": "yahoo_finance_earnings",
                    "ticker": ticker,
                    "etf": "QQQ",
                    "document_type": "earnings",
                    "published_date": earnings.get("date", datetime.now()),
                })

        logger.info(f"Created {len(documents)} documents for {ticker}")
        return documents


def get_data_pipeline() -> ETFDataPipeline:
    """Get the data pipeline instance"""
    return ETFDataPipeline()
