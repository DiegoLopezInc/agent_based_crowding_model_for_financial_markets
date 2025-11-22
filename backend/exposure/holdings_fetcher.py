"""ETF holdings fetcher and manager"""

from typing import List, Dict, Any, Optional
from datetime import date, datetime, timedelta
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.database.connection import get_db_manager
from backend.database.exposure_models import ETFCompositionSnapshot, StockPrice
from backend.utils.logger import setup_logger
from backend.utils.config import get_config

logger = setup_logger("etf_holdings")


class ETFHoldingsFetcher:
    """Fetch and manage ETF holdings data"""

    def __init__(self):
        self.config = get_config()
        self.db_manager = get_db_manager()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def fetch_etf_holdings(self, etf: str) -> List[Dict[str, Any]]:
        """Fetch current holdings for an ETF

        Args:
            etf: ETF ticker symbol

        Returns:
            List of holdings with ticker, weight, shares
        """
        logger.info(f"Fetching holdings for {etf}")

        try:
            ticker_obj = yf.Ticker(etf)

            # Try to get holdings
            holdings = []

            # For ETFs, yfinance provides holdings data
            if hasattr(ticker_obj, 'get_holdings'):
                holdings_data = ticker_obj.get_holdings()

                if holdings_data is not None and not holdings_data.empty:
                    for idx, row in holdings_data.iterrows():
                        holdings.append({
                            'ticker': row.get('symbol', idx),
                            'weight': float(row.get('holdingPercent', 0)),
                            'shares': int(row.get('shares', 0)) if row.get('shares') else None,
                            'value': float(row.get('value', 0)) if row.get('value') else None
                        })

            # Fallback: use configured holdings
            if not holdings:
                logger.warning(f"Could not fetch holdings for {etf}, using configured holdings")
                holdings = self._get_configured_holdings(etf)

            logger.info(f"Fetched {len(holdings)} holdings for {etf}")
            return holdings

        except Exception as e:
            logger.error(f"Failed to fetch holdings for {etf}: {e}")
            # Fallback to configured holdings
            return self._get_configured_holdings(etf)

    def _get_configured_holdings(self, etf: str) -> List[Dict[str, Any]]:
        """Get holdings from configuration

        Args:
            etf: ETF ticker

        Returns:
            List of holdings
        """
        etf = etf.upper()

        if etf == self.config.etfs.primary:
            tickers = self.config.etfs.qqq_ai_stocks
        elif self.config.etfs.opposite_etf and etf == self.config.etfs.opposite_etf:
            tickers = self.config.etfs.opposite_etf_stocks
        else:
            logger.warning(f"No configured holdings for {etf}")
            return []

        # Equal weight for simplicity
        weight = 100.0 / len(tickers) if tickers else 0

        return [
            {'ticker': ticker, 'weight': weight, 'shares': None, 'value': None}
            for ticker in tickers
        ]

    def save_snapshot(self, etf: str, holdings: List[Dict[str, Any]], snapshot_date: Optional[date] = None):
        """Save holdings snapshot to database

        Args:
            etf: ETF ticker
            holdings: List of holdings
            snapshot_date: Date of snapshot (default: today)
        """
        if snapshot_date is None:
            snapshot_date = date.today()

        logger.info(f"Saving snapshot for {etf} on {snapshot_date}")

        with self.db_manager.get_session() as session:
            # Check if snapshot exists
            existing = session.query(ETFCompositionSnapshot).filter_by(
                etf=etf,
                snapshot_date=snapshot_date
            ).first()

            if existing:
                # Update
                existing.holdings = holdings
                existing.total_holdings = len(holdings)
                logger.info(f"Updated existing snapshot for {etf}")
            else:
                # Create new
                snapshot = ETFCompositionSnapshot(
                    etf=etf,
                    snapshot_date=snapshot_date,
                    holdings=holdings,
                    total_holdings=len(holdings)
                )
                session.add(snapshot)
                logger.info(f"Created new snapshot for {etf}")

    def get_latest_snapshot(self, etf: str) -> Optional[Dict[str, Any]]:
        """Get latest holdings snapshot from database

        Args:
            etf: ETF ticker

        Returns:
            Snapshot data or None
        """
        with self.db_manager.get_session() as session:
            snapshot = (
                session.query(ETFCompositionSnapshot)
                .filter_by(etf=etf)
                .order_by(ETFCompositionSnapshot.snapshot_date.desc())
                .first()
            )

            if snapshot:
                return {
                    'etf': snapshot.etf,
                    'date': snapshot.snapshot_date,
                    'holdings': snapshot.holdings,
                    'total_holdings': snapshot.total_holdings
                }
            return None

    def get_holdings_tickers(self, etf: str) -> List[str]:
        """Get list of tickers in ETF

        Args:
            etf: ETF ticker

        Returns:
            List of ticker symbols
        """
        snapshot = self.get_latest_snapshot(etf)

        if snapshot:
            return [h['ticker'] for h in snapshot['holdings']]
        else:
            # Fetch and save
            holdings = self.fetch_etf_holdings(etf)
            if holdings:
                self.save_snapshot(etf, holdings)
                return [h['ticker'] for h in holdings]
            return []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def fetch_price_history(
        self,
        ticker: str,
        start_date: date,
        end_date: Optional[date] = None,
        interval: str = '1d'
    ) -> List[Dict[str, Any]]:
        """Fetch price history for a ticker

        Args:
            ticker: Stock ticker
            start_date: Start date
            end_date: End date (default: today)
            interval: Data interval

        Returns:
            List of price records
        """
        if end_date is None:
            end_date = date.today()

        logger.info(f"Fetching price history for {ticker} from {start_date} to {end_date}")

        try:
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(
                start=start_date,
                end=end_date,
                interval=interval
            )

            if hist.empty:
                logger.warning(f"No price data for {ticker}")
                return []

            prices = []
            for date_idx, row in hist.iterrows():
                prices.append({
                    'ticker': ticker,
                    'date': date_idx.date(),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'adjusted_close': float(row['Close']),  # yfinance adjusts by default
                    'volume': int(row['Volume'])
                })

            logger.info(f"Fetched {len(prices)} price records for {ticker}")
            return prices

        except Exception as e:
            logger.error(f"Failed to fetch prices for {ticker}: {e}")
            return []

    def save_prices(self, prices: List[Dict[str, Any]]):
        """Save price records to database

        Args:
            prices: List of price records
        """
        if not prices:
            return

        logger.info(f"Saving {len(prices)} price records")

        with self.db_manager.get_session() as session:
            for price_data in prices:
                # Check if exists
                existing = session.query(StockPrice).filter_by(
                    ticker=price_data['ticker'],
                    date=price_data['date']
                ).first()

                if existing:
                    # Update
                    for key, value in price_data.items():
                        if key not in ['ticker', 'date']:
                            setattr(existing, key, value)
                else:
                    # Create
                    price = StockPrice(**price_data)
                    session.add(price)

        logger.info("Price records saved")

    def fetch_and_save_etf_prices(
        self,
        etf: str,
        lookback_days: int = 252  # 1 trading year
    ) -> int:
        """Fetch and save prices for all stocks in an ETF

        Args:
            etf: ETF ticker
            lookback_days: Number of days to look back

        Returns:
            Number of stocks processed
        """
        tickers = self.get_holdings_tickers(etf)

        if not tickers:
            logger.warning(f"No tickers found for {etf}")
            return 0

        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)

        logger.info(f"Fetching prices for {len(tickers)} stocks in {etf}")

        processed = 0
        for ticker in tickers:
            try:
                prices = self.fetch_price_history(ticker, start_date, end_date)
                if prices:
                    self.save_prices(prices)
                    processed += 1
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
                continue

        logger.info(f"Processed prices for {processed}/{len(tickers)} stocks")
        return processed


def get_holdings_fetcher() -> ETFHoldingsFetcher:
    """Get holdings fetcher instance"""
    return ETFHoldingsFetcher()
