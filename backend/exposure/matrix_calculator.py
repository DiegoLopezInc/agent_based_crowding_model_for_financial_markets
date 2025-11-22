"""Exposure matrix calculator using linear algebra"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import date, timedelta
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf

from backend.database.connection import get_db_manager
from backend.database.exposure_models import StockPrice, ExposureMatrix, PairwiseExposure
from backend.utils.logger import setup_logger

logger = setup_logger("matrix_calculator")


class ExposureMatrixCalculator:
    """Calculate exposure matrices using linear algebra"""

    def __init__(self):
        self.db_manager = get_db_manager()

    def get_returns_matrix(
        self,
        tickers: List[str],
        start_date: date,
        end_date: date
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Get returns matrix for a list of tickers

        Args:
            tickers: List of stock tickers
            start_date: Start date
            end_date: End date

        Returns:
            (returns_df, valid_tickers) - DataFrame with returns and list of valid tickers
        """
        logger.info(f"Building returns matrix for {len(tickers)} stocks")

        with self.db_manager.get_session() as session:
            # Fetch prices for all tickers
            prices_data = {}

            for ticker in tickers:
                prices = (
                    session.query(StockPrice.date, StockPrice.adjusted_close)
                    .filter(
                        StockPrice.ticker == ticker,
                        StockPrice.date >= start_date,
                        StockPrice.date <= end_date
                    )
                    .order_by(StockPrice.date)
                    .all()
                )

                if prices:
                    prices_data[ticker] = pd.DataFrame(
                        [(p.date, p.adjusted_close) for p in prices],
                        columns=['date', 'price']
                    ).set_index('date')

            if not prices_data:
                logger.warning("No price data found")
                return pd.DataFrame(), []

            # Combine into single DataFrame
            prices_df = pd.DataFrame(prices_data).fillna(method='ffill').dropna()

            # Calculate returns
            returns_df = prices_df.pct_change().dropna()

            logger.info(f"Returns matrix shape: {returns_df.shape}")

            return returns_df, list(returns_df.columns)

    def calculate_correlation_matrix(
        self,
        returns_df: pd.DataFrame
    ) -> np.ndarray:
        """Calculate correlation matrix

        Args:
            returns_df: DataFrame of returns

        Returns:
            Correlation matrix
        """
        logger.info("Calculating correlation matrix")

        corr_matrix = returns_df.corr().values

        logger.info(f"Correlation matrix shape: {corr_matrix.shape}")

        return corr_matrix

    def calculate_covariance_matrix(
        self,
        returns_df: pd.DataFrame,
        use_shrinkage: bool = True
    ) -> np.ndarray:
        """Calculate covariance matrix

        Args:
            returns_df: DataFrame of returns
            use_shrinkage: Use Ledoit-Wolf shrinkage estimator

        Returns:
            Covariance matrix
        """
        logger.info("Calculating covariance matrix")

        if use_shrinkage:
            # Use Ledoit-Wolf shrinkage for more stable estimates
            lw = LedoitWolf()
            cov_matrix = lw.fit(returns_df.values).covariance_
        else:
            cov_matrix = returns_df.cov().values

        logger.info(f"Covariance matrix shape: {cov_matrix.shape}")

        return cov_matrix

    def calculate_idiosyncratic_matrix(
        self,
        returns_df: pd.DataFrame,
        n_factors: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate idiosyncratic risk using factor decomposition

        Uses PCA to extract common factors, then calculates idiosyncratic component.

        Args:
            returns_df: DataFrame of returns
            n_factors: Number of factors to extract

        Returns:
            (idiosyncratic_matrix, factor_loadings, explained_variance)
        """
        logger.info(f"Calculating idiosyncratic exposures with {n_factors} factors")

        # Perform PCA
        pca = PCA(n_components=n_factors)
        factor_returns = pca.fit_transform(returns_df.values)

        # Factor loadings (how much each stock loads on each factor)
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

        # Reconstruct returns from factors
        common_returns = factor_returns @ pca.components_

        # Idiosyncratic returns = actual returns - common factor returns
        idiosyncratic_returns = returns_df.values - common_returns

        # Idiosyncratic correlation matrix
        idio_corr = np.corrcoef(idiosyncratic_returns.T)

        logger.info(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
        logger.info(f"Idiosyncratic matrix shape: {idio_corr.shape}")

        return idio_corr, loadings, pca.explained_variance_ratio_

    def calculate_pairwise_metrics(
        self,
        ticker1: str,
        ticker2: str,
        corr_matrix: np.ndarray,
        cov_matrix: np.ndarray,
        idio_matrix: np.ndarray,
        loadings: np.ndarray,
        tickers_list: List[str]
    ) -> Dict[str, float]:
        """Calculate pairwise exposure metrics

        Args:
            ticker1: First ticker
            ticker2: Second ticker
            corr_matrix: Correlation matrix
            cov_matrix: Covariance matrix
            idio_matrix: Idiosyncratic correlation matrix
            loadings: Factor loadings
            tickers_list: Ordered list of tickers

        Returns:
            Dictionary with pairwise metrics
        """
        idx1 = tickers_list.index(ticker1)
        idx2 = tickers_list.index(ticker2)

        # Correlation and covariance
        correlation = float(corr_matrix[idx1, idx2])
        covariance = float(cov_matrix[idx1, idx2])

        # Idiosyncratic score (how different they are in idiosyncratic space)
        idio_score = 1.0 - float(idio_matrix[idx1, idx2])

        # Common factor exposure (dot product of loadings)
        common_exposure = float(np.dot(loadings[idx1], loadings[idx2]))

        # Specific exposure (idiosyncratic correlation)
        specific_exposure = float(idio_matrix[idx1, idx2])

        return {
            'correlation': correlation,
            'covariance': covariance,
            'idiosyncratic_score': idio_score,
            'common_factor_exposure': common_exposure,
            'specific_exposure': specific_exposure
        }

    def compute_and_save_matrices(
        self,
        etf: str,
        tickers: List[str],
        start_date: date,
        end_date: date,
        n_factors: int = 5
    ) -> Dict[str, Any]:
        """Compute and save all exposure matrices for an ETF

        Args:
            etf: ETF ticker
            tickers: List of stock tickers
            start_date: Start date
            end_date: End date
            n_factors: Number of factors for PCA

        Returns:
            Dictionary with computation results
        """
        logger.info(f"Computing exposure matrices for {etf}")

        # Get returns
        returns_df, valid_tickers = self.get_returns_matrix(tickers, start_date, end_date)

        if returns_df.empty:
            logger.error("No returns data available")
            return {'success': False, 'error': 'No returns data'}

        # Calculate matrices
        corr_matrix = self.calculate_correlation_matrix(returns_df)
        cov_matrix = self.calculate_covariance_matrix(returns_df)
        idio_matrix, loadings, explained_var = self.calculate_idiosyncratic_matrix(
            returns_df, n_factors
        )

        # Save to database
        with self.db_manager.get_session() as session:
            # Correlation matrix
            self._save_matrix(
                session, etf, 'correlation', valid_tickers,
                corr_matrix, start_date, end_date,
                {'n_stocks': len(valid_tickers)}
            )

            # Covariance matrix
            self._save_matrix(
                session, etf, 'covariance', valid_tickers,
                cov_matrix, start_date, end_date,
                {'n_stocks': len(valid_tickers)}
            )

            # Idiosyncratic matrix
            self._save_matrix(
                session, etf, 'idiosyncratic', valid_tickers,
                idio_matrix, start_date, end_date,
                {
                    'n_stocks': len(valid_tickers),
                    'n_factors': n_factors,
                    'explained_variance': explained_var.tolist()
                }
            )

            # Save pairwise exposures
            self._save_pairwise_exposures(
                session, etf, valid_tickers,
                corr_matrix, cov_matrix, idio_matrix,
                loadings, end_date
            )

        logger.info(f"Saved all matrices for {etf}")

        return {
            'success': True,
            'etf': etf,
            'n_stocks': len(valid_tickers),
            'tickers': valid_tickers,
            'date_range': (start_date, end_date),
            'n_factors': n_factors,
            'explained_variance': explained_var.tolist()
        }

    def _save_matrix(
        self,
        session,
        etf: str,
        matrix_type: str,
        tickers: List[str],
        matrix: np.ndarray,
        start_date: date,
        end_date: date,
        params: Dict[str, Any]
    ):
        """Save a matrix to database"""
        # Check if exists
        existing = session.query(ExposureMatrix).filter_by(
            etf=etf,
            matrix_type=matrix_type,
            start_date=start_date,
            end_date=end_date
        ).first()

        # Convert matrix to list for JSON storage
        matrix_list = matrix.tolist()

        if existing:
            existing.tickers = tickers
            existing.matrix_data = {'matrix': matrix_list}
            existing.computation_params = params
        else:
            matrix_obj = ExposureMatrix(
                etf=etf,
                matrix_type=matrix_type,
                tickers=tickers,
                matrix_data={'matrix': matrix_list},
                computation_params=params,
                start_date=start_date,
                end_date=end_date
            )
            session.add(matrix_obj)

    def _save_pairwise_exposures(
        self,
        session,
        etf: str,
        tickers: List[str],
        corr_matrix: np.ndarray,
        cov_matrix: np.ndarray,
        idio_matrix: np.ndarray,
        loadings: np.ndarray,
        computation_date: date
    ):
        """Save pairwise exposures to database"""
        logger.info(f"Saving pairwise exposures for {len(tickers)} stocks")

        for i, ticker1 in enumerate(tickers):
            for j, ticker2 in enumerate(tickers):
                if i >= j:  # Only save upper triangle (including diagonal)
                    continue

                metrics = self.calculate_pairwise_metrics(
                    ticker1, ticker2,
                    corr_matrix, cov_matrix, idio_matrix,
                    loadings, tickers
                )

                # Check if exists
                existing = session.query(PairwiseExposure).filter_by(
                    etf=etf,
                    ticker1=ticker1,
                    ticker2=ticker2,
                    computation_date=computation_date
                ).first()

                if existing:
                    for key, value in metrics.items():
                        setattr(existing, key, value)
                else:
                    pair = PairwiseExposure(
                        etf=etf,
                        ticker1=ticker1,
                        ticker2=ticker2,
                        computation_date=computation_date,
                        **metrics
                    )
                    session.add(pair)

    def get_cached_matrix(
        self,
        etf: str,
        matrix_type: str,
        max_age_days: int = 7
    ) -> Optional[Dict[str, Any]]:
        """Get cached matrix from database

        Args:
            etf: ETF ticker
            matrix_type: Type of matrix
            max_age_days: Maximum age in days

        Returns:
            Matrix data or None
        """
        cutoff_date = date.today() - timedelta(days=max_age_days)

        with self.db_manager.get_session() as session:
            matrix_obj = (
                session.query(ExposureMatrix)
                .filter(
                    ExposureMatrix.etf == etf,
                    ExposureMatrix.matrix_type == matrix_type,
                    ExposureMatrix.created_at >= cutoff_date
                )
                .order_by(ExposureMatrix.created_at.desc())
                .first()
            )

            if matrix_obj:
                return {
                    'tickers': matrix_obj.tickers,
                    'matrix': np.array(matrix_obj.matrix_data['matrix']),
                    'params': matrix_obj.computation_params,
                    'start_date': matrix_obj.start_date,
                    'end_date': matrix_obj.end_date,
                    'created_at': matrix_obj.created_at
                }

            return None


def get_matrix_calculator() -> ExposureMatrixCalculator:
    """Get matrix calculator instance"""
    return ExposureMatrixCalculator()
