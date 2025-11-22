"""ETF comparison engine using linear algebra"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import date
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as cosine_distance

from backend.database.connection import get_db_manager
from backend.database.exposure_models import ETFComparison, PairwiseExposure
from backend.exposure.matrix_calculator import ExposureMatrixCalculator
from backend.utils.logger import setup_logger

logger = setup_logger("comparison_engine")


class ETFComparisonEngine:
    """Compare ETFs using linear algebra"""

    def __init__(self):
        self.db_manager = get_db_manager()
        self.calculator = ExposureMatrixCalculator()

    def compare_etfs(
        self,
        etf1: str,
        etf2: str,
        matrix_type: str = 'correlation',
        max_age_days: int = 7
    ) -> Dict[str, Any]:
        """Compare two ETFs using their exposure matrices

        Args:
            etf1: First ETF
            etf2: Second ETF
            matrix_type: Type of matrix to use
            max_age_days: Max age for cached matrices

        Returns:
            Comparison results
        """
        logger.info(f"Comparing {etf1} vs {etf2}")

        # Get matrices for both ETFs
        matrix1_data = self.calculator.get_cached_matrix(etf1, matrix_type, max_age_days)
        matrix2_data = self.calculator.get_cached_matrix(etf2, matrix_type, max_age_days)

        if not matrix1_data or not matrix2_data:
            logger.error("Matrices not found, need to compute first")
            return {
                'success': False,
                'error': 'Matrices not found. Run computation first.'
            }

        tickers1 = set(matrix1_data['tickers'])
        tickers2 = set(matrix2_data['tickers'])

        # Find common and unique tickers
        common_tickers = list(tickers1 & tickers2)
        etf1_only = list(tickers1 - tickers2)
        etf2_only = list(tickers2 - tickers1)

        logger.info(f"Common tickers: {len(common_tickers)}")
        logger.info(f"{etf1} only: {len(etf1_only)}")
        logger.info(f"{etf2} only: {len(etf2_only)}")

        # Build comparison matrix
        comparison_matrix = self._build_comparison_matrix(
            matrix1_data, matrix2_data, common_tickers
        )

        # Calculate similarity score
        similarity_score = self._calculate_similarity_score(
            comparison_matrix, len(common_tickers), len(tickers1), len(tickers2)
        )

        result = {
            'success': True,
            'etf1': etf1,
            'etf2': etf2,
            'matrix_type': matrix_type,
            'common_tickers': common_tickers,
            'etf1_only_tickers': etf1_only,
            'etf2_only_tickers': etf2_only,
            'comparison_matrix': comparison_matrix,
            'similarity_score': similarity_score,
            'stats': {
                'n_common': len(common_tickers),
                'n_etf1_only': len(etf1_only),
                'n_etf2_only': len(etf2_only),
                'overlap_ratio': len(common_tickers) / max(len(tickers1), len(tickers2))
            }
        }

        # Save to database
        self._save_comparison(result, date.today())

        return result

    def _build_comparison_matrix(
        self,
        matrix1_data: Dict[str, Any],
        matrix2_data: Dict[str, Any],
        common_tickers: List[str]
    ) -> Dict[str, Any]:
        """Build comparison matrix for common tickers

        Uses matrix multiplication to compare exposures.

        Args:
            matrix1_data: First matrix data
            matrix2_data: Second matrix data
            common_tickers: List of common tickers

        Returns:
            Comparison matrix data
        """
        if not common_tickers:
            return {'matrix': [], 'tickers': []}

        # Extract submatrices for common tickers
        tickers1 = matrix1_data['tickers']
        tickers2 = matrix2_data['tickers']
        matrix1 = matrix1_data['matrix']
        matrix2 = matrix2_data['matrix']

        # Get indices for common tickers
        idx1 = [tickers1.index(t) for t in common_tickers]
        idx2 = [tickers2.index(t) for t in common_tickers]

        # Extract submatrices
        submatrix1 = matrix1[np.ix_(idx1, idx1)]
        submatrix2 = matrix2[np.ix_(idx2, idx2)]

        # Compare matrices element-wise
        diff_matrix = np.abs(submatrix1 - submatrix2)

        # Cross-correlation matrix (how similar are the exposure patterns)
        # Flatten to vectors and compute correlation
        vec1 = submatrix1.flatten()
        vec2 = submatrix2.flatten()

        pattern_correlation = np.corrcoef(vec1, vec2)[0, 1]

        return {
            'matrix': diff_matrix.tolist(),
            'tickers': common_tickers,
            'submatrix1': submatrix1.tolist(),
            'submatrix2': submatrix2.tolist(),
            'pattern_correlation': float(pattern_correlation),
            'mean_difference': float(np.mean(diff_matrix)),
            'max_difference': float(np.max(diff_matrix))
        }

    def _calculate_similarity_score(
        self,
        comparison_matrix: Dict[str, Any],
        n_common: int,
        n_etf1: int,
        n_etf2: int
    ) -> float:
        """Calculate overall similarity score

        Args:
            comparison_matrix: Comparison matrix data
            n_common: Number of common tickers
            n_etf1: Number of tickers in ETF1
            n_etf2: Number of tickers in ETF2

        Returns:
            Similarity score (0-1, higher = more similar)
        """
        if n_common == 0:
            return 0.0

        # Components of similarity
        # 1. Overlap ratio (how many stocks in common)
        overlap_score = n_common / max(n_etf1, n_etf2)

        # 2. Pattern similarity (from correlation)
        pattern_score = (comparison_matrix.get('pattern_correlation', 0) + 1) / 2

        # 3. Magnitude similarity (inverse of mean difference)
        mean_diff = comparison_matrix.get('mean_difference', 1.0)
        magnitude_score = 1.0 / (1.0 + mean_diff)

        # Weighted combination
        similarity = (
            0.4 * overlap_score +
            0.4 * pattern_score +
            0.2 * magnitude_score
        )

        return float(similarity)

    def _save_comparison(self, comparison_data: Dict[str, Any], computation_date: date):
        """Save comparison to database"""
        with self.db_manager.get_session() as session:
            existing = session.query(ETFComparison).filter_by(
                etf1=comparison_data['etf1'],
                etf2=comparison_data['etf2'],
                computation_date=computation_date
            ).first()

            if existing:
                existing.comparison_matrix = comparison_data['comparison_matrix']
                existing.common_tickers = comparison_data['common_tickers']
                existing.etf1_only_tickers = comparison_data['etf1_only_tickers']
                existing.etf2_only_tickers = comparison_data['etf2_only_tickers']
                existing.similarity_score = comparison_data['similarity_score']
            else:
                comp = ETFComparison(
                    etf1=comparison_data['etf1'],
                    etf2=comparison_data['etf2'],
                    comparison_matrix=comparison_data['comparison_matrix'],
                    common_tickers=comparison_data['common_tickers'],
                    etf1_only_tickers=comparison_data['etf1_only_tickers'],
                    etf2_only_tickers=comparison_data['etf2_only_tickers'],
                    similarity_score=comparison_data['similarity_score'],
                    computation_date=computation_date
                )
                session.add(comp)

    def get_pairwise_comparison(
        self,
        etf: str,
        ticker1: str,
        ticker2: str,
        max_age_days: int = 7
    ) -> Optional[Dict[str, Any]]:
        """Get pairwise comparison between two stocks

        Args:
            etf: ETF containing both stocks
            ticker1: First ticker
            ticker2: Second ticker
            max_age_days: Max age for cached data

        Returns:
            Pairwise comparison data or None
        """
        from datetime import timedelta

        cutoff_date = date.today() - timedelta(days=max_age_days)

        with self.db_manager.get_session() as session:
            pair = (
                session.query(PairwiseExposure)
                .filter(
                    PairwiseExposure.etf == etf,
                    PairwiseExposure.ticker1 == ticker1,
                    PairwiseExposure.ticker2 == ticker2,
                    PairwiseExposure.computation_date >= cutoff_date
                )
                .order_by(PairwiseExposure.computation_date.desc())
                .first()
            )

            if pair:
                return {
                    'etf': pair.etf,
                    'ticker1': pair.ticker1,
                    'ticker2': pair.ticker2,
                    'correlation': pair.correlation,
                    'covariance': pair.covariance,
                    'idiosyncratic_score': pair.idiosyncratic_score,
                    'common_factor_exposure': pair.common_factor_exposure,
                    'specific_exposure': pair.specific_exposure,
                    'computation_date': pair.computation_date
                }

            return None

    def get_stock_similarity_ranking(
        self,
        etf: str,
        ticker: str,
        metric: str = 'correlation',
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """Get stocks most/least similar to a given stock

        Args:
            etf: ETF
            ticker: Target ticker
            metric: Metric to use for ranking
            top_n: Number of results

        Returns:
            List of similar stocks with scores
        """
        with self.db_manager.get_session() as session:
            # Get all pairs involving this ticker
            pairs = (
                session.query(PairwiseExposure)
                .filter(
                    PairwiseExposure.etf == etf,
                    ((PairwiseExposure.ticker1 == ticker) | (PairwiseExposure.ticker2 == ticker))
                )
                .all()
            )

            if not pairs:
                return []

            # Extract scores
            scores = []
            for pair in pairs:
                other_ticker = pair.ticker2 if pair.ticker1 == ticker else pair.ticker1
                score = getattr(pair, metric, 0)

                scores.append({
                    'ticker': other_ticker,
                    'score': score,
                    'metric': metric
                })

            # Sort by score (descending)
            scores.sort(key=lambda x: x['score'], reverse=True)

            return scores[:top_n]

    def build_cross_etf_matrix(
        self,
        etf1: str,
        etf2: str,
        tickers1: List[str],
        tickers2: List[str]
    ) -> np.ndarray:
        """Build cross-ETF exposure matrix

        For when ETFs have different stocks, this creates a rectangular matrix
        comparing all stocks from ETF1 to all stocks from ETF2.

        Args:
            etf1: First ETF
            etf2: Second ETF
            tickers1: Tickers from ETF1
            tickers2: Tickers from ETF2

        Returns:
            Cross-exposure matrix (n_tickers1 x n_tickers2)
        """
        logger.info(f"Building cross-ETF matrix: {len(tickers1)} x {len(tickers2)}")

        # For stocks not in common, we can use market-level correlations
        # or historical price correlations

        cross_matrix = np.zeros((len(tickers1), len(tickers2)))

        with self.db_manager.get_session() as session:
            for i, t1 in enumerate(tickers1):
                for j, t2 in enumerate(tickers2):
                    # Try to find pairwise exposure in either ETF
                    pair1 = session.query(PairwiseExposure).filter(
                        PairwiseExposure.etf == etf1,
                        PairwiseExposure.ticker1 == min(t1, t2),
                        PairwiseExposure.ticker2 == max(t1, t2)
                    ).first()

                    pair2 = session.query(PairwiseExposure).filter(
                        PairwiseExposure.etf == etf2,
                        PairwiseExposure.ticker1 == min(t1, t2),
                        PairwiseExposure.ticker2 == max(t1, t2)
                    ).first()

                    # Use whichever is available
                    if pair1:
                        cross_matrix[i, j] = pair1.correlation
                    elif pair2:
                        cross_matrix[i, j] = pair2.correlation
                    else:
                        # Default to 0 if no data
                        cross_matrix[i, j] = 0.0

        logger.info(f"Cross-matrix built: {cross_matrix.shape}")

        return cross_matrix


def get_comparison_engine() -> ETFComparisonEngine:
    """Get comparison engine instance"""
    return ETFComparisonEngine()
