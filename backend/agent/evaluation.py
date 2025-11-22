"""Agent evaluation system with quantitative reward function"""

import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

from backend.utils.config import get_config
from backend.utils.logger import setup_logger
from backend.database.connection import get_db_manager
from backend.database.repository import AgentEvaluationRepository

logger = setup_logger("evaluation")


class AgentEvaluator:
    """Evaluate agent responses with quantitative metrics"""

    def __init__(self):
        self.config = get_config().evaluation
        self.reward_config = self.config.reward_function
        self.db_manager = get_db_manager()

    def calculate_relevance_score(
        self,
        question: str,
        answer: str,
        retrieved_documents: List[Dict[str, Any]],
    ) -> float:
        """Calculate relevance score (0-1)

        Measures how relevant the answer is to the question and retrieved documents.

        Args:
            question: User question
            answer: Agent answer
            retrieved_documents: Documents used to generate answer

        Returns:
            Relevance score (0-1)
        """
        score = 0.0

        # 1. Check if answer addresses key terms from question (0.3 weight)
        question_terms = set(re.findall(r'\w+', question.lower()))
        answer_terms = set(re.findall(r'\w+', answer.lower()))

        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'are', 'was', 'were'}
        question_terms = question_terms - stop_words
        answer_terms = answer_terms - stop_words

        if question_terms:
            term_overlap = len(question_terms & answer_terms) / len(question_terms)
            score += term_overlap * 0.3

        # 2. Check if answer uses retrieved documents (0.4 weight)
        if retrieved_documents:
            doc_terms = set()
            for doc in retrieved_documents:
                doc_terms.update(re.findall(r'\w+', doc.get('content', '').lower()))

            doc_terms = doc_terms - stop_words
            if doc_terms:
                doc_usage = len(answer_terms & doc_terms) / len(doc_terms)
                score += min(doc_usage * 2, 1.0) * 0.4  # Scale up and cap at 1.0

        # 3. Check answer length appropriateness (0.3 weight)
        # Too short or too long suggests poor relevance
        answer_length = len(answer.split())
        if 50 <= answer_length <= 500:
            score += 0.3
        elif 20 <= answer_length < 50 or 500 < answer_length <= 1000:
            score += 0.15
        # Else: 0 points

        return min(score, 1.0)

    def calculate_accuracy_score(
        self,
        answer: str,
        retrieved_documents: List[Dict[str, Any]],
    ) -> float:
        """Calculate accuracy score (0-1)

        Measures factual accuracy and proper use of data.

        Args:
            answer: Agent answer
            retrieved_documents: Documents used

        Returns:
            Accuracy score (0-1)
        """
        score = 0.0

        # 1. Check for numerical data usage (0.4 weight)
        numbers = re.findall(r'\d+\.?\d*%?', answer)
        if numbers:
            # More numbers suggest data-driven answer
            num_count = len(numbers)
            score += min(num_count / 10, 1.0) * 0.4

        # 2. Check for ticker symbols (0.2 weight)
        # Common stock ticker pattern
        tickers = re.findall(r'\b[A-Z]{2,5}\b', answer)
        if tickers:
            score += min(len(tickers) / 5, 1.0) * 0.2

        # 3. Check for dates/timeframes (0.2 weight)
        # Matches patterns like "2024", "Q3 2024", "Dec 2024"
        dates = re.findall(r'\b(20\d{2}|Q[1-4]\s*20\d{2}|[A-Z][a-z]+\s+\d{1,2},?\s*20\d{2})\b', answer)
        if dates:
            score += min(len(dates) / 3, 1.0) * 0.2

        # 4. Check if answer contradicts retrieved documents (0.2 weight)
        # Simple heuristic: if answer is very different from documents, may be inaccurate
        if retrieved_documents:
            doc_text = " ".join([doc.get('content', '') for doc in retrieved_documents[:3]])
            answer_terms = set(re.findall(r'\w+', answer.lower()))
            doc_terms = set(re.findall(r'\w+', doc_text.lower()))

            if doc_terms:
                overlap = len(answer_terms & doc_terms) / len(answer_terms) if answer_terms else 0
                score += min(overlap * 1.5, 1.0) * 0.2

        return min(score, 1.0)

    def calculate_completeness_score(
        self,
        question: str,
        answer: str,
        retrieved_documents: List[Dict[str, Any]],
    ) -> float:
        """Calculate completeness score (0-1)

        Measures how thoroughly the answer addresses the question.

        Args:
            question: User question
            answer: Agent answer
            retrieved_documents: Documents retrieved

        Returns:
            Completeness score (0-1)
        """
        score = 0.0

        # 1. Answer length relative to question complexity (0.3 weight)
        question_length = len(question.split())
        answer_length = len(answer.split())

        # Complex questions (longer) should get longer answers
        expected_length = max(100, question_length * 10)
        length_ratio = min(answer_length / expected_length, 1.5)
        score += min(length_ratio, 1.0) * 0.3

        # 2. Number of distinct points/sections (0.3 weight)
        # Count bullet points, numbered lists, or paragraphs
        bullet_points = len(re.findall(r'[\n•\-\*]\s*', answer))
        numbered_points = len(re.findall(r'\n\d+\.', answer))
        paragraphs = len(re.findall(r'\n\n', answer)) + 1

        distinct_points = max(bullet_points, numbered_points, paragraphs)
        score += min(distinct_points / 5, 1.0) * 0.3

        # 3. Coverage of retrieved documents (0.2 weight)
        if retrieved_documents:
            # Check if multiple documents are referenced
            unique_tickers = set([doc.get('ticker') for doc in retrieved_documents if doc.get('ticker')])
            if unique_tickers:
                # Check how many tickers are mentioned in answer
                mentioned_tickers = sum(1 for ticker in unique_tickers if ticker in answer)
                ticker_coverage = mentioned_tickers / len(unique_tickers)
                score += ticker_coverage * 0.2

        # 4. Presence of caveats/limitations (0.2 weight)
        # Good answers acknowledge limitations
        caveat_indicators = [
            'however', 'but', 'although', 'limited', 'note that',
            'it\'s important', 'keep in mind', 'unavailable', 'missing'
        ]
        has_caveats = any(indicator in answer.lower() for indicator in caveat_indicators)
        if has_caveats:
            score += 0.2

        return min(score, 1.0)

    def calculate_data_usage_score(
        self,
        answer: str,
        retrieved_documents: List[Dict[str, Any]],
    ) -> float:
        """Calculate data usage score (0-1)

        Measures how well quantitative data is used.

        Args:
            answer: Agent answer
            retrieved_documents: Documents retrieved

        Returns:
            Data usage score (0-1)
        """
        score = 0.0

        # 1. Presence of specific numbers (0.3 weight)
        numbers = re.findall(r'\$?\d+\.?\d*[MBK%]?', answer)
        if numbers:
            score += min(len(numbers) / 8, 1.0) * 0.3

        # 2. Comparisons with numbers (0.3 weight)
        comparison_patterns = [
            r'\d+%\s*(higher|lower|more|less|increase|decrease)',
            r'(up|down)\s+\d+%',
            r'(beat|missed?|exceeded?)\s+(estimates?|expectations?)',
            r'vs\.?\s+\$?\d+',
        ]
        comparisons = sum(len(re.findall(pattern, answer, re.IGNORECASE))
                         for pattern in comparison_patterns)
        if comparisons:
            score += min(comparisons / 3, 1.0) * 0.3

        # 3. Financial metrics (0.2 weight)
        metrics = [
            'EPS', 'P/E', 'market cap', 'revenue', 'growth', 'margin',
            'valuation', 'earnings', 'profit', 'ROI', 'ROE'
        ]
        metric_count = sum(1 for metric in metrics if metric.lower() in answer.lower())
        if metric_count:
            score += min(metric_count / 5, 1.0) * 0.2

        # 4. Proper formatting of financial data (0.2 weight)
        # Check for proper formats like $123.45M, 15.3%, Q3 2024
        formatted_numbers = re.findall(r'\$\d+\.?\d*[MBK]?|\d+\.?\d+%|Q[1-4]\s*\d{4}', answer)
        if formatted_numbers:
            score += min(len(formatted_numbers) / 5, 1.0) * 0.2

        return min(score, 1.0)

    def calculate_reward(
        self,
        question: str,
        answer: str,
        retrieved_documents: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Calculate overall reward score

        Uses weighted combination of metrics.

        Args:
            question: User question
            answer: Agent answer
            retrieved_documents: Documents retrieved

        Returns:
            Dictionary with individual scores and total reward
        """
        # Calculate individual scores
        relevance = self.calculate_relevance_score(question, answer, retrieved_documents)
        accuracy = self.calculate_accuracy_score(answer, retrieved_documents)
        completeness = self.calculate_completeness_score(question, answer, retrieved_documents)
        data_usage = self.calculate_data_usage_score(answer, retrieved_documents)

        # Calculate weighted reward
        reward = (
            relevance * self.reward_config.relevance_weight +
            accuracy * self.reward_config.accuracy_weight +
            completeness * self.reward_config.completeness_weight +
            data_usage * (1.0 - sum([
                self.reward_config.relevance_weight,
                self.reward_config.accuracy_weight,
                self.reward_config.completeness_weight,
            ]))
        )

        scores = {
            "relevance_score": round(relevance, 3),
            "accuracy_score": round(accuracy, 3),
            "completeness_score": round(completeness, 3),
            "data_usage_score": round(data_usage, 3),
            "reward_score": round(reward, 3),
        }

        logger.info(f"Evaluation scores: {scores}")
        return scores

    def evaluate_and_store(
        self,
        query_id: int,
        question: str,
        answer: str,
        retrieved_documents: List[Dict[str, Any]],
        feedback: Optional[str] = None,
    ) -> Dict[str, float]:
        """Evaluate and store results in database

        Args:
            query_id: Query history ID
            question: User question
            answer: Agent answer
            retrieved_documents: Documents retrieved
            feedback: Optional human feedback

        Returns:
            Evaluation scores
        """
        scores = self.calculate_reward(question, answer, retrieved_documents)

        # Store in database
        with self.db_manager.get_session() as session:
            AgentEvaluationRepository.create_evaluation(
                session=session,
                query_id=query_id,
                relevance_score=scores["relevance_score"],
                accuracy_score=scores["accuracy_score"],
                completeness_score=scores["completeness_score"],
                reward_score=scores["reward_score"],
                feedback=feedback,
            )

        logger.info(f"Stored evaluation for query {query_id}")
        return scores

    def get_average_scores(self) -> Dict[str, float]:
        """Get average evaluation scores across all queries

        Returns:
            Dictionary with average scores
        """
        with self.db_manager.get_session() as session:
            return AgentEvaluationRepository.get_average_scores(session)


def get_evaluator() -> AgentEvaluator:
    """Get agent evaluator instance"""
    return AgentEvaluator()
