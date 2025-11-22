"""Evaluation utilities for fine-tuned embeddings"""

from typing import List, Tuple, Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sentence_transformers import SentenceTransformer
import torch

from backend.utils.logger import setup_logger

logger = setup_logger("ft_evaluation")


class FineTuningEvaluator:
    """Evaluate fine-tuned embedding models"""

    def __init__(self, model: SentenceTransformer):
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        emb1 = self.model.encode(text1, convert_to_numpy=True)
        emb2 = self.model.encode(text2, convert_to_numpy=True)

        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        return float(similarity)

    def evaluate_pairs(
        self,
        pairs: List[Tuple[str, str, int]],
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Evaluate model on pairs with binary classification metrics

        Args:
            pairs: List of (text1, text2, label) tuples
            threshold: Similarity threshold for classification

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating {len(pairs)} pairs with threshold={threshold}")

        similarities = []
        labels = []

        for text1, text2, label in pairs:
            sim = self.compute_similarity(text1, text2)
            similarities.append(sim)
            labels.append(label)

        similarities = np.array(similarities)
        labels = np.array(labels)

        # Convert similarities to binary predictions
        predictions = (similarities >= threshold).astype(int)

        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', zero_division=0
        )

        # ROC AUC (using similarities as scores)
        try:
            auc = roc_auc_score(labels, similarities)
        except ValueError:
            auc = 0.0

        # Calculate mean similarities for positive and negative pairs
        pos_mask = labels == 1
        neg_mask = labels == 0

        mean_pos_sim = similarities[pos_mask].mean() if pos_mask.sum() > 0 else 0
        mean_neg_sim = similarities[neg_mask].mean() if neg_mask.sum() > 0 else 0

        # Separation (how well the model separates positive and negative)
        separation = mean_pos_sim - mean_neg_sim

        results = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "auc_roc": float(auc),
            "mean_positive_similarity": float(mean_pos_sim),
            "mean_negative_similarity": float(mean_neg_sim),
            "separation": float(separation),
            "threshold": threshold,
            "num_pairs": len(pairs),
            "num_positive": int(pos_mask.sum()),
            "num_negative": int(neg_mask.sum()),
        }

        logger.info(f"Evaluation results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        logger.info(f"  AUC-ROC: {auc:.4f}")
        logger.info(f"  Separation: {separation:.4f}")

        return results

    def find_optimal_threshold(
        self,
        pairs: List[Tuple[str, str, int]],
        metric: str = "f1"
    ) -> Tuple[float, Dict[str, Any]]:
        """Find optimal similarity threshold

        Args:
            pairs: Evaluation pairs
            metric: Metric to optimize ("f1", "accuracy", "precision", "recall")

        Returns:
            (optimal_threshold, best_results)
        """
        logger.info(f"Finding optimal threshold based on {metric}")

        thresholds = np.arange(0.1, 1.0, 0.05)
        best_score = -1
        best_threshold = 0.5
        best_results = {}

        for threshold in thresholds:
            results = self.evaluate_pairs(pairs, threshold=threshold)

            if results[metric] > best_score:
                best_score = results[metric]
                best_threshold = threshold
                best_results = results

        logger.info(f"Optimal threshold: {best_threshold:.2f} ({metric}={best_score:.4f})")

        return best_threshold, best_results

    def evaluate_retrieval(
        self,
        query_pairs: List[Tuple[str, List[str], List[int]]],
        top_k: int = 5
    ) -> Dict[str, float]:
        """Evaluate retrieval performance

        Args:
            query_pairs: List of (query, documents, relevance_labels)
            top_k: Number of top results to consider

        Returns:
            Retrieval metrics
        """
        logger.info(f"Evaluating retrieval on {len(query_pairs)} queries")

        precisions_at_k = []
        recalls_at_k = []
        mrrs = []  # Mean Reciprocal Rank

        for query, documents, labels in query_pairs:
            # Encode query
            query_emb = self.model.encode(query, convert_to_numpy=True)

            # Encode documents
            doc_embs = self.model.encode(documents, convert_to_numpy=True)

            # Calculate similarities
            similarities = np.array([
                np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
                for doc_emb in doc_embs
            ])

            # Get top-k indices
            top_k_indices = np.argsort(similarities)[::-1][:top_k]

            # Calculate precision@k
            relevant_retrieved = sum(labels[i] for i in top_k_indices)
            precision_at_k = relevant_retrieved / top_k
            precisions_at_k.append(precision_at_k)

            # Calculate recall@k
            total_relevant = sum(labels)
            recall_at_k = relevant_retrieved / total_relevant if total_relevant > 0 else 0
            recalls_at_k.append(recall_at_k)

            # Calculate MRR
            for rank, idx in enumerate(top_k_indices, 1):
                if labels[idx] == 1:
                    mrrs.append(1.0 / rank)
                    break
            else:
                mrrs.append(0.0)

        results = {
            f"precision@{top_k}": float(np.mean(precisions_at_k)),
            f"recall@{top_k}": float(np.mean(recalls_at_k)),
            "mrr": float(np.mean(mrrs)),
            "num_queries": len(query_pairs)
        }

        logger.info(f"Retrieval results:")
        logger.info(f"  Precision@{top_k}: {results[f'precision@{top_k}']:.4f}")
        logger.info(f"  Recall@{top_k}: {results[f'recall@{top_k}']:.4f}")
        logger.info(f"  MRR: {results['mrr']:.4f}")

        return results


def compare_embeddings(
    base_model: SentenceTransformer,
    fine_tuned_model: SentenceTransformer,
    test_pairs: List[Tuple[str, str, int]]
) -> Dict[str, Any]:
    """Compare base vs fine-tuned embeddings

    Args:
        base_model: Base model
        fine_tuned_model: Fine-tuned model
        test_pairs: Test pairs

    Returns:
        Comparison results
    """
    logger.info("Comparing base vs fine-tuned embeddings")

    # Evaluate base model
    base_evaluator = FineTuningEvaluator(base_model)
    base_threshold, base_results = base_evaluator.find_optimal_threshold(test_pairs)

    # Evaluate fine-tuned model
    ft_evaluator = FineTuningEvaluator(fine_tuned_model)
    ft_threshold, ft_results = ft_evaluator.find_optimal_threshold(test_pairs)

    # Calculate improvements
    improvements = {}
    for metric in ["accuracy", "f1_score", "auc_roc", "separation"]:
        improvement = ft_results[metric] - base_results[metric]
        improvement_pct = (improvement / base_results[metric] * 100) if base_results[metric] > 0 else 0
        improvements[metric] = {
            "absolute": improvement,
            "percentage": improvement_pct
        }

    comparison = {
        "base_model": {
            "threshold": base_threshold,
            "results": base_results
        },
        "fine_tuned_model": {
            "threshold": ft_threshold,
            "results": ft_results
        },
        "improvements": improvements
    }

    logger.info("Comparison complete")
    logger.info(f"F1 improvement: {improvements['f1_score']['absolute']:.4f} ({improvements['f1_score']['percentage']:.2f}%)")
    logger.info(f"Separation improvement: {improvements['separation']['absolute']:.4f}")

    return comparison
