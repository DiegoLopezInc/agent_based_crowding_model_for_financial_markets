"""Contrastive learning trainer for fine-tuning embeddings"""

from typing import List, Tuple, Optional
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import numpy as np

from backend.utils.config import get_config
from backend.utils.logger import setup_logger
from backend.utils.cost_monitor import get_cost_monitor

logger = setup_logger("fine_tuning_trainer")


class ContrastivePairDataset(Dataset):
    """Dataset for contrastive learning pairs"""

    def __init__(self, pairs: List[Tuple[str, str, int]]):
        """Initialize dataset

        Args:
            pairs: List of (text1, text2, label) tuples
                   label = 1 for similar (positive), 0 for dissimilar (negative)
        """
        self.examples = [
            InputExample(texts=[text1, text2], label=float(label))
            for text1, text2, label in pairs
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class ContrastiveEmbeddingTrainer:
    """Trainer for fine-tuning embeddings with contrastive learning"""

    def __init__(self):
        self.config = get_config()
        self.ft_config = self.config.embedding.fine_tuning.training
        self.cost_monitor = get_cost_monitor()

        # Device selection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Create output directories
        Path(self.ft_config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.ft_config.logging_dir).mkdir(parents=True, exist_ok=True)

    def create_model(self, base_model_name: Optional[str] = None) -> SentenceTransformer:
        """Create or load base model

        Args:
            base_model_name: Base model to fine-tune (default from config)

        Returns:
            SentenceTransformer model
        """
        if base_model_name is None:
            base_model_name = self.config.embedding.model_name

        logger.info(f"Loading base model: {base_model_name}")

        model = SentenceTransformer(base_model_name, device=self.device)

        logger.info(f"Model loaded on {self.device}")
        logger.info(f"Model max sequence length: {model.max_seq_length}")

        return model

    def create_loss_function(self, model: SentenceTransformer):
        """Create contrastive loss function

        Args:
            model: The sentence transformer model

        Returns:
            Loss function
        """
        if self.ft_config.distance_metric == "cosine":
            # Cosine similarity loss
            loss = losses.ContrastiveLoss(
                model=model,
                distance_metric=losses.SiameseDistanceMetric.COSINE_DISTANCE,
                margin=self.ft_config.margin
            )
        else:
            # Euclidean distance loss
            loss = losses.ContrastiveLoss(
                model=model,
                distance_metric=losses.SiameseDistanceMetric.EUCLIDEAN_DISTANCE,
                margin=self.ft_config.margin
            )

        logger.info(
            f"Created contrastive loss with {self.ft_config.distance_metric} "
            f"distance and margin={self.ft_config.margin}"
        )

        return loss

    def create_evaluator(
        self,
        eval_pairs: List[Tuple[str, str, int]],
        name: str = "eval"
    ) -> EmbeddingSimilarityEvaluator:
        """Create evaluator for validation

        Args:
            eval_pairs: Evaluation pairs
            name: Name for the evaluator

        Returns:
            Evaluator
        """
        sentences1 = [pair[0] for pair in eval_pairs]
        sentences2 = [pair[1] for pair in eval_pairs]
        scores = [float(pair[2]) for pair in eval_pairs]

        evaluator = EmbeddingSimilarityEvaluator(
            sentences1=sentences1,
            sentences2=sentences2,
            scores=scores,
            name=name,
            write_csv=True
        )

        logger.info(f"Created evaluator with {len(eval_pairs)} pairs")

        return evaluator

    def train(
        self,
        train_pairs: List[Tuple[str, str, int]],
        eval_pairs: Optional[List[Tuple[str, str, int]]] = None,
    ) -> SentenceTransformer:
        """Train the model with contrastive learning

        Args:
            train_pairs: Training pairs (text1, text2, label)
            eval_pairs: Optional evaluation pairs

        Returns:
            Fine-tuned model
        """
        logger.info("Starting fine-tuning")
        logger.info(f"Training pairs: {len(train_pairs)}")
        if eval_pairs:
            logger.info(f"Evaluation pairs: {len(eval_pairs)}")

        # Create model
        model = self.create_model()

        # Create dataset
        train_dataset = ContrastivePairDataset(train_pairs)

        # Create DataLoader
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.ft_config.batch_size
        )

        # Create loss
        train_loss = self.create_loss_function(model)

        # Create evaluator
        evaluator = None
        if eval_pairs:
            evaluator = self.create_evaluator(eval_pairs)

        # Calculate training steps
        steps_per_epoch = len(train_dataloader)
        total_steps = steps_per_epoch * self.ft_config.epochs

        logger.info(f"Training for {self.ft_config.epochs} epochs")
        logger.info(f"Steps per epoch: {steps_per_epoch}")
        logger.info(f"Total steps: {total_steps}")

        # Train
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=self.ft_config.epochs,
            evaluation_steps=self.ft_config.evaluation_steps,
            warmup_steps=self.ft_config.warmup_steps,
            output_path=self.ft_config.output_dir,
            save_best_model=True,
            optimizer_params={'lr': self.ft_config.learning_rate},
            checkpoint_path=self.ft_config.output_dir,
            checkpoint_save_steps=self.ft_config.save_steps,
            use_amp=torch.cuda.is_available(),  # Use automatic mixed precision if CUDA
        )

        logger.info("Fine-tuning completed")

        # Save final model
        final_model_path = self.config.embedding.fine_tuning.fine_tuned_model_path
        Path(final_model_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(final_model_path)

        logger.info(f"Saved fine-tuned model to {final_model_path}")

        return model

    def evaluate_model(
        self,
        model: SentenceTransformer,
        test_pairs: List[Tuple[str, str, int]]
    ) -> dict:
        """Evaluate model performance

        Args:
            model: Model to evaluate
            test_pairs: Test pairs

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating model on {len(test_pairs)} test pairs")

        evaluator = self.create_evaluator(test_pairs, name="test")

        # Evaluate
        score = evaluator(model, output_path=self.ft_config.logging_dir)

        logger.info(f"Test score: {score}")

        return {
            "test_score": score,
            "num_pairs": len(test_pairs)
        }

    def compare_models(
        self,
        base_model_name: str,
        fine_tuned_model_path: str,
        test_pairs: List[Tuple[str, str, int]]
    ) -> dict:
        """Compare base model vs fine-tuned model

        Args:
            base_model_name: Name of base model
            fine_tuned_model_path: Path to fine-tuned model
            test_pairs: Test pairs for comparison

        Returns:
            Comparison results
        """
        logger.info("Comparing base model vs fine-tuned model")

        # Load models
        base_model = SentenceTransformer(base_model_name, device=self.device)
        fine_tuned_model = SentenceTransformer(fine_tuned_model_path, device=self.device)

        # Evaluate both
        base_results = self.evaluate_model(base_model, test_pairs)
        ft_results = self.evaluate_model(fine_tuned_model, test_pairs)

        improvement = ft_results["test_score"] - base_results["test_score"]
        improvement_pct = (improvement / base_results["test_score"]) * 100

        results = {
            "base_model": {
                "name": base_model_name,
                "score": base_results["test_score"]
            },
            "fine_tuned_model": {
                "path": fine_tuned_model_path,
                "score": ft_results["test_score"]
            },
            "improvement": improvement,
            "improvement_percentage": improvement_pct,
            "num_test_pairs": len(test_pairs)
        }

        logger.info(f"Base model score: {base_results['test_score']:.4f}")
        logger.info(f"Fine-tuned model score: {ft_results['test_score']:.4f}")
        logger.info(f"Improvement: {improvement:.4f} ({improvement_pct:.2f}%)")

        return results


def get_trainer() -> ContrastiveEmbeddingTrainer:
    """Get trainer instance"""
    return ContrastiveEmbeddingTrainer()
