"""Embedding model wrapper for HuggingFace transformers"""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from backend.utils.config import get_config
from backend.utils.logger import setup_logger
from backend.utils.cost_monitor import get_cost_monitor

logger = setup_logger("embeddings")


class EmbeddingModel:
    """Wrapper for HuggingFace embedding models"""

    def __init__(self):
        self.config = get_config().embedding
        self.cost_monitor = get_cost_monitor()
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self):
        """Load the embedding model"""
        logger.info(f"Loading embedding model: {self.config.model_name}")
        logger.info(f"Using device: {self.device}")

        try:
            self.model = SentenceTransformer(
                self.config.model_name,
                device=self.device
            )
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def embed(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for text(s)

        Args:
            texts: Single text string or list of texts
            batch_size: Batch size for processing (uses config default if None)
            show_progress: Show progress bar

        Returns:
            Single embedding or list of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
            return_single = True
        else:
            return_single = False

        if batch_size is None:
            batch_size = self.config.batch_size

        logger.info(f"Generating embeddings for {len(texts)} texts")

        try:
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,  # Normalize for cosine similarity
            )

            # Log cost (HuggingFace local models are free, but track for monitoring)
            total_tokens = sum(len(text.split()) * 1.3 for text in texts)  # Rough estimate
            self.cost_monitor.log_embedding_cost(
                provider="huggingface",
                model=self.config.model_name,
                num_texts=len(texts),
                tokens=int(total_tokens),
            )

            logger.info(f"Generated {len(embeddings)} embeddings")

            # Convert to list format
            embeddings_list = embeddings.tolist()

            if return_single:
                return embeddings_list[0]
            return embeddings_list

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a search query

        Args:
            query: Query text

        Returns:
            Query embedding
        """
        return self.embed(query)

    def embed_documents(
        self,
        documents: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> List[List[float]]:
        """Generate embeddings for multiple documents

        Args:
            documents: List of document texts
            batch_size: Batch size for processing
            show_progress: Show progress bar

        Returns:
            List of embeddings
        """
        return self.embed(
            documents,
            batch_size=batch_size,
            show_progress=show_progress
        )

    def similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between two embeddings

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Similarity score (0-1)
        """
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)

        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        return float(similarity)

    def get_dimension(self) -> int:
        """Get the embedding dimension

        Returns:
            Embedding dimension
        """
        return self.model.get_sentence_embedding_dimension()

    def get_model_info(self) -> dict:
        """Get model information

        Returns:
            Dictionary with model info
        """
        return {
            "model_name": self.config.model_name,
            "dimension": self.get_dimension(),
            "device": self.device,
            "max_length": self.config.max_length,
        }


# Global embedding model instance
_embedding_model: Optional[EmbeddingModel] = None


def get_embedding_model() -> EmbeddingModel:
    """Get the global embedding model instance"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model
