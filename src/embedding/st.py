"""Sentence Transformer wrapper for normalized text embeddings."""

from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import ST_MODEL_NAME


class STEmbedder:
    """Encapsulates a SentenceTransformer encoder."""

    def __init__(self) -> None:
        """Load the configured SentenceTransformer model."""
        self.model = SentenceTransformer(ST_MODEL_NAME)

    def encode_texts(
        self,
        texts: str | Sequence[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encode text inputs into normalized embeddings."""
        if isinstance(texts, str):
            texts = [texts]

        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )


def get_st_embedder() -> STEmbedder:
    """Instantiate an STEmbedder with default config."""
    return STEmbedder()
