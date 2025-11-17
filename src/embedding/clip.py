"""CLIP model wrapper for batched embeddings of text and images."""

from typing import Iterable, Sequence

from PIL import Image
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import CLIP_MODEL_NAME, CLIP_PRETRAINED


class _ImageDataset(Dataset[Image.Image]):
    """Dataset wrapper that normalizes image inputs."""

    def __init__(self, images: Sequence[Image.Image | str]):
        """Store image references."""
        self.images = images

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.images)

    def __getitem__(self, idx: int) -> Image.Image:
        """Load and normalize a single image."""
        item = self.images[idx]
        if isinstance(item, Image.Image):
            return item.convert("RGB")
        if isinstance(item, str):
            return Image.open(item).convert("RGB")
        msg = "Unsupported image input type"
        raise TypeError(msg)


class _TextDataset(Dataset[str]):
    """Dataset wrapper for text batches."""

    def __init__(self, texts: Iterable[str]):
        """Capture the provided text sequence."""
        self.texts = list(texts)

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        """Return a single text entry."""
        return self.texts[idx]


class CollateImages:
    """Collate function that applies CLIP preprocessing."""

    def __init__(self, preprocess):
        """Store the preprocessing callable."""
        self.preprocess = preprocess

    def __call__(self, batch: Sequence[Image.Image]) -> Tensor:
        """Convert a batch of images into a tensor."""
        return torch.stack([self.preprocess(img) for img in batch])


class CollateTexts:
    """Collate function that tokenizes text batches."""

    def __init__(self, tokenizer):
        """Store the tokenizer callable."""
        self.tokenizer = tokenizer

    def __call__(self, batch: Sequence[str]) -> Tensor:
        """Tokenize a batch of strings."""
        return self.tokenizer(batch)


class ClipEmbedder:
    """Convenience wrapper around open_clip encoders."""

    def __init__(
        self,
        model_name: str,
        pretrained: str,
        device: str | None = None,
    ) -> None:
        """Load the CLIP model, preprocessors, and tokenizer."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.model = model.to(self.device).eval()

    def encode_images(
        self,
        images: Image.Image | str | Sequence[Image.Image | str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encode image inputs and return normalized embeddings."""
        if isinstance(images, (str, Image.Image)):
            images = [images]

        dataset = _ImageDataset(images)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=CollateImages(self.preprocess),
        )

        embeds = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Embedding images", total=len(loader)):
                batch = batch.to(self.device)
                emb = self.model.encode_image(batch)
                emb = F.normalize(emb, dim=-1)
                embeds.append(emb.cpu())

        return torch.cat(embeds, dim=0).numpy()

    def encode_texts(
        self,
        texts: str | Sequence[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encode text inputs and return normalized embeddings."""
        if isinstance(texts, str):
            texts = [texts]

        dataset = _TextDataset(texts)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=CollateTexts(self.tokenizer),
        )

        embeds = []
        with torch.no_grad():
            for tokens in tqdm(loader, desc="Embedding texts", total=len(loader)):
                tokens = tokens.to(self.device)
                emb = self.model.encode_text(tokens)
                emb = F.normalize(emb, dim=-1)
                embeds.append(emb.cpu())

        return torch.cat(embeds, dim=0).numpy()


def get_clip_embedder() -> ClipEmbedder:
    """Instantiate a ClipEmbedder using config defaults."""
    embedder = ClipEmbedder(CLIP_MODEL_NAME, CLIP_PRETRAINED)
    return embedder
