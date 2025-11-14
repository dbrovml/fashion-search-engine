from PIL import Image
import open_clip
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import CLIP_MODEL_NAME, CLIP_PRETRAINED


class _ImageDataset(Dataset):

    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = self.images[idx]
        if isinstance(item, Image.Image):
            return item.convert("RGB")
        if isinstance(item, str):
            return Image.open(item).convert("RGB")


class _TextDataset(Dataset):

    def __init__(self, texts):
        self.texts = list(texts)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


class CollateImages:
    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, batch):
        return torch.stack([self.preprocess(img) for img in batch])


class CollateTexts:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        return self.tokenizer(batch)


class ClipEmbedder:

    def __init__(self, model_name, pretrained, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.model = model.to(self.device).eval()

    def encode_images(self, images, batch_size=32):
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

    def encode_texts(self, texts, batch_size=32):
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


def get_clip_embedder():
    embedder = ClipEmbedder(CLIP_MODEL_NAME, CLIP_PRETRAINED)
    return embedder
