from sentence_transformers import SentenceTransformer


from src.config import ST_MODEL_NAME


class STEmbedder:

    def __init__(self):
        self.model = SentenceTransformer(ST_MODEL_NAME)

    def encode_texts(self, texts, batch_size=32):
        if isinstance(texts, str):
            texts = [texts]

        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )


def get_st_embedder():
    return STEmbedder()
