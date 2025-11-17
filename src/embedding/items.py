"""Embedding pipeline for catalog items."""

from itertools import batched
from typing import Any

import typer
from tqdm import tqdm

from src.config import IMAGE_DIR
from src.database.manager import Manager
from src.database.schemas import upsert_to_features
from src.embedding.clip import ClipEmbedder, get_clip_embedder
from src.embedding.st import STEmbedder, get_st_embedder

app = typer.Typer()

clip_embedder: ClipEmbedder = get_clip_embedder()
st_embedder: STEmbedder = get_st_embedder()


@app.command("embed")
def embed(batch_size: int = 2048) -> None:
    """Backfill missing CLIP/ST corpus embeddings for catalog items."""
    typer.echo("Embedding items")

    with Manager() as db:
        db.cursor.execute(
            """
                SELECT A.sku, A.image1, A.image2, A.texts
                FROM item.attributes AS A
                LEFT JOIN item.features AS F 
                    ON F.sku = A.sku
                WHERE 1=1
                    AND (
                        F.clip_image1 IS NULL
                        OR F.clip_image2 IS NULL
                        OR F.clip_text IS NULL
                        OR F.st_text IS NULL
                    )
                ;
            """
        )
        records: list[dict[str, Any]] = db.cursor.fetchall()

    total_records = len(records)
    total_batches = (total_records + batch_size - 1) // batch_size

    for record_batch in tqdm(
        batched(records, batch_size),
        desc="Embedding batches",
        total=total_batches,
    ):
        sku_payload: dict[str, dict[str, Any]] = {
            record["sku"]: {"sku": record["sku"]} for record in record_batch
        }

        text_jobs: list[tuple[str, Any]] = []
        for record in record_batch:
            texts = record.get("texts")
            text_jobs.append((record["sku"], texts))

        text_list = [text for _, text in text_jobs if text]
        if text_list:
            st_vectors = st_embedder.encode_texts(text_list, 256)
            text_idx = 0
            for sku, text in text_jobs:
                if text:
                    sku_payload[sku]["st_text"] = st_vectors[text_idx]
                    text_idx += 1

            clip_text_vectors = clip_embedder.encode_texts(text_list, 256)
            text_idx = 0
            for sku, text in text_jobs:
                if text:
                    sku_payload[sku]["clip_text"] = clip_text_vectors[text_idx]
                    text_idx += 1

        image_keys: list[tuple[str, str]] = []
        image_paths: list[str] = []
        for record in record_batch:
            sku = record["sku"]
            sku_dir = IMAGE_DIR / sku
            path1 = sku_dir / "image1.jpeg"
            if path1.exists():
                image_paths.append(str(path1))
                image_keys.append((sku, "image1"))
            path2 = sku_dir / "image2.jpeg"
            if path2.exists():
                image_paths.append(str(path2))
                image_keys.append((sku, "image2"))

        if image_paths:
            image_vectors = clip_embedder.encode_images(image_paths, 128)
            for (sku, key), vector in zip(image_keys, image_vectors):
                sku_payload[sku][f"clip_{key}"] = vector

        payload = list(sku_payload.values())
        # Use larger batch size for DB operations to reduce commit overhead
        upsert_to_features(payload, batch_size=256)

    typer.echo("All batches processed successfully!")


if __name__ == "__main__":
    app()
