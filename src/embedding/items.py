import typer

from src.config import IMAGE_DIR
from src.database.manager import Manager
from src.database.schemas import upsert_to_features
from src.embedding.clip import get_clip_embedder
from src.embedding.st import get_st_embedder

app = typer.Typer()

clip_embedder = get_clip_embedder()
st_embedder = get_st_embedder()


@app.command("embed")
def embed(batch_size: int = 128):
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
                LIMIT 10
                ;
            """
        )
        records = db.cursor.fetchall()

    sku_payload = {record["sku"]: {"sku": record["sku"]} for record in records}

    text_jobs = []
    for record in records:
        texts = record.get("texts")
        text_jobs.append((record["sku"], texts))

    text_vectors = st_embedder.encode_texts([text for _, text in text_jobs], batch_size)
    for (sku, _), vector in zip(text_jobs, text_vectors):
        sku_payload[sku]["st_text"] = vector

    text_vectors = clip_embedder.encode_texts(
        [text for _, text in text_jobs], batch_size
    )
    for (sku, _), vector in zip(text_jobs, text_vectors):
        sku_payload[sku]["clip_text"] = vector

    image_keys = []
    image_paths = []
    for record in records:
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

    image_vectors = clip_embedder.encode_images(image_paths, batch_size)
    for (sku, key), vector in zip(image_keys, image_vectors):
        sku_payload[sku][f"clip_{key}"] = vector

    payload = list(sku_payload.values())
    upsert_to_features(payload)


if __name__ == "__main__":
    app()
