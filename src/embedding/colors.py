"""Color standardization utils using CLIP embeddings."""

from typing import Sequence

import numpy as np
import typer

from src.database.manager import Manager
from src.database.schemas import upsert_to_colors
from src.embedding.clip import ClipEmbedder, get_clip_embedder

app = typer.Typer()

clip_embedder: ClipEmbedder = get_clip_embedder()

CORPUS_COLORS = [
    # Neutrals
    "white",
    "ivory",
    "light gray",
    "gray",
    "dark gray",
    "black",
    "silver",
    "gold",
    "beige",
    "tan",
    "brown",
    "dark brown",
    # Reds / Pinks
    "light pink",
    "pink",
    "hot pink",
    "red",
    "dark red",
    "burgundy",
    # Oranges / Peaches
    "peach",
    "coral",
    "orange",
    "rust",
    # Yellows
    "light yellow",
    "yellow",
    "mustard",
    # Greens
    "mint",
    "light green",
    "green",
    "olive",
    "dark green",
    # Blues
    "light blue",
    "blue",
    "royal blue",
    "navy",
    # Purples
    "lavender",
    "purple",
    "dark purple",
    # Extras
    "multicolor",
]

corpus_color_queries: list[str] = [
    f"A piece of clothing in {color} color." for color in CORPUS_COLORS
]
corpus_features = clip_embedder.encode_texts(corpus_color_queries)


@app.command("embed")
def embed() -> None:
    """Calculate database color embeddings and store mappings."""
    typer.echo("Embedding colors")

    with Manager() as db:
        sql = """SELECT DISTINCT color FROM item.attributes;"""
        db.cursor.execute(sql)
        records = db.cursor.fetchall()

    query_colors: list[str] = [record["color"] for record in records]
    query_color_queries = [
        f"A piece of clothing in {color} color." for color in query_colors
    ]

    query_features = clip_embedder.encode_texts(query_color_queries)
    similarity_matrix = np.dot(corpus_features, query_features.T)

    matches: dict[str, str] = {}
    for i, query_color in enumerate(query_colors):
        best_idx = np.argmax(similarity_matrix[:, i])
        matches[query_color] = CORPUS_COLORS[best_idx]

    payload = [
        {
            "source_color": source_color,
            "target_color": target_color,
        }
        for source_color, target_color in matches.items()
    ]

    typer.echo("Upserting colors")
    upsert_to_colors(payload)


def zero_shot_color(color: str) -> tuple[str, str]:
    """Return the corpus color that best matches the query color."""
    query = f"A piece of clothing in {color} color."
    query_features = clip_embedder.encode_texts([query])[0]
    similarity_matrix = np.dot(corpus_features, query_features.T)
    best_idx = np.argmax(similarity_matrix)
    return color, CORPUS_COLORS[best_idx]


if __name__ == "__main__":
    app()
