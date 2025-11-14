import numpy as np
import typer

from src.database.manager import Manager
from src.embedding.clip import get_clip_embedder

app = typer.Typer()

clip_embedder = get_clip_embedder()

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

corpus_color_queries = []
for color in CORPUS_COLORS:
    query = f"A piece of clothing in {color} color."
    corpus_color_queries.append(query)
corpus_features = clip_embedder.encode_texts(corpus_color_queries)


def embed_corpus_colors():

    with Manager() as db:
        sql = """SELECT DISTINCT color FROM item.attributes;"""
        db.cursor.execute(sql)
        records = db.cursor.fetchall()

    query_colors = [record["color"] for record in records]
    query_color_queries = []
    for color in query_colors:
        query = f"A piece of clothing in {color} color."
        query_color_queries.append(query)

    query_features = clip_embedder.encode_texts(query_color_queries)
    similarity_matrix = np.dot(corpus_features, query_features.T)

    matches = {}
    for i, query_color in enumerate(query_colors):
        best_idx = np.argmax(similarity_matrix[:, i])
        matches[query_color] = CORPUS_COLORS[best_idx]

    return matches


def zero_shot_color(color):
    query = f"A piece of clothing in {color} color."
    query_features = clip_embedder.encode_texts([query])[0]
    similarity_matrix = np.dot(corpus_features, query_features.T)
    best_idx = np.argmax(similarity_matrix)
    return color, CORPUS_COLORS[best_idx]
