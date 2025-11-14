from itertools import batched

import numpy as np
from psycopg2.extras import execute_values
from tqdm import tqdm
import typer

from src.database.manager import Manager

app = typer.Typer()


ddl = f"""
    CREATE SCHEMA IF NOT EXISTS item;

    CREATE TABLE IF NOT EXISTS item.attributes (
        created         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        sku             VARCHAR(100) PRIMARY KEY,
        title           TEXT NOT NULL,
        brand           VARCHAR(100),
        category        VARCHAR(100),
        price           DECIMAL(9, 2),
        color           VARCHAR(100),
        url             TEXT,
        image1          TEXT,
        image2          TEXT,
        text1           TEXT,
        text2           TEXT,
        text3           TEXT,
        texts           TEXT
    );

    CREATE TABLE IF NOT EXISTS item.features (
        created         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        sku             VARCHAR(100) PRIMARY KEY,
        clip_image1     vector(512),
        clip_image2     vector(512),
        clip_text       vector(512),
        bert_text       vector(384)
    );

    CREATE TABLE IF NOT EXISTS item.colors (
        source_color    VARCHAR(100),
        target_color    VARCHAR(100)
    );

    CREATE INDEX IF NOT EXISTS idx_attributes_sku ON item.attributes(sku);
    CREATE INDEX IF NOT EXISTS idx_features_sku ON item.features(sku);

    CREATE INDEX IF NOT EXISTS features_idx_clip_image1 
    ON item.features
    USING ivfflat (clip_image1 vector_cosine_ops)
    WITH (lists = 50);

    CREATE INDEX IF NOT EXISTS features_idx_clip_image2 
    ON item.features
    USING ivfflat (clip_image2 vector_cosine_ops)
    WITH (lists = 50);

    CREATE INDEX IF NOT EXISTS features_idx_clip_text
    ON item.features
    USING ivfflat (clip_text vector_cosine_ops)
    WITH (lists = 50);

    CREATE INDEX IF NOT EXISTS features_idx_bert_text
    ON item.features
    USING ivfflat (bert_text vector_cosine_ops)
    WITH (lists = 50);
"""


@app.command("init-db")
def init_db():
    with Manager() as db:
        statements = [s.strip() for s in ddl.split(";") if s.strip()]
        for s in statements:
            db.cursor.execute(s)
        db.conn.commit()
    typer.echo("Database initialized successfully")


@app.command("drop-db")
def drop_db():
    with Manager() as db:
        db.cursor.execute("DROP SCHEMA IF EXISTS item CASCADE;")
        db.conn.commit()
    typer.echo("Database dropped successfully")


@app.command("upsert-attributes")
def upsert_to_attributes(records, batch_size=32):
    if not isinstance(records, list):
        records = [records]

    normalized = []
    for record in records:
        normalized.append(
            {
                "sku": record["sku"],
                "title": record.get("title"),
                "brand": record.get("brand"),
                "category": record.get("category"),
                "price": record.get("price"),
                "color": record.get("color"),
                "url": record.get("url"),
                "image1": record.get("image1"),
                "image2": record.get("image2"),
                "text1": record.get("text1"),
                "text2": record.get("text2"),
                "text3": record.get("text3"),
                "texts": record.get("texts"),
            }
        )

    upsert_sql = """
        INSERT INTO item.attributes
            (sku, title, brand, category, price, color,
            url, image1, image2, text1, text2, text3, texts)
        VALUES %s
        ON CONFLICT (sku) DO UPDATE SET
            updated         = CURRENT_TIMESTAMP,
            title           = COALESCE(EXCLUDED.title, item.attributes.title),
            brand           = COALESCE(EXCLUDED.brand, item.attributes.brand),
            category        = COALESCE(EXCLUDED.category, item.attributes.category),
            price           = COALESCE(EXCLUDED.price, item.attributes.price),
            color           = COALESCE(EXCLUDED.color, item.attributes.color),
            url             = COALESCE(EXCLUDED.url, item.attributes.url),
            image1          = COALESCE(EXCLUDED.image1, item.attributes.image1),
            image2          = COALESCE(EXCLUDED.image2, item.attributes.image2),
            text1           = COALESCE(EXCLUDED.text1, item.attributes.text1),
            text2           = COALESCE(EXCLUDED.text2, item.attributes.text2),
            text3           = COALESCE(EXCLUDED.text3, item.attributes.text3),
            texts           = COALESCE(EXCLUDED.texts, item.attributes.texts)
    """

    with Manager() as db:
        for batch in tqdm(
            batched(normalized, batch_size),
            desc="Upserting attributes",
            total=len(normalized) // batch_size,
        ):
            values = [
                (
                    row["sku"],
                    row["title"],
                    row["brand"],
                    row["category"],
                    row["price"],
                    row["color"],
                    row["url"],
                    row["image1"],
                    row["image2"],
                    row["text1"],
                    row["text2"],
                    row["text3"],
                    row["texts"],
                )
                for row in batch
            ]
            execute_values(db.cursor, upsert_sql, values)
            db.conn.commit()


@app.command("upsert-features")
def upsert_to_features(records, batch_size=32):
    if not isinstance(records, list):
        records = [records]

    prepared_records = []
    for record in records:
        prepared_record = {"sku": record["sku"]}
        for k, v in record.items():
            if k != "sku":
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                prepared_record[k] = v
        prepared_records.append(prepared_record)

    normalized = []
    for record in prepared_records:
        normalized.append(
            {
                "sku": record["sku"],
                "image1": record.get("image1"),
                "image2": record.get("image2"),
                "clip_text": record.get("clip_text"),
                "st_text": record.get("st_text"),
            }
        )

    upsert_sql = """
        INSERT INTO item.features
        (sku, image1, image2, clip_text, st_text)
        VALUES %s
        ON CONFLICT (sku) DO UPDATE SET
            updated = CURRENT_TIMESTAMP,
            image1  = COALESCE(EXCLUDED.image1, item.features.image1),
            image2  = COALESCE(EXCLUDED.image2, item.features.image2),
            clip_text  = COALESCE(EXCLUDED.clip_text, item.features.clip_text),
            st_text = COALESCE(EXCLUDED.st_text, item.features.st_text)
    """

    with Manager() as db:
        for batch in tqdm(
            batched(normalized, batch_size),
            desc="Upserting features",
            total=len(normalized) // batch_size,
        ):
            values = [
                (
                    row["sku"],
                    row["image1"],
                    row["image2"],
                    row["clip_text"],
                    row["st_text"],
                )
                for row in batch
            ]
            execute_values(db.cursor, upsert_sql, values)
            db.conn.commit()


if __name__ == "__main__":
    app()
