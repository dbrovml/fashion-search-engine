from typing import Optional

from pydantic import BaseModel

from src.database.manager import Manager
from src.embedding.clip import get_clip_embedder
from src.embedding.st import get_st_embedder


class ResultItem(BaseModel):
    sku: str
    title: str
    category: str
    color: str
    brand: str
    price: float
    image: str
    text: str
    url: str
    clip_score: Optional[float]
    st_score: Optional[float]
    clip_score1: Optional[float]
    clip_score2: Optional[float]
    score: float


class Engine:

    def __init__(self):
        self.clip_embedder = get_clip_embedder()
        self.st_embedder = get_st_embedder()

    def parse_filters(self, filters):
        filters_sql, params = "", []
        if filters:
            clauses, params = [], []
            if filters.brand is not None:
                clauses.append("A.brand = %s")
                params.append(filters.brand)
            if filters.category is not None:
                clauses.append("A.category = %s")
                params.append(filters.category)
            if filters.color is not None:
                target_color = filters.color
                clauses.append("C.target_color = %s")
                params.append(target_color)
            if filters.min_price is not None:
                clauses.append("A.price >= %s")
                params.append(filters.min_price)
            if filters.max_price is not None:
                clauses.append("A.price <= %s")
                params.append(filters.max_price)
            filters_sql = f" AND {' AND '.join(clauses)}" if clauses else ""
        return filters_sql, params

    def search_text(self, q_text, k=3, filters=None, clip_weight=0.3, st_weight=0.7):
        clip_emb = self.clip_embedder.encode_texts([q_text])[0]
        clip_v = f"[{','.join(f'{v:.7f}' for v in clip_emb)}]"

        st_emb = self.st_embedder.encode_texts([q_text])[0]
        st_v = f"[{','.join(f'{v:.7f}' for v in st_emb)}]"

        filters_sql, params = self.parse_filters(filters)

        search_query = f"""
            WITH scores AS (
                SELECT
                    F.sku,
                    1 - (F.clip_text <=> '{clip_v}'::vector) as clip_score,
                    1 - (F.st_text <=> '{st_v}'::vector) as st_score
                FROM 
                    item.features as F
                WHERE 1=1
                    AND F.clip_text IS NOT NULL
                    AND F.st_text is NOT NULL
            )
            , weighted AS (
                SELECT 
                    S.sku, S.clip_score, S.st_score,
                    (
                        S.clip_score * {clip_weight} + 
                        S.st_score * {st_weight}
                    ) AS score
                FROM 
                    scores AS S
            )
            SELECT
                W.sku, W.clip_score, W.st_score, W.score, A.title,
                A.brand, A.category, A.color, A.price,
                A.url, COALESCE(A.image2, A.image1) as image,
                COALESCE(A.text1, A.text2, A.text3) as text
            FROM
                weighted AS W
                INNER JOIN item.attributes AS A ON W.sku = A.sku
                LEFT JOIN item.colors AS C ON A.color = C.source_color
            WHERE 1=1
                {filters_sql}
            ORDER BY 
                W.score DESC
            LIMIT {k}
        """

        with Manager() as db:
            db.cursor.execute(search_query, params)
            results = db.cursor.fetchall()

        result_items = []
        for result in results:
            result_item = ResultItem(
                sku=result["sku"],
                title=result["title"],
                category=result["category"],
                color=result["color"],
                brand=result["brand"],
                price=result["price"],
                image=result["image"],
                text=result["text"],
                url=result["url"],
                clip_score=result["clip_score"],
                st_score=result["st_score"],
                clip_score1=None,
                clip_score2=None,
                score=result["score"],
            )
            result_items.append(result_item)

        return result_items

    def search_image(self, image, k=3, filters=None):
        clip_emb = self.clip_embedder.encode_images(image)[0]
        clip_v = f"[{','.join(f'{v:.7f}' for v in clip_emb)}]"

        filters_sql, params = self.parse_filters(filters)

        search_query = f"""
            WITH scores AS (
                SELECT
                    F.sku,
                    1 - (F.clip_image1 <=> '{clip_v}'::vector) as clip_score1,
                    1 - (F.clip_image2 <=> '{clip_v}'::vector) as clip_score2
                FROM item.features AS F
            )
            , weighted AS (
                SELECT
                    S.sku, S.clip_score1, S.clip_score2 ,
                    GREATEST(
                        S.clip_score1,
                        S.clip_score2
                    ) AS score
                FROM scores AS S
            )
            SELECT
                W.sku, W.clip_score1, W.clip_score2, W.score, A.title,
                A.brand, A.category, A.color, A.price,
                A.url, COALESCE(A.image2, A.image1) as image,
                COALESCE(A.text1, A.text2, A.text3) as text
            FROM weighted AS W
            INNER JOIN item.attributes AS A ON W.sku = A.sku
            LEFT JOIN item.colors AS C ON A.color = C.source_color
            WHERE 1=1
                {filters_sql}
            ORDER BY W.score DESC
            LIMIT {k}
        """

        with Manager() as db:
            db.cursor.execute(search_query, params)
            results = db.cursor.fetchall()

        result_items = []
        for result in results:
            result_item = ResultItem(
                sku=result["sku"],
                title=result["title"],
                category=result["category"],
                color=result["color"],
                brand=result["brand"],
                price=result["price"],
                image=result["image"],
                text=result["text"],
                url=result["url"],
                clip_score=None,
                st_score=None,
                clip_score1=result["clip_score1"],
                clip_score2=result["clip_score2"],
                score=result["score"],
            )
            result_items.append(result_item)

        return result_items
