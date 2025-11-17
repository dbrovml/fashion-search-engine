from src.search.filters import Extractor
from src.search.query import Query


class Engine:

    def __init__(self):
        self.extractor = Extractor()
        self.query = Query()

    def run(self, q_text=None, q_image=None, k=9):
        filters = self.extractor(q_text) if q_text else None
        if q_image is not None:
            result_items = self.query.search_image(q_image, k=k, filters=filters)
        else:
            result_items = self.query.search_text(
                filters.clean_query, k=k, filters=filters
            )
        return {
            "Items": self._format_items(result_items),
            "Applied Filters": self._format_filters(filters),
            "Query": self._format_query(q_text, q_image),
        }

    def _format_items(self, items):
        formatted_items = []
        for item in items:
            formatted_items.append(
                {
                    "CLIP Packshot Image Score": (
                        f"{item.clip_score1:.2f}" if item.clip_score1 else None
                    ),
                    "CLIP Model Image Score": (
                        f"{item.clip_score2:.2f}" if item.clip_score2 else None
                    ),
                    "CLIP Text Score": (
                        f"{item.clip_score:.2f}" if item.clip_score else None
                    ),
                    "ST Text Score": (
                        f"{item.st_score:.2f}" if item.st_score else None
                    ),
                    "Final Score": f"{item.score:.2f}",
                    "Title": item.title,
                    "Category": item.category,
                    "Brand": item.brand,
                    "Color": item.color,
                    "Price": f"£{item.price:.2f}",
                    "Image URL": item.image,
                    "Product URL": item.url,
                    "SKU": item.sku,
                }
            )
        return formatted_items

    def _format_query(self, q_text, q_image):
        return {
            "Query Image": q_image,
            "Query Text": q_text,
        }

    def _format_filters(self, filters):
        if not filters:
            return None
        return {
            "Min Price": filters.min_price,
            "Max Price": filters.max_price,
            "Brand": filters.brand,
            "Category": filters.category,
            "Color": filters.color,
        }
