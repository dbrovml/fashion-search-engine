"""Search engine wrapper combining filter extraction and query execution."""

from typing import Any, Iterable

from src.search.filters import Extractor, Filters
from src.search.query import Query, ResultItem


class Engine:
    """High-level orchestrator for search filters and queries."""

    def __init__(self) -> None:
        """Initialize filter extractor and query backends."""
        self.extractor = Extractor()
        self.query = Query()

    def run(
        self,
        q_text: str | None = None,
        q_image: Any | None = None,
        k: int = 9,
    ) -> dict[str, Any]:
        """Execute a multimodal search and return formatted payload."""
        filters = self.extractor(q_text) if q_text else None
        if q_image is not None:
            result_items = self.query.search_image(q_image, k=k, filters=filters)
        else:
            result_items = self.query.search_text(
                filters.clean_query if filters else q_text,
                k=k,
                filters=filters,
            )
        return {
            "Items": self._format_items(result_items),
            "Applied Filters": self._format_filters(filters),
            "Query": self._format_query(q_text, q_image),
        }

    def _format_items(self, items: Iterable[ResultItem]) -> list[dict[str, Any]]:
        """Transform raw search results into template-ready dicts."""
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

    def _format_query(self, q_text: str | None, q_image: Any | None) -> dict[str, Any]:
        """Return a normalized query payload for templates."""
        return {
            "Query Image": q_image,
            "Query Text": q_text,
        }

    def _format_filters(self, filters: Filters | None) -> dict[str, Any] | None:
        """Format extracted filters for display."""
        if not filters:
            return None
        return {
            "Min Price": filters.min_price,
            "Max Price": filters.max_price,
            "Brand": filters.brand,
            "Category": filters.category,
            "Color": filters.color,
        }
