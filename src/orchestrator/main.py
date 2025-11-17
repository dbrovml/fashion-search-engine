from src.agents.extractor import Extractor
from src.search.main import Engine


class Orchestrator:

    def __init__(self):
        self.extractor = Extractor()
        self.engine = Engine()

    def process_query(self, q_text=None, q_image=None, k=3):
        filters = self.extractor(q_text) if q_text else None
        if q_image is not None:
            result_items = self.engine.search_image(q_image, k=k, filters=filters)
        else:
            result_items = self.engine.search_text(
                filters.clean_query, k=k, filters=filters
            )
        return {
            "result_items": result_items,
            "filters": filters,
            "q_text": q_text,
            "q_image": q_image,
        }
