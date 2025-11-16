from src.agents.extractor import Extractor
from src.search.main import Engine


class Orchestrator:

    def __init__(self):
        self.extractor = Extractor()
        self.engine = Engine()

    def process_query(self, q_text=None, q_image=None, k=8):
        filters = self.extractor(q_text)
        if q_image is None:
            results = self.engine.search_text(filters.clean_query, k=k, filters=filters)
        else:
            results = self.engine.search_image(q_image, k=k, filters=filters)
        return {
            "results": results,
            "filters": filters,
            "q_text": q_text,
        }
