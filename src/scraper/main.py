from itertools import batched
import json
import subprocess
import time

import requests
from tqdm import tqdm
import typer

from src.config import ATTRIBUTE_DIR, AWS_S3_BUCKET, IMAGE_DIR
from src.database.schemas import upsert_to_attributes

app = typer.Typer()


class Config:
    url = "https://www.zalando.co.uk/api/graphql/"
    list_query_id = "d9c95b97c8629d02b4fbcf66c3314afd9e2192c802103ebaaad83845afc41b26"
    item_query_id = "8020c8d0d6d82c0cee1ec4c464423f85b5dfb10c6b19ddcd5149abb488b09df1"
    headers = {
        "Referer": "https://www.zalando.co.uk/womens-clothing-dresses/",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/138.0.0.0 Safari/537.36 Edg/138.0.0.0",
        "Accept": "*/*",
        "Origin": "https://www.zalando.cz",
    }
    categories = [
        "womens-clothing-pullovers-and-cardigans",
        "womens-clothing-blouses-tunics",
        "womens-clothing-dresses",
        "womens-clothing-skirts",
        "womens-clothing-trousers",
        "womens-clothing-tops",
        "womens-clothing-jumpers-cardigans",
        "womens-clothing-jeans",
        "womens-clothing-coats",
        "womens-clothing-jackets",
        "womens-clothing-playsuits-jumpsuits",
        "womens-clothing-shorts",
        "womens-shoes-trainers",
        "womens-shoes-ankle-boots",
        "womens-shoes-boots",
        "womens-shoes-heels",
        "womens-shoes-flats-lace-ups",
        "womens-shoes-ballet-pumps",
        "womens-shoes-high-heels",
        "womens-shoes-sandals",
        "womens-sports-shoes",
    ]


def get_session():
    session = requests.session()
    session.headers.update(Config.headers)
    return session


class ListScraper:

    def __init__(self, config=Config):
        self.config = config
        self.session = get_session()

    def scrape_catalog(self, category, max_pages=None):
        category_id = f"ern:collection:cat:categ:{category}"
        item_list = []
        start_cursor = None

        while True:
            payload = [
                {
                    "id": self.config.list_query_id,
                    "variables": {
                        "id": category_id,
                        "orderBy": "POPULARITY",
                        "filters": {
                            "discreteFilters": [],
                            "rangeFilters": [],
                            "toggleFilters": [],
                        },
                        "after": start_cursor,
                        "first": 84,
                        "isPaginationRequest": (start_cursor is None),
                        "fetchExperience": True,
                        "subSli": "client",
                        "width": 2079,
                        "height": 1000,
                        "isLoggedIn": False,
                        "forcedEntities": [None],
                    },
                }
            ]

            response = self.session.post(self.config.url, data=json.dumps(payload))
            response.raise_for_status()
            data = response.json()

            page_info = data[0]["data"]["collection"]["entities"]["pageInfo"]

            if start_cursor is None:
                actual_max_pages = page_info.get("numberOfPages")
                if actual_max_pages:
                    if max_pages is None:
                        max_pages = actual_max_pages
                    else:
                        max_pages = min(max_pages, actual_max_pages)

            for edge in data[0]["data"]["collection"]["entities"]["edges"]:
                item_list.append(edge["node"]["id"])

            if page_info["currentPage"] >= max_pages:
                break
            start_cursor = page_info.get("endCursor")
            if not start_cursor:
                break

        return item_list


class ItemScraper:

    def __init__(self, config=Config):
        self.config = config
        self.session = get_session()
        self.attribute_dir = ATTRIBUTE_DIR
        self.image_dir = IMAGE_DIR

    def _make_payload(self, batch):
        payload = []
        for item in batch:
            payload.append(
                {
                    "id": self.config.item_query_id,
                    "variables": {
                        "displayContext": {"module": "PRODUCT_CARD_WITH_HOVER"},
                        "id": item,
                        "isRatingEnabled": True,
                        "isSustainabilityProductScoreEnabled": False,
                        "moduleInput": {"module": "PRODUCT_CARD_WITH_HOVER"},
                        "shouldLoadGallery": False,
                        "skipHoverData": False,
                        "version": 1,
                    },
                }
            )
        return payload

    def _scrape_batch(self, batch):
        payload = self._make_payload(batch)
        response = self.session.post(
            self.config.url, data=json.dumps(payload), timeout=10
        )
        response.raise_for_status()
        data = response.json()

        parsed_items = []
        for item in data:
            parsed_item = self.parse_item(item)
            if parsed_item:
                parsed_items.append(parsed_item)
        return parsed_items

    def scrape_items(self, items_list, batch_size=16):
        for batch in tqdm(
            batched(items_list, batch_size),
            total=len(items_list) // batch_size,
        ):
            parsed_items = self._scrape_batch(tuple(batch))
            for parsed_item in parsed_items:
                sku = parsed_item.get("sku")
                attribute_path = self.attribute_dir / f"{sku}.json"
                attribute_path.parent.mkdir(parents=True, exist_ok=True)
                attribute_path.write_text(
                    json.dumps(parsed_item, indent=2, sort_keys=True)
                )
                self.scrape_item_images(parsed_item)

    def scrape_item_images(self, parsed_item):
        sku = parsed_item.get("sku")
        sku_dir = self.image_dir / sku
        sku_dir.mkdir(parents=True, exist_ok=True)

        for idx, url in enumerate(
            [parsed_item.get("image1"), parsed_item.get("image2")], start=1
        ):
            if not url:
                continue
            target_path = sku_dir / f"image{idx}.jpeg"
            if target_path.exists():
                continue
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                target_path.write_bytes(response.content)
                time.sleep(0.1)
            except Exception:
                continue

    def _build_texts(self, record):
        meta_free = (
            f"A {record.get('color')} {record.get('category') or ''} "
            f"from {record.get('brand') or ''}. "
        )
        meta_keys = " | ".join(
            [
                f"category: {record.get('category')}",
                f"color: {record.get('color')}",
                f"brand: {record.get('brand')}",
            ]
        )
        free_text = None
        for field in ["text1", "text2", "text3"]:
            value = record.get(field)
            if value:
                free_text = value
                break

        texts = f"{meta_free} {free_text} {meta_keys}".strip()
        return texts

    def parse_item(self, raw_item):
        try:
            product = raw_item["data"]["product"]

            price = product.get("displayPrice", {}).get("trackingCurrentAmount")

            title = product.get("name", "")
            color = title.split("-")[-1].strip().lower() if title else None

            packshot_image = product.get("mediumPackshotImage", {})
            person_image = product.get("mediumDefaultMedia", {})
            image1 = (
                packshot_image.get("uri")
                if packshot_image and isinstance(packshot_image, dict)
                else None
            )
            image2 = (
                person_image.get("uri")
                if person_image and isinstance(person_image, dict)
                else None
            )

            default_media = product.get("defaultMediaInfo", {})
            packshot_media = product.get("packshotImageInfo", {})
            hover_media = product.get("hoverMediaInfo", {})

            text1 = self._clean_text(
                packshot_media.get("alternativeText", "").lower() or None
                if packshot_media
                else None
            )
            text2 = self._clean_text(
                default_media.get("alternativeText", "").lower() or None
                if default_media
                else None
            )
            text3 = self._clean_text(
                hover_media.get("alternativeText", "").lower() or None
                if hover_media
                else None
            )

            brand = product.get("brand", {}).get("name", "")
            brand = brand.lower() if brand else None

            category = product.get("silhouette", "")
            if category:
                category = (
                    category.replace("_", " ")
                    .replace("-", " ")
                    .replace("/", " ")
                    .lower()
                )
            else:
                category = None

            record = {
                "sku": product.get("sku", "") or None,
                "title": title.lower() if title else None,
                "brand": brand,
                "category": category,
                "price": price,
                "color": color,
                "url": product.get("uri"),
                "image1": image1,
                "image2": image2,
                "text1": text1,
                "text2": text2,
                "text3": text3,
            }
            record["texts"] = self._build_texts(record)
            return record
        except Exception:
            return None

    def _clean_text(self, text):
        if text:
            return (
                text.replace("/", " ")
                .replace(";", " ")
                .replace(":", " ")
                .replace("(", " ")
                .replace(")", " ")
            )
        return None


@app.command("pull")
def pull(max_pages: int = typer.Option(None)):
    list_scraper = ListScraper()
    item_scraper = ItemScraper()

    for category in Config.categories:
        typer.echo(f"Scraping category: {category}")
        item_list = list_scraper.scrape_catalog(category, max_pages)
        item_scraper.scrape_items(item_list)


@app.command("push")
def push():
    records = []
    for path in ATTRIBUTE_DIR.glob("*.json"):
        records.append(json.loads(path.read_text()))

    upsert_to_attributes(records)

    s3_path = f"s3://{AWS_S3_BUCKET}/images/"
    subprocess.run(
        [
            "aws",
            "s3",
            "sync",
            str(IMAGE_DIR),
            s3_path,
        ]
    )


if __name__ == "__main__":
    app()
