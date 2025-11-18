"""FastAPI entrypoint for the fashion search application."""

from base64 import b64encode
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Any, AsyncGenerator, Optional

from PIL import Image
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.config import STATIC_DIR, TEMPLATE_DIR
from src.search.engine import Engine


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Create and store the shared search engine instance."""
    app.state.engine = Engine()
    yield


app = FastAPI(name="FashionSearch", lifespan=lifespan)
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    """Return the health of the application."""
    return {"status": "ok"}


def _empty_context() -> dict[str, Any]:
    """Return the default template context."""
    return {
        "items": [],
        "query": {"Query Text": None, "Query Image": None},
        "filters": None,
        "error": None,
    }


def _encode_query_image(image: Image.Image) -> str:
    """Convert a PIL image into a data URL."""
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    encoded = b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Render the search landing page."""
    context = {"request": request, **_empty_context()}
    return templates.TemplateResponse("index.html", context)


@app.post("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    q_text: Optional[str] = Form(None),
    q_image: Optional[UploadFile] = File(None),
) -> HTMLResponse:
    """Process a search submission using text and/or image input."""
    engine: Engine = app.state.engine
    image_data_url: Optional[str] = None

    if not q_text and (not q_image or not q_image.filename):
        context = {
            "request": request,
            **_empty_context(),
            "error": "Enter a description or upload an image to search.",
        }
        return templates.TemplateResponse(
            "partials/results.html",
            context,
        )

    image: Optional[Image.Image] = None
    if q_image and q_image.filename:
        content = await q_image.read()
        try:
            image = Image.open(BytesIO(content)).convert("RGB")
        except Exception as e:
            context = {
                "request": request,
                **_empty_context(),
                "error": f"Invalid image: {e}",
            }
            return templates.TemplateResponse(
                "partials/results.html",
                context,
            )
        image_data_url = _encode_query_image(image)

    output = engine.run(q_text=q_text, q_image=image)
    query_payload = dict(output.get("Query") or {})
    if "Query Text" not in query_payload:
        query_payload["Query Text"] = q_text
    if "Query Image" not in query_payload:
        query_payload["Query Image"] = None
    if image_data_url:
        query_payload["Query Image"] = image_data_url

    context = {
        "request": request,
        "items": output.get("Items", []),
        "query": query_payload,
        "filters": output.get("Applied Filters"),
        "error": None,
    }
    return templates.TemplateResponse(
        "partials/results.html",
        context,
    )
