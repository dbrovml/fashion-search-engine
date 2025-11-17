from contextlib import asynccontextmanager
from io import BytesIO
from typing import Optional
from typing import AsyncGenerator

from PIL import Image
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.config import STATIC_DIR, TEMPLATE_DIR
from src.search.engine import Engine


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.state.engine = Engine()
    yield


app = FastAPI(name="FashionSearch", lifespan=lifespan)
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _empty_context():
    return {
        "items": [],
        "query": {"Query Text": None, "Query Image": None},
        "filters": None,
        "error": None,
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    context = {"request": request, **_empty_context()}
    return templates.TemplateResponse("index.html", context)


@app.post("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    q_text: Optional[str] = Form(None),
    q_image: Optional[UploadFile] = File(None),
):
    engine: Engine = app.state.engine

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

    image = None
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

    output = engine.run(q_text=q_text, q_image=image)
    context = {
        "request": request,
        "items": output.get("Items", []),
        "query": output.get("Query"),
        "filters": output.get("Applied Filters"),
        "error": None,
    }
    return templates.TemplateResponse(
        "partials/results.html",
        context,
    )
