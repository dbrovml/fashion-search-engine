from contextlib import asynccontextmanager
from io import BytesIO
from typing import Optional
from typing import AsyncGenerator

from PIL import Image
from fastapi import FastAPI, Request
from fastapi.params import File, Form, UploadFile
from fastapi.responses import HTMLResponse

from src.search.pipeline import Pipeline


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.state.pipeline = Pipeline()
    yield


app = FastAPI(name="FashionSearch", lifespan=lifespan)


@app.post("/search", response_class=HTMLResponse)
def search(
    request: Request,
    q_text: Optional[str] = Form(None),
    q_image: Optional[UploadFile] = File(None),
):
    pipeline: Pipeline = app.state.pipeline

    if not q_text and (not q_image or not q_image.filename):
        return {"error": "Query or image is required"}

    image = None
    if q_image and q_image.filename:
        content = q_image.read()
        try:
            image = Image.open(BytesIO(content)).convert("RGB")
        except Exception as e:
            return {"error": f"Invalid image: {e}"}

    output = pipeline.run(q_text=q_text, q_image=image)
    return output
