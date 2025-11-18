"""Filesystem config and environment variables."""

from pathlib import Path
import os
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

# Render injects secrets into the /etc/secrets/.env file
secret_env_path = "/etc/secrets/.env"
if os.path.exists(secret_env_path):
    load_dotenv(secret_env_path)

PROJECT_ROOT = Path(__file__).parent.parent

SRC_DIR = PROJECT_ROOT / "src"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DOCKER_DIR = PROJECT_ROOT / "docker"
DATA_DIR = PROJECT_ROOT / "data"

ATTRIBUTE_DIR = DATA_DIR / "attributes"
IMAGE_DIR = DATA_DIR / "images"

TEMPLATE_DIR = FRONTEND_DIR / "templates"
STATIC_DIR = FRONTEND_DIR / "static"

for dir_ in [
    SRC_DIR,
    FRONTEND_DIR,
    SCRIPTS_DIR,
    DOCKER_DIR,
    DATA_DIR,
    ATTRIBUTE_DIR,
    IMAGE_DIR,
    TEMPLATE_DIR,
    STATIC_DIR,
]:
    dir_.mkdir(parents=True, exist_ok=True)

ENVIRONMENT = os.getenv("ENVIRONMENT", "local")

POSTGRES_DB_URL = os.getenv("DATABASE_URL")  # <- render should inject this

if not POSTGRES_DB_URL:
    POSTGRES_HOST = os.getenv("POSTGRES_HOST")
    POSTGRES_DB = os.getenv("POSTGRES_DB")
    POSTGRES_USER = os.getenv("POSTGRES_USER")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT")
    if ENVIRONMENT == "lambda":
        POSTGRES_HOST = "localhost"
        POSTGRES_PORT = "15432"
    POSTGRES_DB_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

LAMBDA_SSH_KEY = os.getenv("LAMBDA_SSH_KEY")
LAMBDA_DIR = os.getenv("LAMBDA_DIR")
LAMBDA_USER = os.getenv("LAMBDA_USER")
LAMBDA_HOST = os.getenv("LAMBDA_HOST")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")

CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME")
CLIP_PRETRAINED = os.getenv("CLIP_PRETRAINED")
ST_MODEL_NAME = os.getenv("ST_MODEL_NAME")
