"""Shared configuration — single source of truth for service endpoints and model ids.

Values mirror DESIGN.md; override via environment variables for remote/training runs.
"""

import os

try:  # repo-root .env holds OPENROUTER_API_KEY; absent in some deploy contexts
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
except ImportError:
    pass

NEO4J_URI = os.environ.get("ECS_NEO4J_URI", "bolt://localhost:7687")
NEO4J_AUTH = (
    os.environ.get("ECS_NEO4J_USER", "neo4j"),
    os.environ.get("ECS_NEO4J_PASSWORD", "ecs-local-dev"),
)

QDRANT_URL = os.environ.get("ECS_QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = "wiki_entities"
EMBED_DIM = 640

EMBED_MODEL = os.environ.get("ECS_EMBED_MODEL", "microsoft/harrier-oss-v1-270m")
EMBED_MODEL_LARGE = "microsoft/harrier-oss-v1-0.6b"  # 1024-dim; requires re-embedding the collection
QUERY_INSTRUCTION = (
    "Given a question, retrieve Wikipedia entities relevant to answering it"
)

TOOLSERVER_URL = os.environ.get("ECS_TOOLSERVER_URL", "http://localhost:7801")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
QGEN_MODEL = os.environ.get("ECS_QGEN_MODEL", "deepseek/deepseek-v4-pro")

DATA_DIR = os.environ.get(
    "ECS_DATA_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "data"),
)

MAX_ANSWER_ENTITIES = 50
