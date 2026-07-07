"""Shared configuration — single source of truth for service endpoints and model ids.

Values mirror DESIGN.md; override via environment variables for remote/training runs.
"""

import os

NEO4J_URI = os.environ.get("ECS_NEO4J_URI", "bolt://localhost:7687")
NEO4J_AUTH = (
    os.environ.get("ECS_NEO4J_USER", "neo4j"),
    os.environ.get("ECS_NEO4J_PASSWORD", "ecs-local-dev"),
)

QDRANT_URL = os.environ.get("ECS_QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = "wiki_entities"
EMBED_DIM = 1024

EMBED_MODEL = os.environ.get("ECS_EMBED_MODEL", "microsoft/harrier-oss-v1-0.6b")
EMBED_MODEL_SMALL = "microsoft/harrier-oss-v1-270m"
QUERY_INSTRUCTION = (
    "Given a question, retrieve Wikipedia entities relevant to answering it"
)

TOOLSERVER_URL = os.environ.get("ECS_TOOLSERVER_URL", "http://localhost:7801")

DATA_DIR = os.environ.get(
    "ECS_DATA_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "data"),
)

MAX_ANSWER_ENTITIES = 50
