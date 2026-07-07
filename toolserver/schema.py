"""Tool schemas for the four toolserver endpoints.

`OPENAI_TOOLS`   — OpenAI-style function-calling schemas (type: "function").
`ANTHROPIC_TOOLS` — Anthropic-style variant ({name, description, input_schema}).

Both describe the same JSON bodies accepted by toolserver/app.py; tool name ==
endpoint path (POST /<name>).
"""

from __future__ import annotations

_PARAMS: dict[str, dict] = {
    "vector_search": {
        "description": (
            "Semantic search over ~4.8M Wikipedia/Wikidata entity abstracts. "
            "Returns the k most similar entities to the query, each with its "
            "Wikidata QID, title, cosine similarity score, and a short snippet "
            "of its abstract. Use this to find entities by description when "
            "you don't know their QID."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language search query.",
                },
                "k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 10,
                    "description": "Number of results to return (max 50).",
                },
            },
            "required": ["query"],
        },
    },
    "get_entity": {
        "description": (
            "Look up one entity by Wikidata QID. Returns its title, full "
            "Wikipedia abstract, known aliases, and its in/out degree in the "
            "knowledge graph (how many relationships point at / away from it)."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "qid": {
                    "type": "string",
                    "pattern": "^Q\\d+$",
                    "description": "Wikidata QID, e.g. 'Q13371'.",
                }
            },
            "required": ["qid"],
        },
    },
    "get_neighbors": {
        "description": (
            "List the knowledge-graph edges attached to an entity, paginated "
            "(hub-safe: check `total` and page with offset). Each edge has "
            "src/dst QIDs, the relation P-id (`rel`), its English label "
            "(`rel_label`), and the destination title. Optionally filter to a "
            "single relation and/or edge direction."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "qid": {
                    "type": "string",
                    "pattern": "^Q\\d+$",
                    "description": "Wikidata QID of the entity.",
                },
                "relation": {
                    "type": "string",
                    "pattern": "^P\\d+$",
                    "description": "Optional relation filter, e.g. 'P69'.",
                },
                "direction": {
                    "type": "string",
                    "enum": ["out", "in", "both"],
                    "default": "both",
                    "description": "Edge direction relative to the entity.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 25,
                    "description": "Edges per page (max 100).",
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "default": 0,
                    "description": "Pagination offset.",
                },
            },
            "required": ["qid"],
        },
    },
    "find_paths": {
        "description": (
            "Find connecting paths between two entities in the knowledge "
            "graph (undirected, shortest paths first, at most max_hops "
            "relationships). Returns each path's node QIDs/titles and the "
            "labeled edges along it. Useful to discover how two entities are "
            "related."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "src_qid": {
                    "type": "string",
                    "pattern": "^Q\\d+$",
                    "description": "QID of the first entity.",
                },
                "dst_qid": {
                    "type": "string",
                    "pattern": "^Q\\d+$",
                    "description": "QID of the second entity.",
                },
                "max_hops": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 4,
                    "default": 3,
                    "description": "Maximum path length in hops (max 4).",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 10,
                    "description": "Maximum number of paths (max 20).",
                },
            },
            "required": ["src_qid", "dst_qid"],
        },
    },
}

OPENAI_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": spec["description"],
            "parameters": spec["schema"],
        },
    }
    for name, spec in _PARAMS.items()
]

ANTHROPIC_TOOLS: list[dict] = [
    {
        "name": name,
        "description": spec["description"],
        "input_schema": spec["schema"],
    }
    for name, spec in _PARAMS.items()
]

TOOL_NAMES = list(_PARAMS)
