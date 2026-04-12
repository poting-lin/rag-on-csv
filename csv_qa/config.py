"""Application-wide configuration constants."""

# Default embedding model for semantic search.
DEFAULT_EMBED_MODEL = "nomic-embed-text"

# Embedding models shown in the "Download Embedding Model" dropdown.
POPULAR_EMBEDDING_MODELS: list[str] = [
    "nomic-embed-text",
    "mxbai-embed-large",
    "all-minilm",
    "snowflake-arctic-embed",
]

# LLM models shown in the "Download New Model" dropdown.
POPULAR_MODELS: list[str] = [
    "llama3.2:1b",
    "llama3.2:3b",
    "llama3.1:8b",
    "gemma3:4b",
    "gemma4:e2b",
    "gemma4:e4b",
    "gemma4:26b",
    "mistral",
    "phi4-mini",
    "qwen3:8b",
    "qwen3.5:2b",
    "qwen3.5:4b",
    "qwen3.5:9b",
]
