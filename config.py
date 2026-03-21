import os
from dataclasses import dataclass


@dataclass
class Settings:
    retrieval_mode: str = "vectorless"  # "vectorless" | "hybrid"
    use_dense_embeddings: bool = False
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    top_k: int = 5
    bm25_index_path: str = "bm25_index.json"
    pageindex_path: str = "pageindex.json"
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    llm_model: str = "claude-sonnet-4-6"
    llm_max_tokens: int = 2048


settings = Settings()
