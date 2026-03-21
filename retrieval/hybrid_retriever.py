from typing import List, Dict, Optional
from dataclasses import dataclass, field
from retrieval.bm25_index import BM25Index
from ingest.page_indexer import HierarchicalPageIndexer


@dataclass
class HybridResult:
    """Result from hybrid (BM25 + structural) retrieval."""
    chunk_id: str
    text: str
    bm25_score: float
    dense_score: float = 0.0
    structural_score: float = 0.0
    combined_score: float = 0.0
    metadata: Dict = field(default_factory=dict)
    breadcrumb: List[str] = field(default_factory=list)
    related_chunks: List = field(default_factory=list)


class HybridRetriever:
    """Combines BM25 keyword search with structural scoring (and optional dense)."""

    STRUCTURAL_SCORES = {
        'function': 0.8,
        'class': 0.7,
        'method': 0.7,
        'import': 0.2,
        'docstring': 0.5,
    }

    def __init__(self, bm25_index: BM25Index, page_indexer: HierarchicalPageIndexer, use_dense: bool = False):
        self.bm25_index = bm25_index
        self.page_indexer = page_indexer
        self.use_dense = use_dense

    def search(
        self,
        query: str,
        top_k: int = 5,
        bm25_weight: float = 0.7,
        dense_weight: float = 0.2,
        structural_weight: float = 0.1,
    ) -> List[HybridResult]:
        """Run hybrid search and return ranked results."""
        results: Dict[str, HybridResult] = {}

        # BM25 search
        bm25_hits = self.bm25_index.search(query, top_k=top_k * 3)
        for chunk_id, bm25_score in bm25_hits:
            doc = self.bm25_index.documents.get(chunk_id, {})
            results[chunk_id] = HybridResult(
                chunk_id=chunk_id,
                text=doc.get('content', ''),
                bm25_score=bm25_score,
                metadata=doc.get('metadata', {}),
            )

        if not results:
            return []

        max_bm25 = max(r.bm25_score for r in results.values()) or 1.0

        # Compute combined scores
        for chunk_id, result in results.items():
            bm25_norm = result.bm25_score / max_bm25
            chunk_type = result.metadata.get('type', 'unknown')
            structural = self.STRUCTURAL_SCORES.get(chunk_type, 0.3)
            result.structural_score = structural
            result.combined_score = (
                bm25_weight * bm25_norm
                + structural_weight * structural
            )
            result.breadcrumb = self.page_indexer.get_breadcrumb(chunk_id)
            context = self.page_indexer.expand_context(chunk_id)
            if context:
                result.related_chunks = context.get('siblings', [])

        sorted_results = sorted(results.values(), key=lambda x: x.combined_score, reverse=True)
        return sorted_results[:top_k]

    @classmethod
    def from_indexes(cls, bm25_path: str, pageindex_path: str, use_dense: bool = False) -> 'HybridRetriever':
        """Load persisted indexes and return a ready retriever."""
        import json
        from ingest.page_indexer import HierarchicalPageIndexer

        bm25 = BM25Index()
        bm25.load(bm25_path)

        with open(pageindex_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        page_indexer = HierarchicalPageIndexer.deserialize(data)

        return cls(bm25_index=bm25, page_indexer=page_indexer, use_dense=use_dense)
