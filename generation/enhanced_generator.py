from typing import Dict, List
from anthropic import Anthropic
from retrieval.hybrid_retriever import HybridRetriever
from config import settings


class EnhancedResponseGenerator:
    """Generate answers with retrieval transparency using BM25-backed context."""

    def __init__(self, bm25_path: str = None, pageindex_path: str = None):
        self.client = Anthropic(api_key=settings.anthropic_api_key)
        bm25_path = bm25_path or settings.bm25_index_path
        pageindex_path = pageindex_path or settings.pageindex_path
        self.retriever = HybridRetriever.from_indexes(bm25_path, pageindex_path, use_dense=False)

    def generate_with_transparency(self, query: str) -> Dict:
        """Retrieve relevant chunks, build context, call Claude, return full response dict."""
        results = self.retriever.search(query, top_k=settings.top_k)

        context_blocks = []
        for i, result in enumerate(results, 1):
            name = result.metadata.get('name') or 'unknown'
            file_ = result.metadata.get('file') or 'unknown'
            type_ = result.metadata.get('type') or 'unknown'
            context_blocks.append(
                f"## Source {i}: {name}\n"
                f"**File**: {file_}  |  **Type**: {type_}\n"
                f"**BM25 Score**: {result.bm25_score:.3f}  |  "
                f"**Combined Relevance**: {result.combined_score:.0%}\n\n"
                f"```\n{result.text[:500]}\n```"
            )

        context_text = '\n\n'.join(context_blocks)
        prompt = (
            f"User Query: {query}\n\n"
            f"Retrieved Code Context (ranked by relevance):\n{context_text}\n\n"
            "Please provide a comprehensive answer based on the code context above. "
            "Cite specific functions/classes and explain their relationships."
        )

        response = self.client.messages.create(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

        return {
            'query': query,
            'response': response.content[0].text,
            'retrieval_results': [
                {
                    'name': r.metadata.get('name'),
                    'file': r.metadata.get('file'),
                    'type': r.metadata.get('type'),
                    'bm25_score': f"{r.bm25_score:.3f}",
                    'combined_score': f"{r.combined_score:.0%}",
                    'breadcrumb': r.breadcrumb,
                }
                for r in results
            ],
            'retrieval_method': 'Vectorless BM25 (Fast & Interpretable)',
        }
