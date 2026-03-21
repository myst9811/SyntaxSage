import json
import os
import tempfile
import pytest
from retrieval.bm25_index import BM25Index
from ingest.page_indexer import HierarchicalPageIndexer
from retrieval.hybrid_retriever import HybridRetriever, HybridResult


SAMPLE_CODE = '''
import os

class UserAuth:
    def login(self, username, password):
        return True

    def logout(self):
        pass

def validate_password(password):
    return len(password) >= 8

def connect_database(host, port):
    return None
'''.strip()


class TestHybridRetriever:
    def setup_method(self):
        self.indexer = HierarchicalPageIndexer()
        self.bm25 = BM25Index()
        page = self.indexer.index_file("auth.py", SAMPLE_CODE, "python")
        for chunk_id, chunk in page.chunks.items():
            self.bm25.index_document(
                chunk_id, chunk.text,
                metadata={
                    'page_id': page.page_id,
                    'file': 'auth.py',
                    'type': chunk.chunk_type,
                    'name': chunk.metadata.get('name'),
                },
            )
        self.bm25.compute_idf()
        self.retriever = HybridRetriever(self.bm25, self.indexer)

    def test_search_returns_results(self):
        results = self.retriever.search("password validation")
        assert len(results) > 0

    def test_search_returns_hybrid_results(self):
        results = self.retriever.search("password")
        assert all(isinstance(r, HybridResult) for r in results)

    def test_scores_are_normalized(self):
        results = self.retriever.search("password")
        for r in results:
            assert 0 <= r.combined_score <= 1.0

    def test_breadcrumb_populated(self):
        results = self.retriever.search("login")
        for r in results:
            assert isinstance(r.breadcrumb, list)

    def test_structural_scores_applied(self):
        results = self.retriever.search("login")
        for r in results:
            assert r.structural_score > 0

    def test_empty_query_returns_empty(self):
        results = self.retriever.search("")
        assert results == []

    def test_from_indexes_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            bm25_path = os.path.join(tmp, "bm25.json")
            page_path = os.path.join(tmp, "pages.json")
            self.bm25.save(bm25_path)
            with open(page_path, 'w') as f:
                json.dump(self.indexer.serialize(), f)
            loaded = HybridRetriever.from_indexes(bm25_path, page_path)
            results = loaded.search("password")
            assert len(results) > 0
