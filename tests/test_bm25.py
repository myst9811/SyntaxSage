import json
import os
import tempfile
import pytest
from retrieval.bm25_index import BM25Index


class TestBM25Index:
    def test_tokenize_basic(self):
        idx = BM25Index()
        tokens = idx.tokenize("hello world foo")
        assert "hello" in tokens
        assert "world" in tokens
        assert "foo" in tokens

    def test_tokenize_removes_stop_words(self):
        idx = BM25Index()
        tokens = idx.tokenize("from import def class return")
        assert tokens == []

    def test_tokenize_splits_camel_case(self):
        idx = BM25Index()
        tokens = idx.tokenize("myFunctionName")
        assert "function" in tokens
        assert "name" in tokens

    def test_tokenize_short_words_removed(self):
        idx = BM25Index()
        tokens = idx.tokenize("a ab abc abcd")
        assert "abc" in tokens
        assert "abcd" in tokens
        assert "ab" not in tokens

    def test_index_and_search(self):
        idx = BM25Index()
        idx.index_document("d1", "password validation function checks user input")
        idx.index_document("d2", "database connection pool management")
        idx.index_document("d3", "validate user password strength")
        idx.compute_idf()
        results = idx.search("password validation", top_k=3)
        ids = [r[0] for r in results]
        assert "d1" in ids
        assert "d3" in ids

    def test_search_returns_empty_for_no_match(self):
        idx = BM25Index()
        idx.index_document("d1", "hello world example")
        idx.compute_idf()
        results = idx.search("zzznonexistent")
        assert results == []

    def test_search_auto_computes_idf(self):
        idx = BM25Index()
        idx.index_document("d1", "hello world example code")
        results = idx.search("hello world")
        assert len(results) > 0

    def test_save_load_roundtrip(self):
        idx = BM25Index(k1=1.8, b=0.6)
        idx.index_document("d1", "password validation function", metadata={"file": "auth.py"})
        idx.index_document("d2", "database connection pool")
        idx.compute_idf()

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            path = f.name

        try:
            idx.save(path)
            loaded = BM25Index()
            loaded.load(path)
            assert loaded.total_docs == 2
            assert loaded.k1 == 1.8
            assert loaded.b == 0.6
            assert loaded.documents["d1"]["metadata"]["file"] == "auth.py"
            results = loaded.search("password validation")
            assert results[0][0] == "d1"
        finally:
            os.unlink(path)

    def test_empty_index_search(self):
        idx = BM25Index()
        results = idx.search("anything")
        assert results == []

    def test_metadata_preserved(self):
        idx = BM25Index()
        idx.index_document("d1", "some code", metadata={"type": "function", "name": "foo"})
        assert idx.documents["d1"]["metadata"]["name"] == "foo"
