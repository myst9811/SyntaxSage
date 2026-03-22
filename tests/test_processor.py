import json
import pytest
from pathlib import Path

from ingest.vectorless_processor import VectorlessDocumentProcessor
from retrieval.hybrid_retriever import HybridRetriever


SAMPLE_CLASS = '''\
class AuthManager:
    """Handles user authentication and session management."""

    def authenticate_user(self, username, password):
        """Validate credentials and return a session token."""
        if not username or not password:
            raise ValueError("Username and password are required")
        return f"token_{username}"

    def revoke_session(self, token):
        """Invalidate an existing session token."""
        pass
'''

SAMPLE_FUNCTION = '''\
def calculate_checksum(data):
    """Compute a simple checksum for data integrity validation."""
    return sum(ord(c) for c in str(data)) % 256
'''


@pytest.fixture()
def sample_repo(tmp_path):
    (tmp_path / "auth.py").write_text(SAMPLE_CLASS)
    (tmp_path / "utils.py").write_text(SAMPLE_FUNCTION)
    return tmp_path


def test_process_repository_indexes_chunks(sample_repo):
    processor = VectorlessDocumentProcessor()
    processor.process_repository(str(sample_repo), extensions=['.py'])
    assert processor.bm25_index.total_docs > 0


def test_process_repository_finds_class_and_function(sample_repo):
    processor = VectorlessDocumentProcessor()
    processor.process_repository(str(sample_repo), extensions=['.py'])
    types = {
        doc['metadata']['type']
        for doc in processor.bm25_index.documents.values()
    }
    assert 'class' in types
    assert 'function' in types or 'method' in types


def test_save_and_reload_indexes(sample_repo, tmp_path):
    processor = VectorlessDocumentProcessor()
    processor.process_repository(str(sample_repo), extensions=['.py'])

    bm25_path = str(tmp_path / 'bm25_index.json')
    pageindex_path = str(tmp_path / 'pageindex.json')
    processor.save_indexes(bm25_path, pageindex_path)

    assert Path(bm25_path).exists()
    assert Path(pageindex_path).exists()

    with open(bm25_path) as f:
        data = json.load(f)
    assert data['total_docs'] > 0


def test_search_returns_relevant_results(sample_repo, tmp_path):
    processor = VectorlessDocumentProcessor()
    processor.process_repository(str(sample_repo), extensions=['.py'])

    bm25_path = str(tmp_path / 'bm25_index.json')
    pageindex_path = str(tmp_path / 'pageindex.json')
    processor.save_indexes(bm25_path, pageindex_path)

    retriever = HybridRetriever.from_indexes(bm25_path, pageindex_path)
    results = retriever.search("authenticate user", top_k=5)

    assert len(results) > 0
    names = [r.metadata.get('name') for r in results]
    assert any(n in ('AuthManager', 'authenticate_user') for n in names)


def test_search_checksum(sample_repo, tmp_path):
    processor = VectorlessDocumentProcessor()
    processor.process_repository(str(sample_repo), extensions=['.py'])

    bm25_path = str(tmp_path / 'bm25_index.json')
    pageindex_path = str(tmp_path / 'pageindex.json')
    processor.save_indexes(bm25_path, pageindex_path)

    retriever = HybridRetriever.from_indexes(bm25_path, pageindex_path)
    results = retriever.search("checksum data integrity", top_k=5)

    assert len(results) > 0
    names = [r.metadata.get('name') for r in results]
    assert 'calculate_checksum' in names


def test_empty_extensions_returns_no_chunks(sample_repo):
    processor = VectorlessDocumentProcessor()
    processor.process_repository(str(sample_repo), extensions=['.rb'])
    assert processor.bm25_index.total_docs == 0
