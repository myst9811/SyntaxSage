from pathlib import Path
from ingest.page_indexer import HierarchicalPageIndexer
from retrieval.bm25_index import BM25Index
import json


class VectorlessDocumentProcessor:
    """Orchestrates BM25 + PageIndex ingestion for a code repository."""

    SKIP_DIRS = {'venv', '.venv', '__pycache__', '.git', 'node_modules', '.tox',
                 'dist', 'build', '.next', '.cache', 'coverage', 'env'}

    def __init__(self):
        self.page_indexer = HierarchicalPageIndexer()
        self.bm25_index = BM25Index()

    def process_repository(self, repo_path: str, extensions: list = None):
        """Walk repo, index all matching files into BM25 + PageIndex."""
        if extensions is None:
            extensions = ['.py']

        repo = Path(repo_path).resolve()
        if not repo.is_dir():
            raise FileNotFoundError(f"Repository path does not exist: {repo}")

        processed = 0
        skipped = 0

        for file_path in repo.rglob('*'):
            if not file_path.is_file():
                continue
            if file_path.suffix not in extensions:
                continue
            if any(part in self.SKIP_DIRS for part in file_path.parts):
                continue

            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                if not content.strip():
                    continue

                language = self._detect_language(file_path.suffix)
                page = self.page_indexer.index_file(str(file_path), content, language)

                for chunk_id, chunk in page.chunks.items():
                    self.bm25_index.index_document(
                        chunk_id,
                        chunk.text,
                        metadata={
                            'page_id': page.page_id,
                            'file': file_path.name,
                            'file_path': str(file_path),
                            'type': chunk.chunk_type,
                            'name': chunk.metadata.get('name'),
                        },
                    )
                processed += 1
            except Exception as e:
                skipped += 1
                print(f"Warning: skipping {file_path}: {e}")

        self.bm25_index.compute_idf()
        print(f"Indexed {self.bm25_index.total_docs} chunks from {processed} files.")
        if skipped:
            print(f"Skipped {skipped} files due to errors.")

    def save_indexes(self, bm25_path: str, pageindex_path: str):
        """Persist BM25 index (JSON) and PageIndex (JSON) to disk."""
        self.bm25_index.save(bm25_path)
        with open(pageindex_path, 'w', encoding='utf-8') as f:
            json.dump(self.page_indexer.serialize(), f, indent=2)
        print(f"Saved BM25 index -> {bm25_path}")
        print(f"Saved PageIndex  -> {pageindex_path}")

    @staticmethod
    def _detect_language(suffix: str) -> str:
        mapping = {
            '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
            '.tsx': 'typescript', '.jsx': 'javascript',
            '.java': 'java', '.cpp': 'cpp', '.c': 'c', '.go': 'go',
        }
        return mapping.get(suffix, 'plaintext')
