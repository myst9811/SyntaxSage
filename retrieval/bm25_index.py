from typing import List, Dict, Tuple
from collections import defaultdict
import math
import re
import pickle


class BM25Index:
    """Okapi BM25 implementation for code search."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = {}
        self.inverted_index = defaultdict(set)
        self.doc_lengths = {}
        self.term_frequencies = defaultdict(lambda: defaultdict(int))
        self.idf = {}
        self.avg_doc_length = 0
        self.total_docs = 0

    def tokenize(self, text: str) -> List[str]:
        """Code-aware tokenization: splits on whitespace + camelCase + underscores."""
        text = text.lower()
        # Split camelCase: e.g. "myFunction" -> "my function"
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        tokens = re.findall(r'\b\w+\b', text)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'import', 'def', 'class', 'if', 'else',
            'return', 'pass', 'self', 'this', 'var', 'let', 'const', 'none',
            'true', 'false', 'not', 'is', 'as', 'try', 'except', 'finally',
        }
        return [t for t in tokens if t not in stop_words and len(t) > 2]

    def index_document(self, doc_id: str, content: str, metadata: Dict = None):
        """Add a document to the BM25 index."""
        self.documents[doc_id] = {'content': content, 'metadata': metadata or {}}
        tokens = self.tokenize(content)
        self.doc_lengths[doc_id] = len(tokens)
        for token in tokens:
            self.term_frequencies[doc_id][token] += 1
            self.inverted_index[token].add(doc_id)
        self.total_docs += 1
        self.avg_doc_length = sum(self.doc_lengths.values()) / self.total_docs

    def compute_idf(self):
        """Compute IDF for all indexed terms. Call after all documents are indexed."""
        for term, doc_ids in self.inverted_index.items():
            df = len(doc_ids)
            self.idf[term] = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Return top-k (doc_id, score) pairs ranked by BM25."""
        if not self.idf:
            self.compute_idf()
        query_tokens = self.tokenize(query)
        scores: Dict[str, float] = defaultdict(float)
        for token in query_tokens:
            if token not in self.idf:
                continue
            idf = self.idf[token]
            for doc_id in self.inverted_index[token]:
                tf = self.term_frequencies[doc_id][token]
                doc_len = self.doc_lengths[doc_id]
                avg = self.avg_doc_length or 1
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / avg))
                scores[doc_id] += idf * (numerator / denominator)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def save(self, path: str):
        """Serialize index to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'inverted_index': dict(self.inverted_index),
                'doc_lengths': self.doc_lengths,
                'term_frequencies': {k: dict(v) for k, v in self.term_frequencies.items()},
                'idf': self.idf,
                'avg_doc_length': self.avg_doc_length,
                'total_docs': self.total_docs,
                'k1': self.k1,
                'b': self.b,
            }, f)

    def load(self, path: str):
        """Load index from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.documents = data['documents']
        self.inverted_index = defaultdict(set, {k: set(v) for k, v in data['inverted_index'].items()})
        self.doc_lengths = data['doc_lengths']
        self.term_frequencies = defaultdict(lambda: defaultdict(int))
        for doc_id, terms in data['term_frequencies'].items():
            for term, count in terms.items():
                self.term_frequencies[doc_id][term] = count
        self.idf = data['idf']
        self.avg_doc_length = data['avg_doc_length']
        self.total_docs = data['total_docs']
        self.k1 = data.get('k1', self.k1)
        self.b = data.get('b', self.b)
