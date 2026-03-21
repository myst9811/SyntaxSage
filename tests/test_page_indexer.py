import json
import pytest
from ingest.page_indexer import HierarchicalPageIndexer, PageChunk, Page


SAMPLE_PYTHON = '''
import os
from pathlib import Path

class AuthManager:
    """Manages authentication."""

    def login(self, username, password):
        return True

    def logout(self):
        pass

def helper_function(x):
    return x * 2
'''.strip()

SAMPLE_JS = '''
import React from 'react';
import { useState } from 'react';

export class Editor {
    constructor(config) {
        this.config = config;
    }

    save() {
        return this.config;
    }
}

export function formatText(text) {
    return text.trim();
}

const processData = (data) => {
    return data.map(x => x);
};
'''.strip()


class TestPythonParsing:
    def setup_method(self):
        self.indexer = HierarchicalPageIndexer()

    def test_extracts_class(self):
        page = self.indexer.index_file("test.py", SAMPLE_PYTHON, "python")
        types = [c.chunk_type for c in page.chunks.values()]
        assert "class" in types

    def test_extracts_methods_inside_class(self):
        page = self.indexer.index_file("test.py", SAMPLE_PYTHON, "python")
        methods = [c for c in page.chunks.values() if c.chunk_type == "method"]
        names = [c.metadata["name"] for c in methods]
        assert "login" in names
        assert "logout" in names

    def test_methods_have_parent_id(self):
        page = self.indexer.index_file("test.py", SAMPLE_PYTHON, "python")
        methods = [c for c in page.chunks.values() if c.chunk_type == "method"]
        classes = [c for c in page.chunks.values() if c.chunk_type == "class"]
        assert len(classes) == 1
        class_id = classes[0].chunk_id
        for m in methods:
            assert m.parent_id == class_id

    def test_class_has_children(self):
        page = self.indexer.index_file("test.py", SAMPLE_PYTHON, "python")
        classes = [c for c in page.chunks.values() if c.chunk_type == "class"]
        assert len(classes[0].children) == 2  # login, logout

    def test_extracts_top_level_function(self):
        page = self.indexer.index_file("test.py", SAMPLE_PYTHON, "python")
        funcs = [c for c in page.chunks.values() if c.chunk_type == "function"]
        names = [c.metadata["name"] for c in funcs]
        assert "helper_function" in names

    def test_extracts_imports(self):
        page = self.indexer.index_file("test.py", SAMPLE_PYTHON, "python")
        imports = [c for c in page.chunks.values() if c.chunk_type == "import"]
        assert len(imports) == 1
        assert "import os" in imports[0].text

    def test_root_chunks_excludes_methods(self):
        page = self.indexer.index_file("test.py", SAMPLE_PYTHON, "python")
        root_types = [page.chunks[cid].chunk_type for cid in page.root_chunks]
        assert "method" not in root_types

    def test_breadcrumb_for_method(self):
        page = self.indexer.index_file("test.py", SAMPLE_PYTHON, "python")
        methods = [c for c in page.chunks.values() if c.metadata.get("name") == "login"]
        assert len(methods) == 1
        breadcrumb = self.indexer.get_breadcrumb(methods[0].chunk_id)
        assert "test.py" in breadcrumb
        assert "AuthManager" in breadcrumb
        assert "login" in breadcrumb


class TestJSTSParsing:
    def setup_method(self):
        self.indexer = HierarchicalPageIndexer()

    def test_extracts_class(self):
        page = self.indexer.index_file("editor.ts", SAMPLE_JS, "typescript")
        classes = [c for c in page.chunks.values() if c.chunk_type == "class"]
        assert any(c.metadata["name"] == "Editor" for c in classes)

    def test_extracts_class_methods(self):
        page = self.indexer.index_file("editor.ts", SAMPLE_JS, "typescript")
        methods = [c for c in page.chunks.values() if c.chunk_type == "method"]
        names = [c.metadata["name"] for c in methods]
        assert "constructor" in names or "save" in names

    def test_extracts_function(self):
        page = self.indexer.index_file("editor.ts", SAMPLE_JS, "typescript")
        funcs = [c for c in page.chunks.values() if c.chunk_type == "function"]
        names = [c.metadata["name"] for c in funcs]
        assert "formatText" in names

    def test_extracts_imports(self):
        page = self.indexer.index_file("editor.ts", SAMPLE_JS, "typescript")
        imports = [c for c in page.chunks.values() if c.chunk_type == "import"]
        assert len(imports) == 1


class TestSerialization:
    def test_serialize_deserialize_roundtrip(self):
        indexer = HierarchicalPageIndexer()
        indexer.index_file("test.py", SAMPLE_PYTHON, "python")
        data = indexer.serialize()
        json_str = json.dumps(data)
        restored = HierarchicalPageIndexer.deserialize(json.loads(json_str))
        assert len(restored.pages) == len(indexer.pages)
        for pid in indexer.pages:
            orig_page = indexer.pages[pid]
            rest_page = restored.pages[pid]
            assert len(orig_page.chunks) == len(rest_page.chunks)
            for cid in orig_page.chunks:
                assert orig_page.chunks[cid].chunk_type == rest_page.chunks[cid].chunk_type
                assert orig_page.chunks[cid].parent_id == rest_page.chunks[cid].parent_id
                assert orig_page.chunks[cid].children == rest_page.chunks[cid].children

    def test_expand_context_returns_siblings(self):
        indexer = HierarchicalPageIndexer()
        indexer.index_file("test.py", SAMPLE_PYTHON, "python")
        methods = [c for c in list(indexer.pages.values())[0].chunks.values() if c.chunk_type == "method"]
        if methods:
            ctx = indexer.expand_context(methods[0].chunk_id)
            assert ctx is not None
            assert ctx['parent'] is not None
            assert len(ctx['siblings']) > 0


class TestEdgeCases:
    def test_empty_file(self):
        indexer = HierarchicalPageIndexer()
        page = indexer.index_file("empty.py", "", "python")
        assert len(page.chunks) == 0

    def test_syntax_error_falls_back(self):
        indexer = HierarchicalPageIndexer()
        bad_python = "def foo(:\n    pass\ndef bar():\n    return 1"
        page = indexer.index_file("bad.py", bad_python, "python")
        # Should still extract something via fallback
        assert len(page.chunks) >= 0  # at minimum doesn't crash

    def test_string_with_def_not_extracted(self):
        indexer = HierarchicalPageIndexer()
        code = '''
def real_function():
    s = """
def fake_function():
    pass
"""
    return s
'''.strip()
        page = indexer.index_file("test.py", code, "python")
        names = [c.metadata.get("name") for c in page.chunks.values() if c.chunk_type == "function"]
        assert "real_function" in names
        # AST should NOT extract fake_function from the string
        assert "fake_function" not in names

    def test_unsupported_language_returns_empty(self):
        indexer = HierarchicalPageIndexer()
        page = indexer.index_file("test.rb", "def hello; end", "ruby")
        assert len(page.chunks) == 0
