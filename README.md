# SyntaxSage

Vectorless code search for your codebase. Indexes source files with BM25 + a hierarchical page structure, then lets you search and query with transparent relevance scores — no embedding API required.

## Install

```bash
git clone <repo-url>
cd SyntaxSage
pip install -e .
```

## Quick start

**1. Index your repository**

```bash
syntaxsage index-vectorless --repo-path ./my-project
```

This creates `.syntaxsage/bm25_index.json` and `.syntaxsage/pageindex.json` inside `my-project/`.

**2. Search**

```bash
syntaxsage search-vectorless --query "authentication middleware"
```

Run from the indexed repo directory (or pass explicit `--bm25-index` / `--pageindex` paths).

**3. Streamlit UI**

```bash
streamlit run app.py
```

Open the sidebar to point the UI at your index files and adjust result count.

## Configuration

Create a `.env` file in the project root (loaded automatically):

```
ANTHROPIC_API_KEY=sk-ant-...
```

Or export it in your shell:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

The LLM key is only required when using `EnhancedResponseGenerator` for AI-generated answers. Pure search works without it.

## CLI reference

```
syntaxsage --help
syntaxsage --version
syntaxsage index-vectorless --help
syntaxsage search-vectorless --help
```

| Option | Default | Description |
|---|---|---|
| `--repo-path` | required | Directory to index |
| `--extensions` | `.py,.js,.ts,.tsx` | File extensions to include |
| `--bm25-index` | `<repo>/.syntaxsage/bm25_index.json` | BM25 index output path |
| `--pageindex` | `<repo>/.syntaxsage/pageindex.json` | PageIndex output path |
| `--query` | required | Search query |
| `--top-k` | `5` | Number of results |

## Architecture

```
ingest/
  code_parser.py          Language detection
  page_indexer.py         AST-based chunking (class / method / function / import)
  vectorless_processor.py Orchestrates indexing a full repo

retrieval/
  bm25_index.py           Custom Okapi BM25 with code-aware tokenization
  hybrid_retriever.py     BM25 + structural re-ranking

generation/
  enhanced_generator.py   RAG answer generation via Claude API

cli.py                    Click CLI entry point
app.py                    Streamlit UI
config.py                 Settings (reads .env automatically)
```

Results include BM25 score, combined relevance %, chunk type, and a file > class > method breadcrumb trail.

## Development

```bash
pytest tests/
```
