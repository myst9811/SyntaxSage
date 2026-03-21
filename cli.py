import sys
from pathlib import Path

import click

from ingest.vectorless_processor import VectorlessDocumentProcessor
from retrieval.hybrid_retriever import HybridRetriever


@click.group()
def cli():
    """SyntaxSage: Vectorless Code Search RAG"""
    pass


@cli.command()
@click.option('--repo-path', required=True, type=click.Path(exists=True, file_okay=False),
              help='Path to the code repository to index.')
@click.option('--bm25-index', default='bm25_index.json', show_default=True, help='Output path for BM25 index.')
@click.option('--pageindex', default='pageindex.json', show_default=True, help='Output path for PageIndex.')
@click.option('--extensions', default='.py,.js,.ts,.tsx', show_default=True,
              help='Comma-separated file extensions to index.')
def index_vectorless(repo_path, bm25_index, pageindex, extensions):
    """Index a repository with BM25 + PageIndex (vectorless)."""
    exts = [e.strip() if e.strip().startswith('.') else f'.{e.strip()}' for e in extensions.split(',')]
    click.echo(f"Indexing {repo_path} (extensions: {', '.join(exts)}) ...")
    try:
        processor = VectorlessDocumentProcessor()
        processor.process_repository(repo_path, extensions=exts)
        if processor.bm25_index.total_docs == 0:
            click.echo("Warning: no code chunks found. Check the repo path and extensions.", err=True)
            return
        processor.save_indexes(bm25_index, pageindex)
        click.echo(f"Done. Indexes saved to {bm25_index} and {pageindex}.")
    except Exception as e:
        click.echo(f"Error during indexing: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--query', required=True, help='Search query.')
@click.option('--bm25-index', default='bm25_index.json', show_default=True,
              type=click.Path(exists=True), help='Path to BM25 index.')
@click.option('--pageindex', default='pageindex.json', show_default=True,
              type=click.Path(exists=True), help='Path to PageIndex.')
@click.option('--top-k', default=5, show_default=True, help='Number of results to return.')
def search_vectorless(query, bm25_index, pageindex, top_k):
    """Search the indexed codebase using vectorless BM25 retrieval."""
    click.echo(f"Searching: {query}\n")
    try:
        retriever = HybridRetriever.from_indexes(bm25_index, pageindex)
        results = retriever.search(query, top_k=top_k)
    except Exception as e:
        click.echo(f"Error loading indexes: {e}", err=True)
        click.echo("Have you indexed a repository first?  python cli.py index-vectorless --repo-path <path>", err=True)
        sys.exit(1)

    if not results:
        click.echo("No results found. Try different search terms.")
        return

    for i, result in enumerate(results, 1):
        name = result.metadata.get('name') or 'N/A'
        file_ = result.metadata.get('file') or 'N/A'
        type_ = result.metadata.get('type') or 'N/A'
        click.echo(f"[{i}] {name}")
        click.echo(f"    File  : {file_}")
        click.echo(f"    Score : {result.combined_score:.0%}  (BM25: {result.bm25_score:.3f})")
        click.echo(f"    Type  : {type_}")
        if result.breadcrumb:
            click.echo(f"    Path  : {' > '.join(result.breadcrumb)}")
        click.echo()


if __name__ == '__main__':
    cli()
