import streamlit as st
from retrieval.hybrid_retriever import HybridRetriever
from config import settings

st.set_page_config(page_title="SyntaxSage", layout="wide")
st.title("SyntaxSage: Vectorless Code Search")

with st.sidebar:
    st.header("Settings")
    retrieval_mode = st.radio(
        "Retrieval Strategy",
        ["Vectorless (Fast)", "Hybrid (Balanced)"],
    )
    bm25_path = st.text_input("BM25 Index Path", value=settings.bm25_index_path)
    pageindex_path = st.text_input("PageIndex Path", value=settings.pageindex_path)
    top_k = st.slider("Results to show", min_value=1, max_value=20, value=settings.top_k)

    st.divider()
    st.metric("Latency", "~10ms" if retrieval_mode == "Vectorless (Fast)" else "~50ms")
    st.metric("Interpretability", "High")

    st.divider()
    st.info(
        "**Why Vectorless?**\n"
        "- 10x faster than vector search\n"
        "- No embedding API costs\n"
        "- Exact keyword matching\n"
        "- Transparent BM25 scores"
    )

col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input("Search your codebase:", placeholder="e.g. how does authentication work?")

    if query:
        try:
            use_dense = retrieval_mode == "Hybrid (Balanced)"
            retriever = HybridRetriever.from_indexes(bm25_path, pageindex_path, use_dense=use_dense)
            results = retriever.search(query, top_k=top_k)

            if not results:
                st.warning("No results found. Have you indexed a repository first?")
            else:
                st.markdown(f"### {len(results)} Results")
                for i, result in enumerate(results, 1):
                    name = result.metadata.get('name') or 'unknown'
                    label = f"[{result.combined_score:.0%}] {name}"
                    with st.expander(label):
                        left, right = st.columns([3, 1])
                        with left:
                            st.code(result.text, language="python")
                        with right:
                            st.metric("BM25 Score", f"{result.bm25_score:.3f}")
                            st.metric("Combined", f"{result.combined_score:.0%}")
                            st.metric("Type", result.metadata.get('type', 'N/A'))
                            if result.breadcrumb:
                                st.caption("Path: " + " > ".join(result.breadcrumb))
        except FileNotFoundError:
            st.error(
                "Index files not found. Run the indexer first:\n\n"
                f"```\npython cli.py index-vectorless --repo-path <your-repo>\n```"
            )

with col2:
    st.markdown("### How to use")
    st.markdown(
        "1. Index your repo via CLI:\n"
        "```\npython cli.py index-vectorless \\\n  --repo-path ./my_project\n```\n"
        "2. Enter a query above.\n"
        "3. Results are ranked by BM25 + structural score."
    )
