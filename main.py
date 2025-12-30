"""
Search Engine Demo - Showcasing all features.

Features demonstrated:
1. Document indexing (from list or directory)
2. Hybrid search (semantic + lexical)
3. FAISS support (optional ANN)
4. Query memory and feedback
5. File watcher (auto-reindex)
6. REST API (run separately with uvicorn)
"""
import sys
from pathlib import Path
from loguru import logger

from search_engine.indexer import Indexer, FAISS_AVAILABLE
from search_engine.core import Searcher
from search_engine.extractor import discover_documents

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")

# Sample documents (fallback if data/ is empty)
SAMPLE_DOCUMENTS = [
    "The quick brown fox jumps over the lazy dog.",
    "A journey of a thousand miles begins with a single step.",
    "To be or not to be, that is the question.",
    "All that glitters is not gold.",
    "The early bird catches the worm.",
    "Actions speak louder than words.",
    "An apple a day keeps the doctor away.",
    "Where there's a will, there's a way.",
    "Machine learning models require large datasets for training.",
    "Python is a popular programming language for data science.",
    "Neural networks can learn complex patterns from data.",
    "The transformer architecture revolutionized natural language processing.",
]


def demo_basic_search():
    """Demo: Basic indexing and search."""
    logger.info("=== Demo: Basic Indexing and Search ===")
    
    with Indexer(db_path="index.duckdb") as indexer:
        docs_df, vectors = indexer.index_documents(SAMPLE_DOCUMENTS)
    
    searcher = Searcher()
    
    queries = [
        "wise sayings about starting",
        "machine learning and AI",
        "programming languages",
    ]
    
    for query in queries:
        logger.info(f"\nQuery: '{query}'")
        results = searcher.search(
            query=query,
            docs_df=docs_df,
            vectors=vectors,
            top_k=3,
            semantic_weight=0.7,
            lexical_weight=0.3
        )
        
        for i, (score, doc, doc_id) in enumerate(results, 1):
            print(f"  {i}. [Score: {score:.4f}] \"{doc[:60]}...\"" if len(doc) > 60 else f"  {i}. [Score: {score:.4f}] \"{doc}\"")


def demo_directory_indexing():
    """Demo: Index documents from data/ directory."""
    logger.info("\n=== Demo: Directory Indexing ===")
    
    data_dir = Path("data")
    
    # Create sample files if data/ is empty
    if not data_dir.exists() or not any(data_dir.iterdir()):
        data_dir.mkdir(exist_ok=True)
        logger.info("Creating sample files in data/...")
        
        (data_dir / "readme.txt").write_text(
            "Welcome to the hybrid search engine.\n"
            "This engine combines semantic and lexical search for better results."
        )
        (data_dir / "notes.md").write_text(
            "# Development Notes\n\n"
            "- Use sentence-transformers for embeddings\n"
            "- DuckDB for storage\n"
            "- FAISS for approximate nearest neighbor search"
        )
        (data_dir / "config.json").write_text(
            '{"model": "all-MiniLM-L6-v2", "semantic_weight": 0.7}'
        )
    
    # Discover and show files
    docs = discover_documents(data_dir)
    logger.info(f"Found {len(docs)} documents in data/")
    for doc in docs:
        logger.info(f"  - {doc['path']} ({doc['metadata']['word_count']} words)")
    
    # Index from directory
    with Indexer(db_path="index.duckdb") as indexer:
        docs_df, vectors = indexer.index_from_directory("data")
    
    if len(docs_df) > 0:
        searcher = Searcher()
        results = searcher.search(
            query="search engine configuration",
            docs_df=docs_df,
            vectors=vectors,
            top_k=3
        )
        
        logger.info("\nSearch results for 'search engine configuration':")
        for i, (score, doc, doc_id) in enumerate(results, 1):
            preview = doc[:80].replace('\n', ' ')
            print(f"  {i}. [Score: {score:.4f}] \"{preview}...\"")


def demo_faiss_search():
    """Demo: FAISS approximate nearest neighbor search."""
    logger.info("\n=== Demo: FAISS ANN Search ===")
    
    if not FAISS_AVAILABLE:
        logger.warning("FAISS not installed. Skipping FAISS demo.")
        logger.info("Install with: pip install faiss-cpu")
        return
    
    # Index with FAISS enabled
    with Indexer(db_path="index.duckdb", use_faiss=True) as indexer:
        docs_df, vectors = indexer.index_documents(SAMPLE_DOCUMENTS)
    
    # Search with FAISS
    searcher = Searcher(use_faiss=True)
    results = searcher.search(
        query="artificial intelligence",
        docs_df=docs_df,
        vectors=vectors,
        top_k=3
    )
    
    logger.info("FAISS search results for 'artificial intelligence':")
    for i, (score, doc, doc_id) in enumerate(results, 1):
        print(f"  {i}. [Score: {score:.4f}] \"{doc}\"")


def demo_feedback_loop():
    """Demo: Query memory and feedback system."""
    logger.info("\n=== Demo: Feedback Loop ===")
    
    with Indexer(db_path="index.duckdb") as indexer:
        docs_df, vectors = indexer.index_documents(SAMPLE_DOCUMENTS)
    
    searcher = Searcher(enable_query_memory=True)
    
    # Perform search
    results = searcher.search(
        query="programming",
        docs_df=docs_df,
        vectors=vectors,
        top_k=3,
        semantic_weight=0.8,
        lexical_weight=0.2
    )
    
    # Simulate user feedback
    if results:
        score, doc, doc_id = results[0]
        # Record that user clicked and found it relevant
        searcher.record_feedback(
            query_id=1,  # First query
            doc_id=doc_id,
            relevance_score=5,
            clicked=True
        )
        logger.info(f"Recorded positive feedback for doc_id={doc_id}")
    
    # Show stats
    stats = searcher.get_stats()
    logger.info(f"Query stats: {stats}")


def demo_api_info():
    """Show how to run the REST API."""
    logger.info("\n=== REST API ===")
    print("""
To start the REST API server:

    uvicorn search_engine.api:app --reload --port 8000

API Endpoints:
    GET  /health              - Health check
    POST /index               - Index documents
    POST /index/directory     - Index from directory
    POST /index/add           - Add to existing index
    POST /search              - Perform search
    POST /feedback            - Submit feedback
    GET  /stats               - Get statistics
    POST /watcher/start       - Start file watcher
    POST /watcher/stop        - Stop file watcher
    GET  /documents           - List indexed documents

Example search request:
    curl -X POST http://localhost:8000/search \\
        -H "Content-Type: application/json" \\
        -d '{"query": "machine learning", "top_k": 5}'
""")


def main():
    """Run all demos."""
    logger.info("=== Hybrid Search Engine Demo ===\n")
    
    # Run demos
    demo_basic_search()
    demo_directory_indexing()
    demo_faiss_search()
    demo_feedback_loop()
    demo_api_info()
    
    logger.success("\n=== All demos complete! ===")


if __name__ == "__main__":
    main()
