"""
Command-line interface for the search engine.

Usage:
    python -m search_engine search "your query"
    python -m search_engine index data/
    python -m search_engine stats
"""
import argparse
import sys
import json
from pathlib import Path
from loguru import logger

from .indexer import Indexer
from .core import Searcher


def setup_logging(verbose: bool = False):
    """Configure logging."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level)


def cmd_search(args):
    """Search command."""
    with Indexer(db_path=args.db) as indexer:
        docs_df, docs = indexer.get_all_documents()
        if len(docs) == 0:
            print("No documents indexed. Run 'index' first.")
            return 1
        vectors = indexer.embed(docs)
    
    searcher = Searcher(db_path=args.db, use_faiss=args.faiss)
    results = searcher.search(
        query=args.query,
        docs_df=docs_df,
        vectors=vectors,
        top_k=args.top_k,
        semantic_weight=args.semantic_weight,
        lexical_weight=1.0 - args.semantic_weight,
        use_learned_weights=args.learned
    )
    
    if args.json:
        output = [{"score": s, "content": c, "doc_id": d} for s, c, d in results]
        print(json.dumps(output, indent=2))
    else:
        print(f"\nResults for: '{args.query}'\n" + "=" * 50)
        for i, (score, content, doc_id) in enumerate(results, 1):
            preview = content[:100].replace('\n', ' ')
            if len(content) > 100:
                preview += "..."
            print(f"{i}. [Score: {score:.4f}] (doc_id={doc_id})")
            print(f"   {preview}\n")
    
    return 0


def cmd_index(args):
    """Index command."""
    path = Path(args.path)
    
    with Indexer(db_path=args.db, use_faiss=args.faiss) as indexer:
        if path.is_dir():
            docs_df, vectors = indexer.index_from_directory(str(path), args.extensions)
            print(f"Indexed {len(docs_df)} documents from {path}")
        elif path.is_file():
            content = path.read_text(encoding='utf-8')
            docs_df, vectors = indexer.index_documents([content], [str(path)])
            print(f"Indexed 1 document: {path}")
        else:
            print(f"Path not found: {path}")
            return 1
    
    return 0


def cmd_add(args):
    """Add documents to existing index."""
    path = Path(args.path)
    
    with Indexer(db_path=args.db) as indexer:
        if path.is_dir():
            from .extractor import discover_documents
            doc_infos = discover_documents(path, args.extensions)
            if doc_infos:
                docs = [d['content'] for d in doc_infos]
                paths = [d['path'] for d in doc_infos]
                indexer.add_documents(docs, paths)
                print(f"Added {len(docs)} documents from {path}")
            else:
                print("No documents found.")
        elif path.is_file():
            content = path.read_text(encoding='utf-8')
            indexer.add_documents([content], [str(path)])
            print(f"Added 1 document: {path}")
        else:
            print(f"Path not found: {path}")
            return 1
    
    return 0


def cmd_stats(args):
    """Show statistics."""
    searcher = Searcher(db_path=args.db)
    stats = searcher.get_stats()
    
    with Indexer(db_path=args.db) as indexer:
        docs_df, _ = indexer.get_all_documents()
        doc_count = len(docs_df)
    
    print("\n=== Search Engine Statistics ===\n")
    print(f"Documents indexed: {doc_count}")
    print(f"Total queries: {stats.get('total_queries', 0)}")
    print(f"Unique queries: {stats.get('unique_queries', 0)}")
    print(f"Total feedback: {stats.get('total_feedback', 0)}")
    print(f"Avg relevance: {stats.get('avg_relevance', 'N/A')}")
    
    if args.json:
        stats['documents'] = doc_count
        print(json.dumps(stats, indent=2))
    
    return 0


def cmd_export(args):
    """Export index to file."""
    with Indexer(db_path=args.db) as indexer:
        docs_df, docs = indexer.get_all_documents()
        vectors = indexer.embed(docs) if docs else None
    
    export_data = {
        'docs': docs_df.to_dicts() if len(docs_df) > 0 else [],
        'vectors': vectors.tolist() if vectors is not None else []
    }
    
    with open(args.output, 'w') as f:
        json.dump(export_data, f)
    
    print(f"Exported {len(docs)} documents to {args.output}")
    return 0


def cmd_import(args):
    """Import index from file."""
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    docs = [d['content'] for d in data['docs']]
    paths = [d.get('source_path', '') for d in data['docs']]
    
    with Indexer(db_path=args.db) as indexer:
        indexer.index_documents(docs, paths)
    
    print(f"Imported {len(docs)} documents from {args.input}")
    return 0


def cmd_serve(args):
    """Start API server."""
    import uvicorn
    print(f"Starting API server on {args.host}:{args.port}")
    uvicorn.run(
        "search_engine.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="search_engine",
        description="Hybrid Search Engine CLI"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--db", default="index.duckdb", help="Database path")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search documents")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("-k", "--top-k", type=int, default=5, help="Number of results")
    search_parser.add_argument("-w", "--semantic-weight", type=float, default=0.7, help="Semantic weight (0-1)")
    search_parser.add_argument("--faiss", action="store_true", help="Use FAISS")
    search_parser.add_argument("--learned", action="store_true", help="Use learned weights")
    search_parser.add_argument("--json", action="store_true", help="JSON output")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Index documents")
    index_parser.add_argument("path", help="File or directory to index")
    index_parser.add_argument("-e", "--extensions", nargs="+", help="File extensions")
    index_parser.add_argument("--faiss", action="store_true", help="Build FAISS index")
    
    # Add command
    add_parser = subparsers.add_parser("add", help="Add to existing index")
    add_parser.add_argument("path", help="File or directory to add")
    add_parser.add_argument("-e", "--extensions", nargs="+", help="File extensions")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show statistics")
    stats_parser.add_argument("--json", action="store_true", help="JSON output")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export index")
    export_parser.add_argument("-o", "--output", default="index_export.json", help="Output file")
    
    # Import command
    import_parser = subparsers.add_parser("import", help="Import index")
    import_parser.add_argument("-i", "--input", required=True, help="Input file")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port")
    serve_parser.add_argument("--reload", action="store_true", help="Auto-reload")
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    if args.command is None:
        parser.print_help()
        return 1
    
    commands = {
        "search": cmd_search,
        "index": cmd_index,
        "add": cmd_add,
        "stats": cmd_stats,
        "export": cmd_export,
        "import": cmd_import,
        "serve": cmd_serve,
    }
    
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
