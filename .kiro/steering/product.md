# Product Overview

A hybrid search engine that combines semantic (vector-based) and lexical (fuzzy text matching) search capabilities.

## Core Functionality

- **Document Indexing**: Converts text documents into vector embeddings using sentence-transformers and stores them in DuckDB
- **Hybrid Search**: Combines semantic similarity (cosine distance) with lexical matching (fuzzy string matching) using configurable weights
- **Configurable Ranking**: Allows tuning the balance between semantic understanding and exact text matching

## Use Cases

- Semantic document search where meaning matters more than exact keywords
- Finding similar content across document collections
- Search applications requiring both conceptual and literal matching
