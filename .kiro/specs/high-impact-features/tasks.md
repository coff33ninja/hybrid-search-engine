# Implementation Plan: High-Impact Features

## Overview

This plan implements six high-impact features in phases, with each feature building on shared infrastructure. The implementation prioritizes core functionality first, then adds advanced capabilities.

## Tasks

- [ ] 1. Set up shared infrastructure and dependencies
  - Add new dependencies to requirements.txt (hypothesis, fasttext, datasketch, celery/rq)
  - Create new module files: autocomplete.py, semantic_cache.py, language.py, deduplication.py, metadata.py, jobs.py
  - Extend DuckDB schema with new columns (language, fingerprint, is_duplicate, canonical_doc_id, metadata)
  - _Requirements: All_

- [ ] 2. Implement Metadata Filtering (foundation for other features)
  - [ ] 2.1 Create MetadataStore class for storing/retrieving document metadata
    - Implement set(), get(), query() methods
    - Use DuckDB JSON column for flexible schema
    - _Requirements: 5.1_
  - [ ] 2.2 Implement filter expression parser
    - Support AND, OR, NOT operators
    - Support field:value, field:>value, field:[v1,v2] syntax
    - Parse into FilterAST
    - _Requirements: 5.2, 5.3, 5.4, 5.5_
  - [ ] 2.3 Implement filter application logic
    - Convert FilterAST to SQL WHERE clause
    - Apply filters before vector search when possible
    - _Requirements: 5.6_
  - [ ] 2.4 Write property test for metadata filter boolean logic
    - **Property 11: Metadata Filter Boolean Logic**
    - **Validates: Requirements 5.2, 5.3, 5.4, 5.5**
  - [ ] 2.5 Write property test for metadata round-trip
    - **Property 12: Metadata Round-Trip**
    - **Validates: Requirements 5.1**

- [ ] 3. Checkpoint - Metadata filtering complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 4. Implement Document Deduplication
  - [ ] 4.1 Create MinHash fingerprint computation
    - Implement MinHashSignature class with jaccard_similarity method
    - Use datasketch library for efficient MinHash
    - _Requirements: 4.1_
  - [ ] 4.2 Implement LSH index for duplicate detection
    - O(1) average lookup time
    - Configurable number of bands and rows
    - _Requirements: 4.4_
  - [ ] 4.3 Implement DeduplicationEngine with configurable actions
    - Support SKIP, MERGE, FLAG actions
    - Integrate with Indexer.index_documents()
    - _Requirements: 4.2, 4.3_
  - [ ] 4.4 Add API endpoints for duplicate management
    - GET /documents/duplicates - list duplicates
    - POST /documents/{id}/mark-duplicate - manual marking
    - _Requirements: 4.5, 4.6_
  - [ ] 4.5 Write property test for duplicate detection consistency
    - **Property 9: Duplicate Detection Consistency**
    - **Validates: Requirements 4.2**
  - [ ] 4.6 Write property test for duplicate action correctness
    - **Property 10: Duplicate Action Correctness**
    - **Validates: Requirements 4.3**

- [ ] 5. Checkpoint - Deduplication complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. Implement Multi-Language Support
  - [ ] 6.1 Create LanguageDetector class
    - Use fasttext for language detection
    - Implement detect() and detect_batch() methods
    - _Requirements: 3.1_
  - [ ] 6.2 Integrate multilingual embedding model
    - Add paraphrase-multilingual-MiniLM-L12-v2 as option
    - Update Indexer to use multilingual model when configured
    - _Requirements: 3.2, 3.3_
  - [ ] 6.3 Add language metadata to indexing pipeline
    - Auto-detect language during indexing
    - Store in docs.language column
    - _Requirements: 3.4_
  - [ ] 6.4 Implement language filtering in search
    - Add language parameter to search endpoint
    - Filter results by language metadata
    - _Requirements: 3.5_
  - [ ] 6.5 Write property test for language filter correctness
    - **Property 8: Language Filter Correctness**
    - **Validates: Requirements 3.5**
  - [ ] 6.6 Write property test for cross-lingual search
    - **Property 7: Cross-Lingual Search**
    - **Validates: Requirements 3.3**

- [ ] 7. Checkpoint - Multi-language support complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 8. Implement Semantic Caching
  - [ ] 8.1 Create LSHIndex class for similarity lookup
    - Implement add() and query() methods
    - Use random hyperplane LSH for cosine similarity
    - _Requirements: 2.5_
  - [ ] 8.2 Implement CacheBackend interface with InMemory and Redis implementations
    - Support TTL and max_size configuration
    - Implement LRU eviction for InMemory backend
    - _Requirements: 2.7, 2.6_
  - [ ] 8.3 Create SemanticCache class
    - Implement get() with LSH similarity lookup
    - Implement set() with embedding storage
    - Log cache hits for monitoring
    - _Requirements: 2.1, 2.2, 2.3, 2.4_
  - [ ] 8.4 Integrate cache with Searcher
    - Check cache before computing search
    - Store results after search
    - Add cache_enabled parameter
    - _Requirements: 2.3_
  - [ ] 8.5 Write property test for cache similarity threshold
    - **Property 3: Semantic Cache Similarity Threshold**
    - **Validates: Requirements 2.2, 2.3**
  - [ ] 8.6 Write property test for cache LRU eviction
    - **Property 4: Cache LRU Eviction**
    - **Validates: Requirements 2.6**
  - [ ] 8.7 Write property test for cache round-trip
    - **Property 5: Cache Round-Trip**
    - **Validates: Requirements 2.1**

- [ ] 9. Checkpoint - Semantic caching complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 10. Implement Query Autocomplete
  - [ ] 10.1 Create PrefixTrie class
    - Implement insert(), search_prefix(), fuzzy_search() methods
    - Store query frequency for ranking
    - _Requirements: 1.2, 1.3_
  - [ ] 10.2 Implement AutocompleteEngine
    - Combine prefix matching, fuzzy matching, and semantic similarity
    - Rank by popularity and relevance
    - _Requirements: 1.1, 1.5_
  - [ ] 10.3 Add query history tracking
    - Log queries to query_suggestions table
    - Update frequency on repeated queries
    - _Requirements: 1.6_
  - [ ] 10.4 Add /autocomplete API endpoint
    - Accept partial query, return suggestions
    - Target < 50ms response time
    - _Requirements: 1.1_
  - [ ] 10.5 Write property test for autocomplete response constraints
    - **Property 1: Autocomplete Response Constraints**
    - **Validates: Requirements 1.1, 1.5**
  - [ ] 10.6 Write property test for autocomplete fuzzy tolerance
    - **Property 2: Autocomplete Fuzzy Tolerance**
    - **Validates: Requirements 1.3**

- [ ] 11. Checkpoint - Autocomplete complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 12. Implement Async Indexing
  - [ ] 12.1 Create JobQueue class with InMemory backend
    - Implement enqueue(), get_status(), cancel() methods
    - Support max_concurrent limit
    - _Requirements: 6.1, 6.5, 6.7_
  - [ ] 12.2 Implement job worker with retry logic
    - Process documents in batches
    - Exponential backoff on failure
    - Update progress during processing
    - _Requirements: 6.4, 6.6_
  - [ ] 12.3 Add webhook notification support
    - Call webhook URL on job completion
    - Include job status and results in payload
    - _Requirements: 6.3_
  - [ ] 12.4 Add job management API endpoints
    - POST /jobs/index - create async indexing job
    - GET /jobs/{id} - get job status
    - DELETE /jobs/{id} - cancel job
    - _Requirements: 6.1, 6.2_
  - [ ] 12.5 Integrate with existing /index endpoint
    - Auto-switch to async for batches > 100 docs
    - Return job_id instead of immediate results
    - _Requirements: 6.1_
  - [ ] 12.6 Write property test for async job lifecycle
    - **Property 13: Async Job Lifecycle**
    - **Validates: Requirements 6.1, 6.2, 6.6**
  - [ ] 12.7 Write property test for job retry behavior
    - **Property 14: Job Retry Behavior**
    - **Validates: Requirements 6.4**
  - [ ] 12.8 Write property test for job cancellation
    - **Property 15: Job Cancellation**
    - **Validates: Requirements 6.5**
  - [ ] 12.9 Write property test for concurrent job limit
    - **Property 16: Concurrent Job Limit**
    - **Validates: Requirements 6.7**

- [ ] 13. Final checkpoint - All features complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 14. Integration and documentation
  - [ ] 14.1 Update README with new features and API endpoints
  - [ ] 14.2 Add configuration examples for each feature
  - [ ] 14.3 Update demo script (main.py) to showcase new features

## Notes

- All property-based tests are required for comprehensive coverage
- Each checkpoint ensures incremental validation before proceeding
- Features are ordered by dependency: Metadata → Deduplication → Language → Cache → Autocomplete → Jobs
- Property tests use the `hypothesis` library with minimum 100 iterations
