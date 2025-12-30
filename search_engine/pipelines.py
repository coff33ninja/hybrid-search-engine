"""
Pre-built search pipelines for common use cases.

Each pipeline combines multiple components (chunking, search, reranking, etc.)
into a ready-to-use configuration.
"""
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

import numpy as np
from loguru import logger

from .core import Searcher
from .indexer import Indexer
from .chunker import get_chunker, Chunk
from .highlighter import Highlighter, HighlightedResult
from .bm25 import BM25


# Type alias using Tuple
SearchResult = Tuple[float, str, int]  # (score, content, doc_id)


@dataclass
class PipelineResult:
    """Result from a search pipeline."""
    query: str
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    highlighted: Optional[List[HighlightedResult]] = None


class BasePipeline:
    """Base class for search pipelines."""
    
    def __init__(self, db_path: str = "index.duckdb", enable_highlighting: bool = False):
        self.db_path = db_path
        self.docs_df = None
        self.vectors = None
        self.searcher = None
        self.highlighter = Highlighter() if enable_highlighting else None
    
    def index(self, documents: List[str], **kwargs):
        """Index documents."""
        raise NotImplementedError
    
    def search(self, query: str, top_k: int = 5, **kwargs) -> PipelineResult:
        """Search documents."""
        raise NotImplementedError
    
    def _highlight_results(
        self, 
        results: List[SearchResult], 
        query: str
    ) -> Optional[List[HighlightedResult]]:
        """Apply highlighting to results if enabled."""
        if self.highlighter is None:
            return None
        return self.highlighter.highlight_results(results, query)


class BasicPipeline(BasePipeline):
    """
    Basic hybrid search pipeline.
    
    Good for: Small to medium document collections, general-purpose search.
    """
    
    def __init__(
        self, 
        db_path: str = "index.duckdb", 
        semantic_weight: float = 0.7,
        enable_highlighting: bool = False
    ):
        super().__init__(db_path, enable_highlighting)
        self.semantic_weight = semantic_weight
        self.lexical_weight = 1.0 - semantic_weight
    
    def index(self, documents: List[str], source_paths: Optional[List[str]] = None):
        with Indexer(db_path=self.db_path) as indexer:
            self.docs_df, self.vectors = indexer.index_documents(documents, source_paths)
        self.searcher = Searcher(db_path=self.db_path)
        logger.info(f"BasicPipeline indexed {len(documents)} documents")
    
    def search(self, query: str, top_k: int = 5) -> PipelineResult:
        results: List[SearchResult] = self.searcher.search(
            query=query,
            docs_df=self.docs_df,
            vectors=self.vectors,
            top_k=top_k,
            semantic_weight=self.semantic_weight,
            lexical_weight=self.lexical_weight
        )
        
        return PipelineResult(
            query=query,
            results=[
                {"score": s, "content": c, "doc_id": d}
                for s, c, d in results
            ],
            metadata={"pipeline": "basic", "weights": {"semantic": self.semantic_weight}},
            highlighted=self._highlight_results(results, query)
        )


class ChunkedPipeline(BasePipeline):
    """
    Pipeline with document chunking for long documents.
    
    Good for: Long documents, articles, books, technical documentation.
    """
    
    def __init__(
        self, 
        db_path: str = "index.duckdb",
        chunk_strategy: str = "paragraph",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        super().__init__(db_path)
        self.chunk_strategy = chunk_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: List[Chunk] = []
        self.chunk_to_doc: Dict[int, int] = {}  # chunk_id -> original doc_id
    
    def index(self, documents: List[str], source_paths: List[str] = None):
        chunker = get_chunker(
            self.chunk_strategy,
            max_length=self.chunk_size if self.chunk_strategy == "paragraph" else None,
            window_size=self.chunk_size if self.chunk_strategy == "sliding" else None,
            overlap=self.chunk_overlap if self.chunk_strategy == "sliding" else None
        )
        
        all_chunks = []
        chunk_contents = []
        chunk_paths = []
        
        for doc_id, doc in enumerate(documents):
            path = source_paths[doc_id] if source_paths else ""
            doc_chunks = chunker.chunk(doc, doc_id=doc_id, source_path=path)
            
            for chunk in doc_chunks:
                self.chunk_to_doc[len(all_chunks)] = doc_id
                all_chunks.append(chunk)
                chunk_contents.append(chunk.content)
                chunk_paths.append(path)
        
        self.chunks = all_chunks
        
        with Indexer(db_path=self.db_path) as indexer:
            self.docs_df, self.vectors = indexer.index_documents(chunk_contents, chunk_paths)
        
        self.searcher = Searcher(db_path=self.db_path)
        logger.info(f"ChunkedPipeline: {len(documents)} docs → {len(all_chunks)} chunks")
    
    def search(self, query: str, top_k: int = 5, return_parent: bool = False) -> PipelineResult:
        results = self.searcher.search(
            query=query,
            docs_df=self.docs_df,
            vectors=self.vectors,
            top_k=top_k * 2 if return_parent else top_k  # Get more for dedup
        )
        
        output = []
        seen_docs = set()
        
        for score, content, chunk_id in results:
            original_doc_id = self.chunk_to_doc.get(chunk_id, chunk_id)
            
            if return_parent and original_doc_id in seen_docs:
                continue
            seen_docs.add(original_doc_id)
            
            output.append({
                "score": score,
                "content": content,
                "chunk_id": chunk_id,
                "original_doc_id": original_doc_id
            })
            
            if len(output) >= top_k:
                break
        
        return PipelineResult(
            query=query,
            results=output,
            metadata={
                "pipeline": "chunked",
                "strategy": self.chunk_strategy,
                "total_chunks": len(self.chunks)
            }
        )


class RerankedPipeline(BasePipeline):
    """
    Pipeline with cross-encoder reranking for higher precision.
    
    Good for: When accuracy matters more than speed, Q&A systems.
    """
    
    def __init__(
        self,
        db_path: str = "index.duckdb",
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        initial_k: int = 20
    ):
        super().__init__(db_path)
        self.rerank_model = rerank_model
        self.initial_k = initial_k
        self._reranker = None
    
    @property
    def reranker(self):
        if self._reranker is None:
            from .reranker import CrossEncoderReranker
            self._reranker = CrossEncoderReranker(model_name=self.rerank_model)
        return self._reranker
    
    def index(self, documents: List[str], source_paths: List[str] = None):
        with Indexer(db_path=self.db_path) as indexer:
            self.docs_df, self.vectors = indexer.index_documents(documents, source_paths)
        self.searcher = Searcher(db_path=self.db_path)
        logger.info(f"RerankedPipeline indexed {len(documents)} documents")
    
    def search(self, query: str, top_k: int = 5) -> PipelineResult:
        # Initial retrieval
        initial_results = self.searcher.search(
            query=query,
            docs_df=self.docs_df,
            vectors=self.vectors,
            top_k=self.initial_k
        )
        
        # Rerank
        reranked = self.reranker.rerank(query, initial_results, top_k=top_k)
        
        return PipelineResult(
            query=query,
            results=[
                {"score": s, "content": c, "doc_id": d, "reranked": True}
                for s, c, d in reranked
            ],
            metadata={
                "pipeline": "reranked",
                "rerank_model": self.rerank_model,
                "initial_k": self.initial_k
            }
        )


class BM25Pipeline(BasePipeline):
    """
    Pure BM25 lexical search pipeline.
    
    Good for: Keyword-heavy search, when exact terms matter.
    """
    
    def __init__(self, db_path: str = "index.duckdb", k1: float = 1.5, b: float = 0.75):
        super().__init__(db_path)
        self.bm25 = BM25(k1=k1, b=b)
        self.documents: List[str] = []
    
    def index(self, documents: List[str], source_paths: List[str] = None):
        self.documents = documents
        self.bm25.fit(documents)
        logger.info(f"BM25Pipeline indexed {len(documents)} documents")
    
    def search(self, query: str, top_k: int = 5) -> PipelineResult:
        results = self.bm25.search(query, top_k=top_k)
        
        return PipelineResult(
            query=query,
            results=[
                {"score": score, "content": self.documents[idx], "doc_id": idx}
                for idx, score in results
            ],
            metadata={"pipeline": "bm25", "k1": self.bm25.k1, "b": self.bm25.b}
        )


class HybridBM25Pipeline(BasePipeline):
    """
    Hybrid pipeline combining semantic search with BM25.
    
    Good for: Best of both worlds - meaning + keywords.
    """
    
    def __init__(
        self,
        db_path: str = "index.duckdb",
        semantic_weight: float = 0.6,
        bm25_weight: float = 0.4
    ):
        super().__init__(db_path)
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        self.bm25 = BM25()
        self.documents: List[str] = []
    
    def index(self, documents: List[str], source_paths: List[str] = None):
        self.documents = documents
        
        # Index for semantic search
        with Indexer(db_path=self.db_path) as indexer:
            self.docs_df, self.vectors = indexer.index_documents(documents, source_paths)
        self.searcher = Searcher(db_path=self.db_path)
        
        # Fit BM25
        self.bm25.fit(documents)
        
        logger.info(f"HybridBM25Pipeline indexed {len(documents)} documents")
    
    def search(self, query: str, top_k: int = 5) -> PipelineResult:
        # Semantic scores
        semantic_results = self.searcher.search(
            query=query,
            docs_df=self.docs_df,
            vectors=self.vectors,
            top_k=len(self.documents),
            semantic_weight=1.0,
            lexical_weight=0.0
        )
        semantic_scores = {d: s for s, _, d in semantic_results}
        
        # BM25 scores
        bm25_scores = self.bm25.score_batch(query)
        
        # Normalize
        max_sem = max(semantic_scores.values()) if semantic_scores else 1
        max_bm25 = bm25_scores.max() if bm25_scores.max() > 0 else 1
        
        # Combine scores
        combined: List[SearchResult] = []
        for i, doc in enumerate(self.documents):
            sem_score = semantic_scores.get(i, 0) / max_sem
            bm25_score = bm25_scores[i] / max_bm25
            hybrid = (sem_score * self.semantic_weight) + (bm25_score * self.bm25_weight)
            combined.append((hybrid, doc, i))
        
        combined.sort(key=lambda x: x[0], reverse=True)
        top_results = combined[:top_k]
        
        return PipelineResult(
            query=query,
            results=[
                {"score": s, "content": c, "doc_id": d}
                for s, c, d in top_results
            ],
            metadata={
                "pipeline": "hybrid_bm25",
                "semantic_weight": self.semantic_weight,
                "bm25_weight": self.bm25_weight
            },
            highlighted=self._highlight_results(top_results, query)
        )


class RAGPipeline(BasePipeline):
    """
    Full RAG pipeline: retrieve → rerank → generate.
    
    Good for: Question answering, chatbots, knowledge bases.
    """
    
    def __init__(
        self,
        db_path: str = "index.duckdb",
        llm_provider: str = "openai",
        model: str = "gpt-3.5-turbo",
        use_reranking: bool = True
    ):
        super().__init__(db_path)
        self.llm_provider = llm_provider
        self.model = model
        self.use_reranking = use_reranking
        self._rag = None
        self._reranker = None
    
    def index(self, documents: List[str], source_paths: List[str] = None):
        with Indexer(db_path=self.db_path) as indexer:
            self.docs_df, self.vectors = indexer.index_documents(documents, source_paths)
        self.searcher = Searcher(db_path=self.db_path)
        logger.info(f"RAGPipeline indexed {len(documents)} documents")
    
    def search(self, query: str, top_k: int = 5) -> PipelineResult:
        # Retrieve
        results = self.searcher.search(
            query=query,
            docs_df=self.docs_df,
            vectors=self.vectors,
            top_k=top_k * 2 if self.use_reranking else top_k
        )
        
        # Rerank if enabled
        if self.use_reranking:
            from .reranker import CrossEncoderReranker
            if self._reranker is None:
                self._reranker = CrossEncoderReranker()
            results = self._reranker.rerank(query, results, top_k=top_k)
        
        return PipelineResult(
            query=query,
            results=[
                {"score": s, "content": c, "doc_id": d}
                for s, c, d in results
            ],
            metadata={"pipeline": "rag", "reranked": self.use_reranking}
        )
    
    def ask(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Answer a question using RAG."""
        from .rag import RAGPipeline as RAG
        
        if self._rag is None:
            self._rag = RAG(
                searcher=self.searcher,
                docs_df=self.docs_df,
                vectors=self.vectors,
                llm_provider=self.llm_provider,
                model=self.model
            )
        
        response = self._rag.ask(question, top_k=top_k)
        
        return {
            "question": question,
            "answer": response.answer,
            "sources": response.sources,
            "model": response.model
        }


class MultiStagePipeline(BasePipeline):
    """
    Multi-stage retrieval pipeline:
    1. Fast initial retrieval (semantic)
    2. BM25 filtering
    3. Cross-encoder reranking
    
    Good for: Large collections where precision is critical.
    """
    
    def __init__(
        self,
        db_path: str = "index.duckdb",
        stage1_k: int = 100,
        stage2_k: int = 20,
        final_k: int = 5
    ):
        super().__init__(db_path)
        self.stage1_k = stage1_k
        self.stage2_k = stage2_k
        self.final_k = final_k
        self.bm25 = BM25()
        self.documents: List[str] = []
        self._reranker = None
    
    def index(self, documents: List[str], source_paths: List[str] = None):
        self.documents = documents
        
        with Indexer(db_path=self.db_path) as indexer:
            self.docs_df, self.vectors = indexer.index_documents(documents, source_paths)
        self.searcher = Searcher(db_path=self.db_path)
        self.bm25.fit(documents)
        
        logger.info(f"MultiStagePipeline indexed {len(documents)} documents")
    
    def search(self, query: str, top_k: int = None) -> PipelineResult:
        top_k = top_k or self.final_k
        
        # Stage 1: Fast semantic retrieval
        stage1_results = self.searcher.search(
            query=query,
            docs_df=self.docs_df,
            vectors=self.vectors,
            top_k=self.stage1_k,
            semantic_weight=1.0,
            lexical_weight=0.0
        )
        
        # Stage 2: BM25 scoring on candidates
        candidate_ids = np.array([d for _, _, d in stage1_results])
        bm25_scores = [(self.bm25.score(query, i), c, i) for _, c, i in stage1_results]
        bm25_scores.sort(key=lambda x: x[0], reverse=True)
        stage2_results = [(s, c, d) for s, c, d in bm25_scores[:self.stage2_k]]
        
        logger.debug(f"Stage 2 filtered {len(candidate_ids)} candidates to {len(stage2_results)}")
        
        # Stage 3: Cross-encoder reranking
        from .reranker import CrossEncoderReranker
        if self._reranker is None:
            self._reranker = CrossEncoderReranker()
        
        final_results: List[SearchResult] = self._reranker.rerank(query, stage2_results, top_k=top_k)
        
        return PipelineResult(
            query=query,
            results=[
                {"score": s, "content": c, "doc_id": d, "stage": "final"}
                for s, c, d in final_results
            ],
            metadata={
                "pipeline": "multi_stage",
                "stage1_k": self.stage1_k,
                "stage2_k": self.stage2_k,
                "final_k": top_k
            },
            highlighted=self._highlight_results(final_results, query)
        )


class DiversityPipeline(BasePipeline):
    """
    Pipeline that promotes result diversity using MMR.
    
    Good for: Avoiding redundant results, exploration.
    """
    
    def __init__(self, db_path: str = "index.duckdb", lambda_param: float = 0.5):
        super().__init__(db_path)
        self.lambda_param = lambda_param  # Balance relevance vs diversity
    
    def index(self, documents: List[str], source_paths: Optional[List[str]] = None):
        with Indexer(db_path=self.db_path) as indexer:
            self.docs_df, self.vectors = indexer.index_documents(documents, source_paths)
        self.searcher = Searcher(db_path=self.db_path)
        logger.info(f"DiversityPipeline indexed {len(documents)} documents")
    
    def _mmr(
        self,
        query_embedding,
        doc_embeddings,
        doc_scores,
        top_k: int
    ) -> List[int]:
        """Maximal Marginal Relevance selection."""
        from .utils import cosine_sim
        
        selected = []
        remaining = list(range(len(doc_scores)))
        
        while len(selected) < top_k and remaining:
            mmr_scores = []
            
            for idx in remaining:
                relevance = doc_scores[idx]
                
                # Max similarity to already selected
                if selected:
                    similarities = [
                        cosine_sim(doc_embeddings[idx], doc_embeddings[s])
                        for s in selected
                    ]
                    max_sim = max(similarities)
                else:
                    max_sim = 0
                
                # MMR score
                mmr = self.lambda_param * relevance - (1 - self.lambda_param) * max_sim
                mmr_scores.append((idx, mmr))
            
            # Select highest MMR
            best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected.append(best_idx)
            remaining.remove(best_idx)
        
        return selected
    
    def search(self, query: str, top_k: int = 5) -> PipelineResult:
        # Get more candidates for diversity selection
        results = self.searcher.search(
            query=query,
            docs_df=self.docs_df,
            vectors=self.vectors,
            top_k=top_k * 4
        )
        
        if not results:
            return PipelineResult(query=query, results=[], metadata={})
        
        # Get embeddings for MMR
        doc_ids = [d for _, _, d in results]
        doc_embeddings = self.vectors[doc_ids]
        doc_scores = np.array([s for s, _, _ in results])
        
        # Normalize scores
        doc_scores = (doc_scores - doc_scores.min()) / (doc_scores.max() - doc_scores.min() + 1e-8)
        
        # Query embedding
        query_emb = self.searcher.model.encode([query], convert_to_numpy=True)[0]
        
        # MMR selection
        selected_indices = self._mmr(query_emb, doc_embeddings, doc_scores, top_k)
        
        return PipelineResult(
            query=query,
            results=[
                {
                    "score": results[i][0],
                    "content": results[i][1],
                    "doc_id": results[i][2],
                    "diversity_rank": rank
                }
                for rank, i in enumerate(selected_indices)
            ],
            metadata={
                "pipeline": "diversity",
                "lambda": self.lambda_param,
                "method": "mmr"
            }
        )


# Factory function
def create_pipeline(
    pipeline_type: str = "basic",
    **kwargs
) -> BasePipeline:
    """
    Create a search pipeline by name.
    
    Args:
        pipeline_type: One of 'basic', 'chunked', 'reranked', 'bm25', 
                      'hybrid_bm25', 'rag', 'multi_stage', 'diversity'
        **kwargs: Pipeline-specific arguments
    
    Returns:
        Pipeline instance
    """
    pipelines = {
        "basic": BasicPipeline,
        "chunked": ChunkedPipeline,
        "reranked": RerankedPipeline,
        "bm25": BM25Pipeline,
        "hybrid_bm25": HybridBM25Pipeline,
        "rag": RAGPipeline,
        "multi_stage": MultiStagePipeline,
        "diversity": DiversityPipeline,
    }
    
    if pipeline_type not in pipelines:
        raise ValueError(f"Unknown pipeline: {pipeline_type}. Choose from {list(pipelines.keys())}")
    
    return pipelines[pipeline_type](**kwargs)
