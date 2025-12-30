"""
Evaluation framework for measuring search quality.
"""
import json
import time
from typing import List, Dict, Any
from dataclasses import dataclass, field, asdict
import numpy as np
from loguru import logger


@dataclass
class QueryResult:
    """Single query evaluation result."""
    query: str
    expected_doc_ids: List[int]
    retrieved_doc_ids: List[int]
    precision: float
    recall: float
    f1: float
    mrr: float  # Mean Reciprocal Rank
    ndcg: float  # Normalized Discounted Cumulative Gain
    latency_ms: float


@dataclass
class EvaluationReport:
    """Full evaluation report."""
    num_queries: int
    avg_precision: float
    avg_recall: float
    avg_f1: float
    avg_mrr: float
    avg_ndcg: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    query_results: List[QueryResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def summary(self) -> str:
        return f"""
=== Evaluation Report ===
Queries: {self.num_queries}
Precision: {self.avg_precision:.4f}
Recall: {self.avg_recall:.4f}
F1: {self.avg_f1:.4f}
MRR: {self.avg_mrr:.4f}
NDCG: {self.avg_ndcg:.4f}
Latency (avg): {self.avg_latency_ms:.2f}ms
Latency (p50): {self.p50_latency_ms:.2f}ms
Latency (p95): {self.p95_latency_ms:.2f}ms
Latency (p99): {self.p99_latency_ms:.2f}ms
"""


class SearchEvaluator:
    """Evaluates search quality using standard IR metrics."""
    
    def __init__(self, searcher, docs_df, vectors):
        """
        Initialize evaluator.
        
        Args:
            searcher: Searcher instance.
            docs_df: Documents DataFrame.
            vectors: Document embeddings.
        """
        self.searcher = searcher
        self.docs_df = docs_df
        self.vectors = vectors
    
    def _precision_at_k(self, retrieved: List[int], relevant: List[int], k: int) -> float:
        """Precision@k: fraction of retrieved docs that are relevant."""
        retrieved_k = retrieved[:k]
        if not retrieved_k:
            return 0.0
        relevant_set = set(relevant)
        hits = sum(1 for doc_id in retrieved_k if doc_id in relevant_set)
        return hits / len(retrieved_k)
    
    def _recall_at_k(self, retrieved: List[int], relevant: List[int], k: int) -> float:
        """Recall@k: fraction of relevant docs that are retrieved."""
        if not relevant:
            return 0.0
        retrieved_k = set(retrieved[:k])
        hits = sum(1 for doc_id in relevant if doc_id in retrieved_k)
        return hits / len(relevant)
    
    def _f1_at_k(self, precision: float, recall: float) -> float:
        """F1 score."""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def _mrr(self, retrieved: List[int], relevant: List[int]) -> float:
        """Mean Reciprocal Rank: 1/rank of first relevant result."""
        relevant_set = set(relevant)
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant_set:
                return 1.0 / (i + 1)
        return 0.0
    
    def _dcg(self, retrieved: List[int], relevant: List[int], k: int) -> float:
        """Discounted Cumulative Gain."""
        relevant_set = set(relevant)
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k]):
            if doc_id in relevant_set:
                # Binary relevance: 1 if relevant, 0 otherwise
                dcg += 1.0 / np.log2(i + 2)  # +2 because i starts at 0
        return dcg
    
    def _ndcg(self, retrieved: List[int], relevant: List[int], k: int) -> float:
        """Normalized DCG."""
        dcg = self._dcg(retrieved, relevant, k)
        # Ideal DCG: all relevant docs at top
        ideal_retrieved = relevant[:k] + [r for r in retrieved if r not in relevant]
        idcg = self._dcg(ideal_retrieved, relevant, k)
        if idcg == 0:
            return 0.0
        return dcg / idcg
    
    def evaluate_query(
        self, 
        query: str, 
        relevant_doc_ids: List[int],
        top_k: int = 10,
        **search_kwargs
    ) -> QueryResult:
        """
        Evaluate a single query.
        
        Args:
            query: Search query.
            relevant_doc_ids: List of relevant document IDs (ground truth).
            top_k: Number of results to retrieve.
            **search_kwargs: Additional arguments for searcher.
        
        Returns:
            QueryResult with metrics.
        """
        # Time the search
        start = time.perf_counter()
        results = self.searcher.search(
            query=query,
            docs_df=self.docs_df,
            vectors=self.vectors,
            top_k=top_k,
            **search_kwargs
        )
        latency_ms = (time.perf_counter() - start) * 1000
        
        retrieved_ids = [doc_id for _, _, doc_id in results]
        
        precision = self._precision_at_k(retrieved_ids, relevant_doc_ids, top_k)
        recall = self._recall_at_k(retrieved_ids, relevant_doc_ids, top_k)
        f1 = self._f1_at_k(precision, recall)
        mrr = self._mrr(retrieved_ids, relevant_doc_ids)
        ndcg = self._ndcg(retrieved_ids, relevant_doc_ids, top_k)
        
        return QueryResult(
            query=query,
            expected_doc_ids=relevant_doc_ids,
            retrieved_doc_ids=retrieved_ids,
            precision=precision,
            recall=recall,
            f1=f1,
            mrr=mrr,
            ndcg=ndcg,
            latency_ms=latency_ms
        )
    
    def evaluate(
        self, 
        test_queries: List[Dict[str, Any]],
        top_k: int = 10,
        **search_kwargs
    ) -> EvaluationReport:
        """
        Evaluate multiple queries.
        
        Args:
            test_queries: List of {"query": str, "relevant_doc_ids": List[int]}.
            top_k: Number of results per query.
            **search_kwargs: Additional arguments for searcher.
        
        Returns:
            EvaluationReport with aggregated metrics.
        """
        results = []
        latencies = []
        
        for tq in test_queries:
            result = self.evaluate_query(
                query=tq["query"],
                relevant_doc_ids=tq["relevant_doc_ids"],
                top_k=top_k,
                **search_kwargs
            )
            results.append(result)
            latencies.append(result.latency_ms)
        
        latencies = np.array(latencies)
        
        return EvaluationReport(
            num_queries=len(results),
            avg_precision=np.mean([r.precision for r in results]),
            avg_recall=np.mean([r.recall for r in results]),
            avg_f1=np.mean([r.f1 for r in results]),
            avg_mrr=np.mean([r.mrr for r in results]),
            avg_ndcg=np.mean([r.ndcg for r in results]),
            avg_latency_ms=np.mean(latencies),
            p50_latency_ms=np.percentile(latencies, 50),
            p95_latency_ms=np.percentile(latencies, 95),
            p99_latency_ms=np.percentile(latencies, 99),
            query_results=results
        )
    
    def load_test_set(self, path: str) -> List[Dict[str, Any]]:
        """
        Load test queries from JSON file.
        
        Expected format:
        [
            {"query": "search query", "relevant_doc_ids": [1, 2, 3]},
            ...
        ]
        """
        with open(path, 'r') as f:
            return json.load(f)
    
    def compare_configs(
        self,
        test_queries: List[Dict[str, Any]],
        configs: List[Dict[str, Any]],
        top_k: int = 10
    ) -> Dict[str, EvaluationReport]:
        """
        Compare different search configurations.
        
        Args:
            test_queries: Test query set.
            configs: List of {"name": str, **search_kwargs}.
            top_k: Results per query.
        
        Returns:
            Dict mapping config name to EvaluationReport.
        """
        reports = {}
        
        for config in configs:
            name = config.pop("name")
            logger.info(f"Evaluating config: {name}")
            report = self.evaluate(test_queries, top_k=top_k, **config)
            reports[name] = report
            config["name"] = name  # Restore
        
        return reports


def create_test_set_from_feedback(db_path: str = "index.duckdb") -> List[Dict[str, Any]]:
    """
    Create a test set from user feedback data.
    
    Uses queries where users marked documents as relevant.
    """
    import duckdb
    
    test_queries = []
    
    with duckdb.connect(db_path) as con:
        results = con.execute("""
            SELECT 
                qh.query_text,
                ARRAY_AGG(f.doc_id) as relevant_docs
            FROM query_history qh
            JOIN feedback f ON qh.query_id = f.query_id
            WHERE f.relevance_score >= 3 OR f.clicked = TRUE
            GROUP BY qh.query_text
            HAVING COUNT(*) >= 1
        """).fetchall()
        
        for query_text, relevant_docs in results:
            test_queries.append({
                "query": query_text,
                "relevant_doc_ids": list(relevant_docs)
            })
    
    return test_queries
