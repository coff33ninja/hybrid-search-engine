"""
RAG (Retrieval-Augmented Generation) integration.

Connects search results to LLMs for answer generation.
"""
import os
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from loguru import logger


@dataclass
class RAGResponse:
    """Response from RAG pipeline."""
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    model: str
    tokens_used: Optional[int] = None


class RAGPipeline:
    """
    RAG pipeline that retrieves documents and generates answers.
    """
    
    def __init__(
        self,
        searcher,
        docs_df,
        vectors,
        llm_provider: str = "openai",
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        max_context_docs: int = 5,
        max_context_chars: int = 4000
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            searcher: Searcher instance.
            docs_df: Documents DataFrame.
            vectors: Document embeddings.
            llm_provider: 'openai', 'anthropic', or 'local'.
            model: Model name.
            api_key: API key (or set via environment variable).
            max_context_docs: Maximum documents to include in context.
            max_context_chars: Maximum characters for context.
        """
        self.searcher = searcher
        self.docs_df = docs_df
        self.vectors = vectors
        self.llm_provider = llm_provider
        self.model = model
        self.api_key = api_key
        self.max_context_docs = max_context_docs
        self.max_context_chars = max_context_chars
    
    def _build_context(self, results: List[Tuple[float, str, int]]) -> str:
        """Build context string from search results."""
        context_parts = []
        total_chars = 0
        
        for i, (score, content, doc_id) in enumerate(results[:self.max_context_docs]):
            # Truncate if needed
            remaining = self.max_context_chars - total_chars
            if remaining <= 0:
                break
            
            truncated = content[:remaining]
            context_parts.append(f"[Document {i+1}]\n{truncated}")
            total_chars += len(truncated)
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build the prompt for the LLM."""
        return f"""Answer the question based on the provided context. If the context doesn't contain enough information, say so.

Context:
{context}

Question: {query}

Answer:"""
    
    def _call_openai(self, prompt: str) -> Tuple[str, int]:
        """Call OpenAI API."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")
        
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided")
        
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        
        answer = response.choices[0].message.content
        tokens = response.usage.total_tokens if response.usage else None
        return answer, tokens
    
    def _call_anthropic(self, prompt: str) -> Tuple[str, int]:
        """Call Anthropic API."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")
        
        api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not provided")
        
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        answer = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens
        return answer, tokens
    
    def _call_local(self, prompt: str) -> Tuple[str, int]:
        """Call local LLM (Ollama)."""
        try:
            import requests
        except ImportError:
            raise ImportError("Install requests: pip install requests")
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"Ollama error: {response.text}")
        
        data = response.json()
        return data.get("response", ""), None
    
    def ask(self, query: str, top_k: int = 5) -> RAGResponse:
        """
        Ask a question and get an answer based on retrieved documents.
        
        Args:
            query: User question.
            top_k: Number of documents to retrieve.
        
        Returns:
            RAGResponse with answer and sources.
        """
        # Retrieve relevant documents
        results = self.searcher.search(
            query=query,
            docs_df=self.docs_df,
            vectors=self.vectors,
            top_k=top_k
        )
        
        # Build context and prompt
        context = self._build_context(results)
        prompt = self._build_prompt(query, context)
        
        # Call LLM
        if self.llm_provider == "openai":
            answer, tokens = self._call_openai(prompt)
        elif self.llm_provider == "anthropic":
            answer, tokens = self._call_anthropic(prompt)
        elif self.llm_provider == "local":
            answer, tokens = self._call_local(prompt)
        else:
            raise ValueError(f"Unknown LLM provider: {self.llm_provider}")
        
        # Build sources
        sources = [
            {"doc_id": doc_id, "score": score, "preview": content[:200]}
            for score, content, doc_id in results
        ]
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            query=query,
            model=self.model,
            tokens_used=tokens
        )


class HyDEPipeline:
    """
    Hypothetical Document Embeddings (HyDE).
    
    Instead of embedding the query directly, generate a hypothetical
    answer and embed that for better retrieval.
    """
    
    def __init__(
        self,
        searcher,
        docs_df,
        vectors,
        llm_provider: str = "openai",
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None
    ):
        self.searcher = searcher
        self.docs_df = docs_df
        self.vectors = vectors
        self.llm_provider = llm_provider
        self.model = model
        self.api_key = api_key
    
    def _generate_hypothetical(self, query: str) -> str:
        """Generate a hypothetical document that would answer the query."""
        prompt = f"""Write a short paragraph that would be a perfect answer to this question. 
Write it as if it's from a document, not as a direct answer.

Question: {query}

Hypothetical document:"""
        
        # Use same LLM calling logic as RAGPipeline
        if self.llm_provider == "openai":
            from openai import OpenAI
            api_key = self.api_key or os.getenv("OPENAI_API_KEY")
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200
            )
            return response.choices[0].message.content
        else:
            # Fallback to original query
            return query
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, str, int]]:
        """
        Search using HyDE - generate hypothetical doc, then search.
        
        Args:
            query: User query.
            top_k: Number of results.
        
        Returns:
            Search results.
        """
        # Generate hypothetical document
        hypothetical = self._generate_hypothetical(query)
        logger.debug(f"HyDE hypothetical: {hypothetical[:100]}...")
        
        # Search using the hypothetical as query
        return self.searcher.search(
            query=hypothetical,
            docs_df=self.docs_df,
            vectors=self.vectors,
            top_k=top_k
        )


class AgenticSearch:
    """
    Agentic search that iteratively refines queries.
    
    The LLM decides:
    1. What to search for
    2. Whether results are sufficient
    3. How to refine the query
    """
    
    def __init__(
        self,
        searcher,
        docs_df,
        vectors,
        llm_provider: str = "openai",
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        max_iterations: int = 3
    ):
        self.searcher = searcher
        self.docs_df = docs_df
        self.vectors = vectors
        self.llm_provider = llm_provider
        self.model = model
        self.api_key = api_key
        self.max_iterations = max_iterations
        self.rag = RAGPipeline(
            searcher, docs_df, vectors, 
            llm_provider, model, api_key
        )
    
    def _should_continue(self, query: str, results: List, iteration: int) -> Tuple[bool, str]:
        """Ask LLM if we should continue searching."""
        if iteration >= self.max_iterations:
            return False, query
        
        context = "\n".join([f"- {content[:100]}..." for _, content, _ in results[:3]])
        
        prompt = f"""Given this search query and results, should we search again with a different query?

Query: {query}
Results preview:
{context}

If the results seem relevant, respond with: DONE
If we should search again, respond with: SEARCH: <new query>

Response:"""
        
        # Call LLM (simplified)
        if self.llm_provider == "openai":
            from openai import OpenAI
            api_key = self.api_key or os.getenv("OPENAI_API_KEY")
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100
            )
            answer = response.choices[0].message.content.strip()
            
            if answer.startswith("DONE"):
                return False, query
            elif answer.startswith("SEARCH:"):
                new_query = answer.replace("SEARCH:", "").strip()
                return True, new_query
        
        return False, query
    
    def search(self, query: str, top_k: int = 5) -> RAGResponse:
        """
        Perform agentic search with query refinement.
        
        Args:
            query: Initial query.
            top_k: Results per iteration.
        
        Returns:
            RAGResponse with final answer.
        """
        current_query = query
        all_results = []
        
        for i in range(self.max_iterations):
            logger.info(f"Agentic search iteration {i+1}: '{current_query}'")
            
            results = self.searcher.search(
                query=current_query,
                docs_df=self.docs_df,
                vectors=self.vectors,
                top_k=top_k
            )
            
            all_results.extend(results)
            
            should_continue, new_query = self._should_continue(current_query, results, i)
            
            if not should_continue:
                break
            
            current_query = new_query
        
        # Deduplicate results by doc_id
        seen = set()
        unique_results = []
        for r in all_results:
            if r[2] not in seen:
                seen.add(r[2])
                unique_results.append(r)
        
        # Generate final answer
        return self.rag.ask(query, top_k=min(top_k, len(unique_results)))
