"""
Document chunking strategies for better retrieval.
"""
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Chunk:
    """Represents a document chunk."""
    content: str
    chunk_id: int
    doc_id: Optional[int] = None
    source_path: Optional[str] = None
    start_char: int = 0
    end_char: int = 0
    metadata: Optional[Dict[str, Any]] = None


class BaseChunker:
    """Base class for chunking strategies."""
    
    def chunk(self, text: str, doc_id: int = 0, source_path: str = "") -> List[Chunk]:
        raise NotImplementedError


class SentenceChunker(BaseChunker):
    """Split text into sentence-based chunks."""
    
    def __init__(self, max_sentences: int = 3, overlap_sentences: int = 1):
        """
        Args:
            max_sentences: Maximum sentences per chunk.
            overlap_sentences: Number of overlapping sentences between chunks.
        """
        self.max_sentences = max_sentences
        self.overlap_sentences = overlap_sentences
        # Simple sentence boundary regex
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+')
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = self.sentence_pattern.split(text.strip())
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk(self, text: str, doc_id: int = 0, source_path: str = "") -> List[Chunk]:
        sentences = self._split_sentences(text)
        chunks = []
        chunk_id = 0
        
        i = 0
        while i < len(sentences):
            # Get chunk sentences
            end = min(i + self.max_sentences, len(sentences))
            chunk_sentences = sentences[i:end]
            content = ' '.join(chunk_sentences)
            
            chunks.append(Chunk(
                content=content,
                chunk_id=chunk_id,
                doc_id=doc_id,
                source_path=source_path,
                metadata={'sentences': len(chunk_sentences), 'start_sentence': i}
            ))
            
            chunk_id += 1
            i += self.max_sentences - self.overlap_sentences
            if i <= 0:
                i = self.max_sentences
        
        return chunks


class ParagraphChunker(BaseChunker):
    """Split text into paragraph-based chunks."""
    
    def __init__(self, min_length: int = 50, max_length: int = 1000):
        """
        Args:
            min_length: Minimum chunk length (merge small paragraphs).
            max_length: Maximum chunk length (split large paragraphs).
        """
        self.min_length = min_length
        self.max_length = max_length
    
    def chunk(self, text: str, doc_id: int = 0, source_path: str = "") -> List[Chunk]:
        # Split on double newlines or multiple newlines
        paragraphs = re.split(r'\n\s*\n', text.strip())
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        chunk_id = 0
        current_chunk = ""
        start_char = 0
        
        for para in paragraphs:
            # If paragraph is too long, split it
            if len(para) > self.max_length:
                # Flush current chunk first
                if current_chunk:
                    chunks.append(Chunk(
                        content=current_chunk,
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        source_path=source_path,
                        start_char=start_char
                    ))
                    chunk_id += 1
                    current_chunk = ""
                
                # Split long paragraph by sentences
                sentence_chunker = SentenceChunker(max_sentences=5, overlap_sentences=1)
                sub_chunks = sentence_chunker.chunk(para, doc_id, source_path)
                for sc in sub_chunks:
                    sc.chunk_id = chunk_id
                    chunks.append(sc)
                    chunk_id += 1
            else:
                # Try to merge small paragraphs
                if len(current_chunk) + len(para) + 1 <= self.max_length:
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para
                        start_char = text.find(para)
                else:
                    # Flush current chunk
                    if current_chunk:
                        chunks.append(Chunk(
                            content=current_chunk,
                            chunk_id=chunk_id,
                            doc_id=doc_id,
                            source_path=source_path,
                            start_char=start_char
                        ))
                        chunk_id += 1
                    current_chunk = para
                    start_char = text.find(para)
        
        # Don't forget the last chunk
        if current_chunk and len(current_chunk) >= self.min_length:
            chunks.append(Chunk(
                content=current_chunk,
                chunk_id=chunk_id,
                doc_id=doc_id,
                source_path=source_path,
                start_char=start_char
            ))
        elif current_chunk and chunks:
            # Merge with previous if too small
            chunks[-1].content += "\n\n" + current_chunk
        elif current_chunk:
            chunks.append(Chunk(
                content=current_chunk,
                chunk_id=chunk_id,
                doc_id=doc_id,
                source_path=source_path,
                start_char=start_char
            ))
        
        return chunks


class SlidingWindowChunker(BaseChunker):
    """Fixed-size sliding window chunking."""
    
    def __init__(self, window_size: int = 512, overlap: int = 128):
        """
        Args:
            window_size: Number of characters per chunk.
            overlap: Number of overlapping characters.
        """
        self.window_size = window_size
        self.overlap = overlap
    
    def chunk(self, text: str, doc_id: int = 0, source_path: str = "") -> List[Chunk]:
        chunks = []
        chunk_id = 0
        step = self.window_size - self.overlap
        
        for i in range(0, len(text), step):
            content = text[i:i + self.window_size].strip()
            if content:
                chunks.append(Chunk(
                    content=content,
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    source_path=source_path,
                    start_char=i,
                    end_char=min(i + self.window_size, len(text))
                ))
                chunk_id += 1
            
            if i + self.window_size >= len(text):
                break
        
        return chunks


class SemanticChunker(BaseChunker):
    """
    Semantic chunking based on embedding similarity.
    Splits where semantic meaning changes significantly.
    """
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        threshold: float = 0.5,
        min_chunk_size: int = 100
    ):
        """
        Args:
            model_name: Sentence transformer model.
            threshold: Similarity threshold for splitting.
            min_chunk_size: Minimum characters per chunk.
        """
        self.model_name = model_name
        self.threshold = threshold
        self.min_chunk_size = min_chunk_size
        self._model = None
    
    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def chunk(self, text: str, doc_id: int = 0, source_path: str = "") -> List[Chunk]:
        import numpy as np
        
        # First split into sentences
        sentence_pattern = re.compile(r'(?<=[.!?])\s+')
        sentences = sentence_pattern.split(text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 1:
            return [Chunk(content=text, chunk_id=0, doc_id=doc_id, source_path=source_path)]
        
        # Embed all sentences
        embeddings = self.model.encode(sentences, convert_to_numpy=True)
        
        # Find split points based on similarity drops
        chunks = []
        current_sentences = [sentences[0]]
        chunk_id = 0
        
        for i in range(1, len(sentences)):
            # Cosine similarity between consecutive sentences
            sim = np.dot(embeddings[i-1], embeddings[i]) / (
                np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i])
            )
            
            current_content = ' '.join(current_sentences)
            
            # Split if similarity is low and chunk is big enough
            if sim < self.threshold and len(current_content) >= self.min_chunk_size:
                chunks.append(Chunk(
                    content=current_content,
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    source_path=source_path,
                    metadata={'similarity_at_split': float(sim)}
                ))
                chunk_id += 1
                current_sentences = [sentences[i]]
            else:
                current_sentences.append(sentences[i])
        
        # Last chunk
        if current_sentences:
            chunks.append(Chunk(
                content=' '.join(current_sentences),
                chunk_id=chunk_id,
                doc_id=doc_id,
                source_path=source_path
            ))
        
        return chunks


def get_chunker(strategy: str = "paragraph", **kwargs) -> BaseChunker:
    """
    Factory function to get a chunker by name.
    
    Args:
        strategy: One of 'sentence', 'paragraph', 'sliding', 'semantic'.
        **kwargs: Arguments passed to the chunker.
    
    Returns:
        Chunker instance.
    """
    chunkers = {
        'sentence': SentenceChunker,
        'paragraph': ParagraphChunker,
        'sliding': SlidingWindowChunker,
        'semantic': SemanticChunker,
    }
    
    if strategy not in chunkers:
        raise ValueError(f"Unknown chunking strategy: {strategy}. Choose from {list(chunkers.keys())}")
    
    return chunkers[strategy](**kwargs)
