"""
Document deduplication using MinHash and LSH.

Detects near-duplicate documents before indexing with configurable actions.
"""
import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple
import numpy as np
from loguru import logger

try:
    from datasketch import MinHash, MinHashLSH
    DATASKETCH_AVAILABLE = True
except ImportError:
    DATASKETCH_AVAILABLE = False
    logger.warning("datasketch not available. Deduplication disabled.")


class DedupeAction(Enum):
    """Action to take when duplicate is detected."""
    SKIP = "skip"      # Don't index duplicate
    MERGE = "merge"    # Combine metadata with canonical
    FLAG = "flag"      # Index but mark as duplicate


@dataclass
class MinHashSignature:
    """MinHash signature for a document."""
    values: np.ndarray  # uint64 array of hash values
    
    def jaccard_similarity(self, other: "MinHashSignature") -> float:
        """Estimate Jaccard similarity using MinHash."""
        if len(self.values) != len(other.values):
            raise ValueError("Signatures must have same length")
        return np.mean(self.values == other.values)


@dataclass
class DuplicateMatch:
    """Result of duplicate detection."""
    doc_id: int
    similarity: float
    is_canonical: bool


class DeduplicationEngine:
    """
    Detect and handle near-duplicate documents using MinHash/LSH.
    
    Uses locality-sensitive hashing for O(1) average lookup time.
    """
    
    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.9,
        action: DedupeAction = DedupeAction.FLAG
    ):
        """
        Initialize deduplication engine.
        
        Args:
            num_perm: Number of permutations for MinHash (higher = more accurate)
            threshold: Similarity threshold for duplicate detection (0-1)
            action: Action to take when duplicate is detected
        """
        if not DATASKETCH_AVAILABLE:
            raise ImportError("datasketch required for deduplication. Install with: pip install datasketch")
        
        self.num_perm = num_perm
        self.threshold = threshold
        self.action = action
        
        # LSH index for fast lookup
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        
        # Store signatures for similarity calculation
        self._signatures: dict[int, MinHash] = {}
        
        logger.info(f"Initialized DeduplicationEngine with threshold={threshold}, action={action.value}")
    
    def compute_fingerprint(self, content: str) -> MinHashSignature:
        """
        Compute MinHash fingerprint for document content.
        
        Args:
            content: Document text content
            
        Returns:
            MinHash signature
        """
        minhash = MinHash(num_perm=self.num_perm)
        
        # Tokenize content into shingles (3-grams)
        shingles = self._get_shingles(content, k=3)
        
        for shingle in shingles:
            minhash.update(shingle.encode('utf-8'))
        
        return MinHashSignature(values=np.array(minhash.hashvalues, dtype=np.uint64))
    
    def _get_shingles(self, text: str, k: int = 3) -> List[str]:
        """Extract k-character shingles from text."""
        text = text.lower().strip()
        if len(text) < k:
            return [text] if text else []
        return [text[i:i+k] for i in range(len(text) - k + 1)]
    
    def compute_content_hash(self, content: str) -> str:
        """
        Compute a deterministic hash of document content.
        
        Useful for exact duplicate detection before MinHash comparison.
        
        Args:
            content: Document text content
            
        Returns:
            SHA-256 hex digest of normalized content
        """
        normalized = content.lower().strip()
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    def find_duplicates(self, fingerprint: MinHashSignature) -> List[DuplicateMatch]:
        """
        Find existing documents similar to this fingerprint.
        
        Args:
            fingerprint: MinHash signature to check
            
        Returns:
            List of duplicate matches with similarity scores
        """
        # Create MinHash object from signature
        minhash = MinHash(num_perm=self.num_perm)
        minhash.hashvalues = fingerprint.values.astype(np.uint64)
        
        # Query LSH index
        candidates = self.lsh.query(minhash)
        
        matches = []
        for doc_id in candidates:
            if doc_id in self._signatures:
                stored_minhash = self._signatures[doc_id]
                similarity = minhash.jaccard(stored_minhash)
                if similarity >= self.threshold:
                    matches.append(DuplicateMatch(
                        doc_id=doc_id,
                        similarity=similarity,
                        is_canonical=True  # First document is canonical
                    ))
        
        # Sort by similarity descending
        matches.sort(key=lambda m: m.similarity, reverse=True)
        return matches
    
    def add_to_index(self, doc_id: int, fingerprint: MinHashSignature) -> None:
        """
        Add fingerprint to LSH index for future lookups.
        
        Args:
            doc_id: Document ID
            fingerprint: MinHash signature
        """
        minhash = MinHash(num_perm=self.num_perm)
        minhash.hashvalues = fingerprint.values.astype(np.uint64)
        
        # Add to LSH index
        self.lsh.insert(doc_id, minhash)
        
        # Store signature for similarity calculation
        self._signatures[doc_id] = minhash
        
        logger.debug(f"Added doc_id={doc_id} to deduplication index")
    
    def remove_from_index(self, doc_id: int) -> bool:
        """
        Remove document from LSH index.
        
        Args:
            doc_id: Document ID to remove
            
        Returns:
            True if removed, False if not found
        """
        if doc_id in self._signatures:
            stored_minhash = self._signatures[doc_id]
            self.lsh.remove(doc_id)
            del self._signatures[doc_id]
            logger.debug(f"Removed doc_id={doc_id} from deduplication index (had {len(stored_minhash.hashvalues)} hash values)")
            return True
        return False
    
    def check_and_handle(
        self,
        doc_id: int,
        content: str,
        existing_metadata: Optional[dict] = None
    ) -> Tuple[bool, Optional[int], Optional[dict]]:
        """
        Check for duplicates and handle according to configured action.
        
        Args:
            doc_id: New document ID
            content: Document content
            existing_metadata: Metadata of new document
            
        Returns:
            Tuple of (should_index, canonical_doc_id, merged_metadata)
            - should_index: Whether to index this document
            - canonical_doc_id: ID of canonical document if duplicate
            - merged_metadata: Merged metadata if action is MERGE
        """
        fingerprint = self.compute_fingerprint(content)
        duplicates = self.find_duplicates(fingerprint)
        
        if not duplicates:
            # No duplicates found, add to index
            self.add_to_index(doc_id, fingerprint)
            return True, None, None
        
        # Found duplicate(s)
        canonical = duplicates[0]
        logger.info(f"Duplicate detected: doc_id={doc_id} matches doc_id={canonical.doc_id} "
                   f"(similarity={canonical.similarity:.3f})")
        
        if self.action == DedupeAction.SKIP:
            return False, canonical.doc_id, None
        
        elif self.action == DedupeAction.FLAG:
            self.add_to_index(doc_id, fingerprint)
            return True, canonical.doc_id, None
        
        elif self.action == DedupeAction.MERGE:
            # Merge metadata (new document's metadata added to canonical)
            merged = existing_metadata.copy() if existing_metadata else {}
            return False, canonical.doc_id, merged
        
        return True, None, None
    
    def clear(self) -> None:
        """Clear the deduplication index."""
        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        self._signatures.clear()
        logger.info("Cleared deduplication index")
    
    @property
    def size(self) -> int:
        """Number of documents in the index."""
        return len(self._signatures)
