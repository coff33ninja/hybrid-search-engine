import polars as pl
import numpy as np
import duckdb
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Tuple
from pathlib import Path
from loguru import logger

from .extractor import discover_documents, preprocess_text

# Optional FAISS import
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Using brute-force search.")


class FAISSIndex:
    """Wrapper for FAISS index operations."""
    
    def __init__(self, dimension: int, use_gpu: bool = False):
        """
        Initialize FAISS index.
        
        Args:
            dimension: Vector dimension.
            use_gpu: Whether to use GPU (requires faiss-gpu).
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is not installed. Install with: pip install faiss-cpu")
        
        self.dimension = dimension
        # Use IVF index for larger datasets, flat for small
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine after normalization)
        self.is_trained = True
        logger.info(f"Created FAISS index with dimension {dimension}")
    
    def add(self, vectors: np.ndarray):
        """Add vectors to the index."""
        # Normalize for cosine similarity
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        logger.info(f"Added {len(vectors)} vectors to FAISS index")
    
    def search(self, query: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors.
        
        Returns:
            Tuple of (distances, indices).
        """
        query = query.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)
        distances, indices = self.index.search(query, top_k)
        return distances[0], indices[0]
    
    def save(self, path: str):
        """Save index to disk."""
        faiss.write_index(self.index, path)
        logger.info(f"Saved FAISS index to {path}")
    
    def load(self, path: str):
        """Load index from disk."""
        self.index = faiss.read_index(path)
        logger.info(f"Loaded FAISS index from {path}")


class Indexer:
    """
    Handles document indexing, vectorization, and database storage.
    """
    def __init__(
        self, 
        db_path: str = "index.duckdb", 
        model_name: str = "all-MiniLM-L6-v2",
        use_faiss: bool = False,
        faiss_index_path: Optional[str] = None
    ):
        """
        Initializes the Indexer.

        Args:
            db_path: Path to the DuckDB database file.
            model_name: The name of the sentence-transformer model to use.
            use_faiss: Whether to use FAISS for ANN search.
            faiss_index_path: Path to save/load FAISS index.
        """
        self.db_path = db_path
        self.model = SentenceTransformer(model_name)
        self.con = None
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.faiss_index_path = faiss_index_path or "index.faiss"
        self.faiss_index: Optional[FAISSIndex] = None
        logger.info(f"Initialized Indexer with DB path '{db_path}' and model '{model_name}'.")
        if self.use_faiss:
            logger.info("FAISS enabled for approximate nearest neighbor search.")

    def __enter__(self):
        """Opens the database connection."""
        self.con = duckdb.connect(self.db_path)
        self.setup_database()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Closes the database connection."""
        if self.con:
            self.con.close()
            logger.info("Database connection closed.")

    def setup_database(self):
        """Creates the necessary tables in the database if they don't exist."""
        if not self.con:
            raise ConnectionError("Database connection is not open.")
        
        # Main documents table with extended schema for high-impact features
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS docs (
                doc_id INTEGER PRIMARY KEY,
                content TEXT,
                source_path TEXT,
                char_count INTEGER,
                word_count INTEGER,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                language VARCHAR(10),
                fingerprint BLOB,
                is_duplicate BOOLEAN DEFAULT FALSE,
                canonical_doc_id INTEGER,
                metadata JSON
            )
        """)
        
        # Add columns if they don't exist (for existing databases)
        try:
            self.con.execute("ALTER TABLE docs ADD COLUMN IF NOT EXISTS language VARCHAR(10)")
            self.con.execute("ALTER TABLE docs ADD COLUMN IF NOT EXISTS fingerprint BLOB")
            self.con.execute("ALTER TABLE docs ADD COLUMN IF NOT EXISTS is_duplicate BOOLEAN DEFAULT FALSE")
            self.con.execute("ALTER TABLE docs ADD COLUMN IF NOT EXISTS canonical_doc_id INTEGER")
            self.con.execute("ALTER TABLE docs ADD COLUMN IF NOT EXISTS metadata JSON")
        except Exception:
            pass  # Columns already exist or ALTER not supported
        
        # Query history for learning
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS query_history (
                query_id INTEGER PRIMARY KEY,
                query_text TEXT,
                semantic_weight FLOAT,
                lexical_weight FLOAT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Query suggestions for autocomplete
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS query_suggestions (
                query_text TEXT PRIMARY KEY,
                frequency INTEGER DEFAULT 1,
                last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                avg_result_count FLOAT
            )
        """)
        
        # Feedback for ranking improvement
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id INTEGER PRIMARY KEY,
                query_id INTEGER,
                doc_id INTEGER,
                relevance_score INTEGER,
                clicked BOOLEAN DEFAULT FALSE,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Indexing jobs for async processing
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS indexing_jobs (
                job_id VARCHAR(36) PRIMARY KEY,
                status VARCHAR(20),
                total_docs INTEGER,
                processed_docs INTEGER,
                created_at TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                error_message TEXT,
                webhook_url TEXT,
                retry_count INTEGER DEFAULT 0
            )
        """)
        
        # Create sequence for query_id if not exists
        self.con.execute("""
            CREATE SEQUENCE IF NOT EXISTS query_id_seq START 1
        """)
        self.con.execute("""
            CREATE SEQUENCE IF NOT EXISTS feedback_id_seq START 1
        """)
        
        logger.info("Database setup complete. Tables ready.")

    def build_table(self, docs: List[str], source_paths: Optional[List[str]] = None) -> pl.DataFrame:
        """
        Creates a Polars DataFrame from a list of documents.

        Args:
            docs: A list of document strings.
            source_paths: Optional list of source file paths.

        Returns:
            A Polars DataFrame with document info.
        """
        logger.info("Building DataFrame from documents...")
        
        if source_paths is None:
            source_paths = ["" for _ in docs]
        
        df = pl.DataFrame({
            "doc_id": range(len(docs)),
            "content": docs,
            "source_path": source_paths,
            "char_count": [len(d) for d in docs],
            "word_count": [len(d.split()) for d in docs],
        })
        logger.info(f"Created DataFrame with {len(df)} rows.")
        return df

    def embed(self, docs: List[str]) -> np.ndarray:
        """
        Embeds a list of documents into vector representations.

        Args:
            docs: A list of document strings.

        Returns:
            A NumPy array of document embeddings.
        """
        logger.info(f"Embedding {len(docs)} documents...")
        vectors = self.model.encode(docs, show_progress_bar=True, convert_to_numpy=True)
        vectors = np.array(vectors, dtype=np.float32)
        logger.info(f"Created {vectors.shape} embedding matrix.")
        return vectors

    def index_documents(
        self, 
        docs: List[str], 
        source_paths: Optional[List[str]] = None
    ) -> Tuple[pl.DataFrame, np.ndarray]:
        """
        Indexes a list of documents into the database and returns their embeddings.

        Args:
            docs: A list of document strings.
            source_paths: Optional list of source file paths.

        Returns:
            A tuple containing the Polars DataFrame and embeddings.
        """
        if not self.con:
            raise ConnectionError("Database connection is not open.")

        # Preprocess documents
        docs = [preprocess_text(d) for d in docs]
        df = self.build_table(docs, source_paths)

        logger.info("Clearing existing index and inserting new documents...")
        self.con.execute("DELETE FROM docs")
        self.con.register('df_view', df)
        self.con.execute("""
            INSERT INTO docs (doc_id, content, source_path, char_count, word_count)
            SELECT doc_id, content, source_path, char_count, word_count FROM df_view
        """)
        self.con.unregister('df_view')
        logger.info(f"Inserted {len(df)} documents into DuckDB.")

        vectors = self.embed(df["content"].to_list())
        
        # Build FAISS index if enabled
        if self.use_faiss:
            self.faiss_index = FAISSIndex(vectors.shape[1])
            self.faiss_index.add(vectors.copy())
            self.faiss_index.save(self.faiss_index_path)

        return df, vectors

    def index_from_directory(
        self, 
        data_dir: str = "data",
        extensions: Optional[List[str]] = None
    ) -> Tuple[pl.DataFrame, np.ndarray]:
        """
        Indexes documents from a directory.

        Args:
            data_dir: Path to the data directory.
            extensions: File extensions to include.

        Returns:
            A tuple containing the Polars DataFrame and embeddings.
        """
        data_path = Path(data_dir)
        logger.info(f"Discovering documents in '{data_dir}'...")
        
        doc_infos = discover_documents(data_path, extensions)
        
        if not doc_infos:
            logger.warning(f"No documents found in '{data_dir}'")
            return pl.DataFrame(), np.array([])
        
        docs = [d['content'] for d in doc_infos]
        paths = [d['path'] for d in doc_infos]
        
        logger.info(f"Found {len(docs)} documents to index.")
        return self.index_documents(docs, paths)

    def add_documents(
        self, 
        docs: List[str], 
        source_paths: Optional[List[str]] = None
    ) -> Tuple[pl.DataFrame, np.ndarray]:
        """
        Adds documents to existing index (incremental indexing).

        Args:
            docs: A list of document strings.
            source_paths: Optional list of source file paths.

        Returns:
            A tuple containing the new documents DataFrame and embeddings.
        """
        if not self.con:
            raise ConnectionError("Database connection is not open.")

        # Get current max doc_id
        result = self.con.execute("SELECT COALESCE(MAX(doc_id), -1) FROM docs").fetchone()
        start_id = result[0] + 1

        docs = [preprocess_text(d) for d in docs]
        if source_paths is None:
            source_paths = ["" for _ in docs]

        df = pl.DataFrame({
            "doc_id": range(start_id, start_id + len(docs)),
            "content": docs,
            "source_path": source_paths,
            "char_count": [len(d) for d in docs],
            "word_count": [len(d.split()) for d in docs],
        })

        self.con.register('df_view', df)
        self.con.execute("""
            INSERT INTO docs (doc_id, content, source_path, char_count, word_count)
            SELECT doc_id, content, source_path, char_count, word_count FROM df_view
        """)
        self.con.unregister('df_view')
        logger.info(f"Added {len(df)} new documents.")

        vectors = self.embed(df["content"].to_list())
        return df, vectors

    def get_all_documents(self) -> Tuple[pl.DataFrame, List[str]]:
        """
        Retrieves all documents from the database.

        Returns:
            Tuple of (DataFrame, list of contents).
        """
        if not self.con:
            raise ConnectionError("Database connection is not open.")
        
        result = self.con.execute("SELECT * FROM docs ORDER BY doc_id").pl()
        return result, result["content"].to_list() if len(result) > 0 else []
