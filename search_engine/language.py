"""
Multi-language support for the search engine.

Provides language detection and multilingual embeddings.
"""
from dataclasses import dataclass
from typing import List, Optional
from loguru import logger

try:
    from langdetect import detect, detect_langs, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logger.warning("langdetect not available. Language detection disabled.")

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


@dataclass
class LanguageResult:
    """Result of language detection."""
    language: str  # ISO 639-1 code (e.g., "en", "fr", "de")
    confidence: float
    script: Optional[str] = None  # e.g., "Latin", "Cyrillic"


# Supported languages for multilingual model
SUPPORTED_LANGUAGES = [
    "en", "de", "fr", "es", "it", "pt", "nl", "pl", "ru", "ja",
    "zh", "ko", "ar", "tr", "vi", "th", "id", "hi", "bn", "ta",
    "te", "mr", "gu", "kn", "ml", "pa", "ur", "fa", "he", "el",
    "cs", "sk", "hu", "ro", "bg", "uk", "hr", "sr", "sl", "lt",
    "lv", "et", "fi", "sv", "da", "no", "is", "ga", "cy", "mt"
]


class LanguageDetector:
    """
    Detect document language using langdetect.
    
    Supports 55 languages with good accuracy.
    """
    
    def __init__(self, min_confidence: float = 0.5):
        """
        Initialize language detector.
        
        Args:
            min_confidence: Minimum confidence threshold for detection
        """
        if not LANGDETECT_AVAILABLE:
            raise ImportError(
                "langdetect required for language detection. "
                "Install with: pip install langdetect"
            )
        
        self.min_confidence = min_confidence
        logger.info("Initialized LanguageDetector")
    
    def detect(self, text: str) -> LanguageResult:
        """
        Detect language of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            LanguageResult with language code and confidence
        """
        if not text or len(text.strip()) < 10:
            return LanguageResult(language="unknown", confidence=0.0)
        
        try:
            # Clean text for detection
            clean_text = ' '.join(text.split())[:1000]  # Limit length
            
            # Get language probabilities
            langs = detect_langs(clean_text)
            
            if not langs:
                return LanguageResult(language="unknown", confidence=0.0)
            
            # Get top result
            top = langs[0]
            lang = top.lang
            score = top.prob
            
            if score < self.min_confidence:
                return LanguageResult(language="unknown", confidence=score)
            
            return LanguageResult(
                language=lang,
                confidence=score,
                script=self._detect_script(text)
            )
        except LangDetectException as e:
            logger.warning(f"Language detection failed: {e}")
            return LanguageResult(language="unknown", confidence=0.0)
        except Exception as e:
            logger.warning(f"Language detection error: {e}")
            return LanguageResult(language="unknown", confidence=0.0)
    
    def detect_simple(self, text: str) -> str:
        """
        Quick language detection returning just the language code.
        
        Args:
            text: Text to analyze
            
        Returns:
            ISO 639-1 language code (e.g., "en", "fr") or "unknown"
        """
        if not text or len(text.strip()) < 10:
            return "unknown"
        
        try:
            clean_text = ' '.join(text.split())[:1000]
            return detect(clean_text)
        except LangDetectException:
            return "unknown"
    
    def detect_batch(self, texts: List[str]) -> List[LanguageResult]:
        """
        Detect language for multiple texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of LanguageResult objects
        """
        return [self.detect(text) for text in texts]
    
    def _detect_script(self, text: str) -> Optional[str]:
        """Detect writing script from text sample."""
        if not text:
            return None
        
        # Sample characters
        sample = text[:500]
        
        # Count character types
        latin = sum(1 for c in sample if '\u0041' <= c <= '\u007A' or '\u00C0' <= c <= '\u024F')
        cyrillic = sum(1 for c in sample if '\u0400' <= c <= '\u04FF')
        arabic = sum(1 for c in sample if '\u0600' <= c <= '\u06FF')
        cjk = sum(1 for c in sample if '\u4E00' <= c <= '\u9FFF')
        hangul = sum(1 for c in sample if '\uAC00' <= c <= '\uD7AF')
        devanagari = sum(1 for c in sample if '\u0900' <= c <= '\u097F')
        
        scripts = {
            'Latin': latin,
            'Cyrillic': cyrillic,
            'Arabic': arabic,
            'CJK': cjk,
            'Hangul': hangul,
            'Devanagari': devanagari
        }
        
        max_script = max(scripts, key=scripts.get)
        if scripts[max_script] > 0:
            return max_script
        return None


class MultilingualEmbedder:
    """
    Generate embeddings for text in any supported language.
    
    Uses multilingual sentence-transformers model.
    """
    
    DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize multilingual embedder.
        
        Args:
            model_name: Sentence-transformers model name
        """
        if not EMBEDDINGS_AVAILABLE:
            raise ImportError("sentence-transformers required for embeddings")
        
        self.model_name = model_name or self.DEFAULT_MODEL
        self.model = SentenceTransformer(self.model_name)
        
        logger.info(f"Initialized MultilingualEmbedder with model '{self.model_name}'")
    
    def encode(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: List of texts in any supported language
            show_progress: Show progress bar
            
        Returns:
            NumPy array of embeddings
        """
        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return np.array(embeddings, dtype=np.float32)
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode single text."""
        return self.encode([text])[0]
    
    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
    
    @property
    def supported_languages(self) -> List[str]:
        """List of supported language codes."""
        return SUPPORTED_LANGUAGES.copy()


def get_language_name(code: str) -> str:
    """Get full language name from ISO 639-1 code."""
    names = {
        "en": "English", "de": "German", "fr": "French", "es": "Spanish",
        "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "pl": "Polish",
        "ru": "Russian", "ja": "Japanese", "zh": "Chinese", "ko": "Korean",
        "ar": "Arabic", "tr": "Turkish", "vi": "Vietnamese", "th": "Thai",
        "id": "Indonesian", "hi": "Hindi", "bn": "Bengali", "ta": "Tamil",
        "unknown": "Unknown"
    }
    return names.get(code, code.upper())
