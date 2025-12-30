import re
from typing import List, Dict, Any
from pathlib import Path

# Common English stopwords
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this',
    'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
}


def extract_tokens(text: str, remove_stopwords: bool = False) -> List[str]:
    """
    Extracts alphanumeric tokens from a string.

    Args:
        text: The input string.
        remove_stopwords: Whether to remove common stopwords.

    Returns:
        A list of lowercased tokens.
    """
    if not text:
        return []
    tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def preprocess_text(text: str, remove_stopwords: bool = False) -> str:
    """
    Preprocesses text by normalizing whitespace and optionally removing stopwords.

    Args:
        text: The input string.
        remove_stopwords: Whether to remove common stopwords.

    Returns:
        Preprocessed text string.
    """
    if not text:
        return ""
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    if remove_stopwords:
        tokens = extract_tokens(text, remove_stopwords=True)
        return ' '.join(tokens)
    return text


def extract_metadata(text: str) -> Dict[str, Any]:
    """
    Extracts metadata from text content.

    Args:
        text: The input string.

    Returns:
        Dictionary with extracted metadata.
    """
    tokens = extract_tokens(text)
    return {
        'char_count': len(text),
        'word_count': len(tokens),
        'unique_words': len(set(tokens)),
        'avg_word_length': sum(len(t) for t in tokens) / len(tokens) if tokens else 0
    }


def load_file_content(file_path: Path) -> str:
    """
    Loads content from a file based on its extension.

    Args:
        file_path: Path to the file.

    Returns:
        File content as string.
    """
    suffix = file_path.suffix.lower()
    
    try:
        if suffix in {'.txt', '.md', '.log', '.csv'}:
            return file_path.read_text(encoding='utf-8')
        elif suffix == '.json':
            import json
            data = json.loads(file_path.read_text(encoding='utf-8'))
            # Flatten JSON to text
            if isinstance(data, dict):
                return ' '.join(str(v) for v in data.values() if isinstance(v, str))
            elif isinstance(data, list):
                return ' '.join(str(item) for item in data if isinstance(item, str))
            return str(data)
        else:
            # Try to read as text
            return file_path.read_text(encoding='utf-8')
    except Exception:
        return ""


def discover_documents(data_dir: Path, extensions: List[str] = None) -> List[Dict[str, Any]]:
    """
    Discovers documents in a directory.

    Args:
        data_dir: Path to the data directory.
        extensions: List of file extensions to include (e.g., ['.txt', '.md']).
                   If None, includes common text formats.

    Returns:
        List of dicts with 'path', 'content', and 'metadata'.
    """
    if extensions is None:
        extensions = ['.txt', '.md', '.log', '.json', '.csv']
    
    documents = []
    if not data_dir.exists():
        return documents
    
    for file_path in data_dir.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            content = load_file_content(file_path)
            if content.strip():
                documents.append({
                    'path': str(file_path),
                    'content': preprocess_text(content),
                    'metadata': extract_metadata(content)
                })
    
    return documents
