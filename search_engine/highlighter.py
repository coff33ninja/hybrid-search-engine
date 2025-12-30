"""
Result highlighting - show which parts matched the query.
"""
import re
from typing import List, Tuple
from dataclasses import dataclass

from .extractor import extract_tokens


@dataclass
class HighlightedResult:
    """Search result with highlighted snippets."""
    content: str
    highlighted: str
    snippets: List[str]
    score: float
    doc_id: int


class Highlighter:
    """Highlights matching terms in search results."""
    
    def __init__(
        self,
        before_tag: str = "<mark>",
        after_tag: str = "</mark>",
        snippet_length: int = 150,
        max_snippets: int = 3
    ):
        """
        Args:
            before_tag: HTML/text tag before highlighted term.
            after_tag: HTML/text tag after highlighted term.
            snippet_length: Characters per snippet.
            max_snippets: Maximum number of snippets to return.
        """
        self.before_tag = before_tag
        self.after_tag = after_tag
        self.snippet_length = snippet_length
        self.max_snippets = max_snippets
    
    def highlight_text(self, text: str, query: str) -> str:
        """
        Highlight query terms in text.
        
        Args:
            text: Document text.
            query: Search query.
        
        Returns:
            Text with highlighted terms.
        """
        query_tokens = set(extract_tokens(query.lower()))
        
        if not query_tokens:
            return text
        
        # Build regex pattern for all query terms
        pattern = r'\b(' + '|'.join(re.escape(t) for t in query_tokens) + r')\b'
        
        def replace_match(match):
            return f"{self.before_tag}{match.group(0)}{self.after_tag}"
        
        highlighted = re.sub(pattern, replace_match, text, flags=re.IGNORECASE)
        return highlighted
    
    def extract_snippets(self, text: str, query: str) -> List[str]:
        """
        Extract relevant snippets containing query terms.
        
        Args:
            text: Document text.
            query: Search query.
        
        Returns:
            List of snippet strings with highlighted terms.
        """
        query_tokens = set(extract_tokens(query.lower()))
        
        if not query_tokens:
            # Return beginning of text if no query terms
            return [text[:self.snippet_length] + "..." if len(text) > self.snippet_length else text]
        
        snippets = []
        text_lower = text.lower()
        used_positions = set()
        
        # Find positions of query terms
        positions = []
        for token in query_tokens:
            for match in re.finditer(r'\b' + re.escape(token) + r'\b', text_lower):
                pos = match.start()
                # Check if this position overlaps with existing snippets
                overlaps = any(
                    abs(pos - used_pos) < self.snippet_length 
                    for used_pos in used_positions
                )
                if not overlaps:
                    positions.append((pos, token))
        
        # Sort by position
        positions.sort(key=lambda x: x[0])
        
        # Extract snippets around each position
        for pos, token in positions[:self.max_snippets]:
            # Calculate snippet boundaries
            half_len = self.snippet_length // 2
            start = max(0, pos - half_len)
            end = min(len(text), pos + half_len)
            
            # Adjust to word boundaries
            if start > 0:
                # Find previous space
                space_pos = text.rfind(' ', max(0, start - 20), start)
                if space_pos > 0:
                    start = space_pos + 1
            
            if end < len(text):
                # Find next space
                space_pos = text.find(' ', end, min(len(text), end + 20))
                if space_pos > 0:
                    end = space_pos
            
            snippet = text[start:end]
            
            # Add ellipsis
            if start > 0:
                snippet = "..." + snippet
            if end < len(text):
                snippet = snippet + "..."
            
            # Highlight the snippet
            snippet = self.highlight_text(snippet, query)
            snippets.append(snippet)
            used_positions.add(pos)
        
        # If no snippets found, return beginning
        if not snippets:
            snippet = text[:self.snippet_length]
            if len(text) > self.snippet_length:
                snippet += "..."
            snippets.append(snippet)
        
        return snippets
    
    def highlight_result(
        self, 
        content: str, 
        query: str, 
        score: float, 
        doc_id: int
    ) -> HighlightedResult:
        """
        Create a highlighted result object.
        
        Args:
            content: Document content.
            query: Search query.
            score: Search score.
            doc_id: Document ID.
        
        Returns:
            HighlightedResult with highlighted text and snippets.
        """
        return HighlightedResult(
            content=content,
            highlighted=self.highlight_text(content, query),
            snippets=self.extract_snippets(content, query),
            score=score,
            doc_id=doc_id
        )
    
    def highlight_results(
        self, 
        results: List[Tuple[float, str, int]], 
        query: str
    ) -> List[HighlightedResult]:
        """
        Highlight multiple search results.
        
        Args:
            results: List of (score, content, doc_id) tuples.
            query: Search query.
        
        Returns:
            List of HighlightedResult objects.
        """
        return [
            self.highlight_result(content, query, score, doc_id)
            for score, content, doc_id in results
        ]


class TerminalHighlighter(Highlighter):
    """Highlighter with ANSI color codes for terminal output."""
    
    def __init__(self, **kwargs):
        super().__init__(
            before_tag="\033[1;33m",  # Bold yellow
            after_tag="\033[0m",      # Reset
            **kwargs
        )


class HTMLHighlighter(Highlighter):
    """Highlighter with HTML tags."""
    
    def __init__(self, css_class: str = "highlight", **kwargs):
        super().__init__(
            before_tag=f'<span class="{css_class}">',
            after_tag="</span>",
            **kwargs
        )


class MarkdownHighlighter(Highlighter):
    """Highlighter with Markdown bold."""
    
    def __init__(self, **kwargs):
        super().__init__(
            before_tag="**",
            after_tag="**",
            **kwargs
        )
