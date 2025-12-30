"""
File watcher for automatic re-indexing when documents change.
"""
import time
from pathlib import Path
from typing import Callable, Optional, List, Set
from loguru import logger

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    logger.warning("watchdog not installed. File watching disabled.")


class DocumentChangeHandler(FileSystemEventHandler):
    """Handles file system events for document changes."""
    
    def __init__(
        self, 
        callback: Callable[[str, str], None],
        extensions: List[str] = None,
        debounce_seconds: float = 2.0
    ):
        """
        Initialize the handler.
        
        Args:
            callback: Function to call on changes. Receives (event_type, file_path).
            extensions: File extensions to watch (e.g., ['.txt', '.md']).
            debounce_seconds: Minimum time between callbacks for same file.
        """
        super().__init__()
        self.callback = callback
        self.extensions = set(extensions or ['.txt', '.md', '.log', '.json', '.csv'])
        self.debounce_seconds = debounce_seconds
        self._last_events: dict = {}
    
    def _should_process(self, path: str) -> bool:
        """Check if file should be processed based on extension."""
        return Path(path).suffix.lower() in self.extensions
    
    def _is_debounced(self, path: str) -> bool:
        """Check if event is within debounce window."""
        now = time.time()
        last_time = self._last_events.get(path, 0)
        if now - last_time < self.debounce_seconds:
            return True
        self._last_events[path] = now
        return False
    
    def on_created(self, event: FileSystemEvent):
        if event.is_directory:
            return
        if self._should_process(event.src_path) and not self._is_debounced(event.src_path):
            logger.info(f"New file detected: {event.src_path}")
            self.callback('created', event.src_path)
    
    def on_modified(self, event: FileSystemEvent):
        if event.is_directory:
            return
        if self._should_process(event.src_path) and not self._is_debounced(event.src_path):
            logger.info(f"File modified: {event.src_path}")
            self.callback('modified', event.src_path)
    
    def on_deleted(self, event: FileSystemEvent):
        if event.is_directory:
            return
        if self._should_process(event.src_path) and not self._is_debounced(event.src_path):
            logger.info(f"File deleted: {event.src_path}")
            self.callback('deleted', event.src_path)


class FileWatcher:
    """
    Watches a directory for file changes and triggers re-indexing.
    """
    
    def __init__(
        self,
        watch_dir: str = "data",
        extensions: List[str] = None,
        on_change: Optional[Callable[[str, str], None]] = None,
        debounce_seconds: float = 2.0
    ):
        """
        Initialize the file watcher.
        
        Args:
            watch_dir: Directory to watch.
            extensions: File extensions to monitor.
            on_change: Callback function for changes.
            debounce_seconds: Debounce time for rapid changes.
        """
        if not WATCHDOG_AVAILABLE:
            raise ImportError("watchdog is required for file watching. Install with: pip install watchdog")
        
        self.watch_dir = Path(watch_dir)
        self.extensions = extensions or ['.txt', '.md', '.log', '.json', '.csv']
        self.on_change = on_change or self._default_callback
        self.debounce_seconds = debounce_seconds
        self._observer: Optional[Observer] = None
        self._running = False
        self._pending_changes: Set[str] = set()
    
    def _default_callback(self, event_type: str, file_path: str):
        """Default callback that logs changes."""
        logger.info(f"File {event_type}: {file_path}")
        self._pending_changes.add(file_path)
    
    def start(self):
        """Start watching for file changes."""
        if not self.watch_dir.exists():
            self.watch_dir.mkdir(parents=True)
            logger.info(f"Created watch directory: {self.watch_dir}")
        
        handler = DocumentChangeHandler(
            callback=self.on_change,
            extensions=self.extensions,
            debounce_seconds=self.debounce_seconds
        )
        
        self._observer = Observer()
        self._observer.schedule(handler, str(self.watch_dir), recursive=True)
        self._observer.start()
        self._running = True
        logger.info(f"Started watching directory: {self.watch_dir}")
    
    def stop(self):
        """Stop watching for file changes."""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._running = False
            logger.info("File watcher stopped.")
    
    def is_running(self) -> bool:
        """Check if watcher is running."""
        return self._running
    
    def get_pending_changes(self) -> Set[str]:
        """Get files that have changed since last check."""
        changes = self._pending_changes.copy()
        self._pending_changes.clear()
        return changes
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def create_auto_indexer(
    indexer_factory: Callable,
    watch_dir: str = "data",
    extensions: List[str] = None
) -> FileWatcher:
    """
    Create a file watcher that automatically re-indexes on changes.
    
    Args:
        indexer_factory: Function that returns an Indexer instance.
        watch_dir: Directory to watch.
        extensions: File extensions to monitor.
    
    Returns:
        Configured FileWatcher instance.
    """
    def on_change(event_type: str, file_path: str):
        logger.info(f"Auto-reindexing triggered by {event_type}: {file_path}")
        try:
            with indexer_factory() as indexer:
                indexer.index_from_directory(watch_dir, extensions)
            logger.success("Auto-reindex complete.")
        except Exception as e:
            logger.error(f"Auto-reindex failed: {e}")
    
    return FileWatcher(
        watch_dir=watch_dir,
        extensions=extensions,
        on_change=on_change
    )
