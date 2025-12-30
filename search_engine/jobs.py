"""
Async job queue for background indexing operations.

Supports job tracking, retries, and webhook notifications.
"""
import json
import threading
import time
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from queue import Queue, Empty
from typing import Any, Callable, Dict, List, Optional
import requests
from loguru import logger


class JobState(Enum):
    """Job execution state."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobProgress:
    """Job progress information."""
    total: int
    processed: int = 0
    
    @property
    def percentage(self) -> float:
        return (self.processed / self.total * 100) if self.total > 0 else 0


@dataclass
class Job:
    """Job representation."""
    id: str
    status: JobState
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: JobProgress = field(default_factory=lambda: JobProgress(total=0))
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    retry_count: int = 0
    webhook_url: Optional[str] = None


@dataclass
class IndexingTask:
    """Task for indexing documents."""
    documents: List[Dict[str, Any]]  # List of {content, source_path, metadata}
    options: Dict[str, Any] = field(default_factory=dict)


class JobBackend(ABC):
    """Abstract job storage backend."""
    
    @abstractmethod
    def save(self, job: Job) -> None:
        pass
    
    @abstractmethod
    def get(self, job_id: str) -> Optional[Job]:
        pass
    
    @abstractmethod
    def list_jobs(self, status: Optional[JobState] = None) -> List[Job]:
        pass
    
    @abstractmethod
    def delete(self, job_id: str) -> bool:
        pass


class InMemoryJobBackend(JobBackend):
    """In-memory job storage."""
    
    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
    
    def save(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job
    
    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)
    
    def list_jobs(self, status: Optional[JobState] = None) -> List[Job]:
        jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)
    
    def delete(self, job_id: str) -> bool:
        with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]
                return True
            return False


class JobQueue:
    """
    Background job queue for async indexing.
    
    Features:
    - Configurable concurrency limit
    - Automatic retries with exponential backoff
    - Progress tracking
    - Webhook notifications
    """
    
    def __init__(
        self,
        backend: Optional[JobBackend] = None,
        max_concurrent: int = 3,
        max_retries: int = 3,
        base_retry_delay: float = 1.0
    ):
        """
        Initialize job queue.
        
        Args:
            backend: Job storage backend
            max_concurrent: Maximum concurrent jobs
            max_retries: Maximum retry attempts
            base_retry_delay: Base delay for exponential backoff (seconds)
        """
        self.backend = backend or InMemoryJobBackend()
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.base_retry_delay = base_retry_delay
        
        # Task queue and executor
        self._queue: Queue[str] = Queue()
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self._active_jobs: Dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        
        # Task handler (set by user)
        self._task_handler: Optional[Callable[[IndexingTask, Callable[[int], None]], Dict[str, Any]]] = None
        
        # Start worker threads
        self._running = True
        self._workers = [
            threading.Thread(target=self._worker, daemon=True)
            for _ in range(max_concurrent)
        ]
        for w in self._workers:
            w.start()
        
        logger.info(f"Initialized JobQueue (max_concurrent={max_concurrent}, max_retries={max_retries})")
    
    def set_task_handler(
        self,
        handler: Callable[[IndexingTask, Callable[[int], None]], Dict[str, Any]]
    ) -> None:
        """
        Set the function that processes indexing tasks.
        
        Args:
            handler: Function(task, progress_callback) -> result_dict
        """
        self._task_handler = handler
    
    def enqueue(
        self,
        task: IndexingTask,
        webhook_url: Optional[str] = None
    ) -> Job:
        """
        Enqueue indexing task.
        
        Args:
            task: Indexing task to process
            webhook_url: URL to notify on completion
            
        Returns:
            Job object with ID for tracking
        """
        job_id = str(uuid.uuid4())
        
        job = Job(
            id=job_id,
            status=JobState.PENDING,
            created_at=datetime.now(),
            progress=JobProgress(total=len(task.documents)),
            webhook_url=webhook_url
        )
        
        # Store job and task
        self.backend.save(job)
        self._store_task(job_id, task)
        
        # Add to queue
        self._queue.put(job_id)
        
        logger.info(f"Enqueued job {job_id} with {len(task.documents)} documents")
        return job
    
    def get_status(self, job_id: str) -> Optional[Job]:
        """Get job status."""
        return self.backend.get(job_id)
    
    def cancel(self, job_id: str) -> bool:
        """
        Cancel pending job.
        
        Args:
            job_id: Job ID to cancel
            
        Returns:
            True if cancelled, False if not cancellable
        """
        job = self.backend.get(job_id)
        if not job:
            return False
        
        if job.status != JobState.PENDING:
            logger.warning(f"Cannot cancel job {job_id} in state {job.status}")
            return False
        
        # Mark as cancelled
        job.status = JobState.CANCELLED
        job.completed_at = datetime.now()
        self.backend.save(job)
        
        # Signal cancellation if processing
        if job_id in self._active_jobs:
            self._active_jobs[job_id].set()
        
        logger.info(f"Cancelled job {job_id}")
        return True
    
    def _worker(self) -> None:
        """Worker thread that processes jobs."""
        while self._running:
            try:
                job_id = self._queue.get(timeout=1.0)
            except Empty:
                continue
            
            job = self.backend.get(job_id)
            if not job or job.status == JobState.CANCELLED:
                continue
            
            self._process_job(job)
    
    def _process_job(self, job: Job) -> None:
        """Process a single job with retries."""
        task = self._get_task(job.id)
        if not task:
            job.status = JobState.FAILED
            job.error = "Task data not found"
            self.backend.save(job)
            return
        
        # Create cancellation event
        cancel_event = threading.Event()
        with self._lock:
            self._active_jobs[job.id] = cancel_event
        
        try:
            # Update status
            job.status = JobState.PROCESSING
            job.started_at = datetime.now()
            self.backend.save(job)
            
            # Progress callback
            def update_progress(processed: int):
                job.progress.processed = processed
                self.backend.save(job)
            
            # Execute task
            if self._task_handler:
                result = self._task_handler(task, update_progress)
                job.result = result
            else:
                # Default: just mark progress
                for i in range(len(task.documents)):
                    if cancel_event.is_set():
                        raise Exception("Job cancelled")
                    time.sleep(0.01)  # Simulate work
                    update_progress(i + 1)
                job.result = {"documents_indexed": len(task.documents)}
            
            # Success
            job.status = JobState.COMPLETED
            job.completed_at = datetime.now()
            job.progress.processed = job.progress.total
            self.backend.save(job)
            
            logger.info(f"Job {job.id} completed successfully")
            
        except Exception as e:
            logger.error(f"Job {job.id} failed: {e}")
            
            # Retry logic
            if job.retry_count < self.max_retries:
                job.retry_count += 1
                delay = self.base_retry_delay * (2 ** (job.retry_count - 1))
                
                logger.info(f"Retrying job {job.id} in {delay}s (attempt {job.retry_count})")
                
                job.status = JobState.PENDING
                self.backend.save(job)
                
                # Re-queue after delay
                threading.Timer(delay, lambda: self._queue.put(job.id)).start()
            else:
                job.status = JobState.FAILED
                job.error = str(e)
                job.completed_at = datetime.now()
                self.backend.save(job)
        
        finally:
            with self._lock:
                self._active_jobs.pop(job.id, None)
            
            # Send webhook notification
            if job.webhook_url and job.status in (JobState.COMPLETED, JobState.FAILED):
                self._send_webhook(job)
    
    def _send_webhook(self, job: Job) -> None:
        """Send webhook notification."""
        try:
            payload = {
                "job_id": job.id,
                "status": job.status.value,
                "progress": {
                    "total": job.progress.total,
                    "processed": job.progress.processed
                },
                "result": job.result,
                "error": job.error,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None
            }
            
            logger.debug(f"Sending webhook payload: {json.dumps(payload)}")
            
            response = requests.post(
                job.webhook_url,
                json=payload,
                timeout=10
            )
            
            if response.ok:
                logger.info(f"Webhook sent for job {job.id}")
            else:
                logger.warning(f"Webhook failed for job {job.id}: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Webhook error for job {job.id}: {e}")
    
    # Task storage (in-memory for simplicity)
    _tasks: Dict[str, IndexingTask] = {}
    _tasks_lock = threading.Lock()
    
    def _store_task(self, job_id: str, task: IndexingTask) -> None:
        with self._tasks_lock:
            self._tasks[job_id] = task
    
    def _get_task(self, job_id: str) -> Optional[IndexingTask]:
        return self._tasks.get(job_id)
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the job queue."""
        self._running = False
        if wait:
            for w in self._workers:
                w.join(timeout=5.0)
        self._executor.shutdown(wait=wait)
        logger.info("JobQueue shutdown complete")
    
    @property
    def pending_count(self) -> int:
        """Number of pending jobs."""
        return len(self.backend.list_jobs(JobState.PENDING))
    
    @property
    def active_count(self) -> int:
        """Number of active jobs."""
        return len(self._active_jobs)
