import threading
import time
from typing import Optional
from uuid import UUID
from backend.app.domain import models
from backend.app.domain.ports import MetadataStore
from backend.app.services.indexing import IndexingService

class JobRunner:
    def __init__(self, metadata_store: MetadataStore, indexing_service: IndexingService):
        self.metadata = metadata_store
        self.indexing = indexing_service
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Starts the worker loop in a background thread."""
        if self._thread and self._thread.is_alive():
            return
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()
        print("JobRunner started.")

    def stop(self):
        """Stops the worker loop gracefully."""
        if not self._thread:
            return
            
        print("JobRunner stopping...")
        self._stop_event.set()
        self._thread.join()
        print("JobRunner stopped.")

    def enqueue_job(self, type: models.JobType, payload: dict) -> models.Job:
        """Enqueues a new job."""
        job = models.Job(type=type, payload=payload, status=models.JobStatus.PENDING)
        return self.metadata.upsert_job(job)

    def _worker_loop(self):
        """Main worker loop that polls for pending jobs."""
        while not self._stop_event.is_set():
            try:
                # 1. Poll for pending jobs
                # We limit to 1 because we process sequentially in this thread
                pending_jobs = self.metadata.get_pending_jobs(limit=1)
                
                if pending_jobs:
                    job = pending_jobs[0]
                    self._process_job(job)
                else:
                    # No jobs, sleep for a bit
                    time.sleep(1)
            except Exception as e:
                print(f"JobRunner worker error: {e}")
                time.sleep(5) # Backoff on error

    def _process_job(self, job: models.Job):
        print(f"Processing job {job.id} ({job.type})")
        
        # Note: We rely on the service methods to update job status to RUNNING/DONE/FAILED.
        # However, for robustness, we might want to mark it RUNNING here if the service doesn't immediately.
        # But IndexingService.scan_source updates it.
        
        try:
            if job.type == models.JobType.SCAN_SOURCE:
                source_id_str = job.payload.get("source_id")
                if not source_id_str:
                    raise ValueError("Missing source_id in job payload")
                
                self.indexing.scan_source(UUID(source_id_str), job)
                
            elif job.type == models.JobType.INDEX_DOC:
                # TODO: Implement single doc indexing job logic if needed
                # For now, scan_source handles indexing too
                pass
                
            elif job.type == models.JobType.REINDEX_ALL:
                # TODO: Implement reindex all
                # self.indexing.reindex_all() 
                # reindex_all in service doesn't take job yet.
                pass
                
            else:
                raise ValueError(f"Unknown job type: {job.type}")
                
        except Exception as e:
            print(f"Job {job.id} failed: {e}")
            job.status = models.JobStatus.FAILED
            job.error = str(e)
            self.metadata.upsert_job(job)
