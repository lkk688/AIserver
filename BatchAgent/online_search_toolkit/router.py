"""FastAPI router for online_search_toolkit — search + scheduler endpoints.

Mount in your FastAPI app::

    from BatchAgent.online_search_toolkit.router import router as online_search_router
    app.include_router(online_search_router, prefix="/api/v1")
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel

from .bootstrap import create_online_search_service
from .config import SearchConfig
from .models import SearchRequest, SearchRecord
from .scheduler import DEFAULT_SEED_URLS, OnlineSearchScheduler

router = APIRouter(prefix="/online-search", tags=["online-search"])

# ── Singleton service + scheduler ─────────────────────────────────────────────

_service = None
_scheduler: Optional[OnlineSearchScheduler] = None
_scheduler_running: bool = False
_scheduler_lock = threading.Lock()
_scheduler_config: Dict[str, Any] = {}


def _get_service():
    global _service
    if _service is None:
        _service = create_online_search_service(SearchConfig())
    return _service


# ── Pydantic models ────────────────────────────────────────────────────────────

class SearchReq(BaseModel):
    query: str
    limit: int = 8
    language: str = "mixed"
    category: Optional[str] = None
    no_cache: bool = False


class URLReq(BaseModel):
    url: str
    domain: str = "general"
    category: str = "general"
    force_refresh: bool = False


class SchedulerConfigReq(BaseModel):
    breaking_interval_minutes: Optional[int] = None
    daily_refresh_hour_1: Optional[int] = None
    daily_refresh_hour_2: Optional[int] = None
    seed_crawl_hour: Optional[int] = None
    seed_urls: Optional[List[str]] = None
    news_refresh_queries: Optional[List[str]] = None
    academic_refresh_queries: Optional[List[str]] = None
    web_refresh_queries: Optional[List[str]] = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _record_to_dict(r: SearchRecord) -> Dict[str, Any]:
    d = r.model_dump(mode="json")
    if d.get("embedding"):
        d["embedding"] = None  # strip large vector from API response
    return d


def _result_response(result) -> Dict[str, Any]:
    # fetched_ids are IDs that came from a live online fetch (not cache)
    fetched_ids: set = set(result.metadata.get("fetched_ids", []))
    items = []
    for r in result.items:
        d = _record_to_dict(r)
        d["from_cache"] = r.id not in fetched_ids if fetched_ids else result.metadata.get("all_from_cache", False)
        items.append(d)
    return {
        "query": result.query,
        "domain": result.domain,
        "count": result.count,
        "items": items,
        "metadata": result.metadata,
    }


# ── Search endpoints ───────────────────────────────────────────────────────────

@router.post("/web")
def search_web(req: SearchReq):
    svc = _get_service()
    result = svc.search(SearchRequest(
        query=req.query, domain="general", limit=req.limit,
        language=req.language, category=req.category, use_cache=not req.no_cache,
    ))
    return _result_response(result)


@router.post("/news")
def search_news(req: SearchReq):
    svc = _get_service()
    result = svc.search(SearchRequest(
        query=req.query, domain="news", limit=req.limit,
        language=req.language, category=req.category, use_cache=not req.no_cache,
    ))
    return _result_response(result)


@router.post("/academic")
def search_academic(req: SearchReq):
    svc = _get_service()
    result = svc.search(SearchRequest(
        query=req.query, domain="academic", limit=req.limit,
        language="en", category=req.category, use_cache=not req.no_cache,
    ))
    return _result_response(result)


@router.post("/medical")
def search_medical(req: SearchReq):
    svc = _get_service()
    result = svc.search(SearchRequest(
        query=req.query, domain="medical", limit=req.limit,
        language="en", category=req.category, use_cache=not req.no_cache,
    ))
    return _result_response(result)


@router.post("/medical_academic")
def search_medical_academic(req: SearchReq):
    """Search PubMed + Europe PMC for research papers (no consumer health pages)."""
    svc = _get_service()
    result = svc.search(SearchRequest(
        query=req.query, domain="medical_academic", limit=req.limit,
        language="en", category=req.category, use_cache=not req.no_cache,
    ))
    return _result_response(result)


@router.post("/url")
def read_url(req: URLReq):
    svc = _get_service()
    record = svc.read_url(
        req.url,
        domain=req.domain,
        category=req.category,
        persist=True,
        force_refresh=req.force_refresh,
    )
    return _record_to_dict(record)


@router.get("/cache")
def get_cache(
    domain: Optional[str] = Query(None, description="Filter by domain: news|academic|medical|general"),
    category: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    recent_hours: Optional[int] = Query(None),
):
    svc = _get_service()
    records = svc.file_store.get_recent_records(
        limit=limit,
        domain=domain,
        category=category,
        recent_hours=recent_hours,
    )
    return {"count": len(records), "items": [_record_to_dict(r) for r in records]}


@router.delete("/cache/{record_id}")
def delete_cache_record(record_id: str):
    """Delete a single cached record by ID."""
    svc = _get_service()
    deleted = svc.file_store.delete_record(record_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Record '{record_id}' not found")
    return {"deleted": record_id}


# ── Scheduler helpers ──────────────────────────────────────────────────────────

def _scheduler_job_list() -> List[Dict[str, Any]]:
    if _scheduler is None:
        return []
    jobs = []
    for job in _scheduler.scheduler.get_jobs():
        jobs.append({
            "id": job.id,
            "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
            "trigger": str(job.trigger),
        })
    return jobs


def _current_config() -> Dict[str, Any]:
    if _scheduler is None:
        svc_cfg = _get_service().config.scheduler
        return {
            "breaking_interval_minutes": svc_cfg.breaking_interval_minutes,
            "daily_refresh_hour_1": svc_cfg.daily_refresh_hour_1,
            "daily_refresh_hour_2": svc_cfg.daily_refresh_hour_2,
            "seed_crawl_hour": 3,
            "seed_urls": DEFAULT_SEED_URLS,
            "news_refresh_queries": [],
            "academic_refresh_queries": [],
            "web_refresh_queries": [],
        }
    return {
        "breaking_interval_minutes": _scheduler.breaking_interval_minutes,
        "daily_refresh_hour_1": _scheduler.daily_refresh_hour_1,
        "daily_refresh_hour_2": _scheduler.daily_refresh_hour_2,
        "seed_crawl_hour": _scheduler.seed_crawl_hour,
        "seed_urls": _scheduler.seed_urls,
        "news_refresh_queries": _scheduler.news_refresh_queries,
        "academic_refresh_queries": _scheduler.academic_refresh_queries,
        "web_refresh_queries": _scheduler.web_refresh_queries,
    }


# ── Scheduler endpoints ────────────────────────────────────────────────────────

@router.get("/scheduler/status")
def scheduler_status():
    return {
        "running": _scheduler_running,
        "jobs": _scheduler_job_list(),
        "config": _current_config(),
    }


@router.post("/scheduler/start")
def scheduler_start(cfg: Optional[SchedulerConfigReq] = None):
    global _scheduler, _scheduler_running
    with _scheduler_lock:
        if _scheduler_running:
            return {"status": "already_running"}
        svc = _get_service()
        kwargs = {}
        if cfg:
            for field in (
                "breaking_interval_minutes", "daily_refresh_hour_1",
                "daily_refresh_hour_2", "seed_crawl_hour",
                "seed_urls", "news_refresh_queries",
                "academic_refresh_queries", "web_refresh_queries",
            ):
                val = getattr(cfg, field, None)
                if val is not None:
                    kwargs[field] = val
        _scheduler = OnlineSearchScheduler(svc, **kwargs)
        _scheduler.start()
        _scheduler_running = True
    return {"status": "started", "jobs": _scheduler_job_list()}


@router.post("/scheduler/stop")
def scheduler_stop():
    global _scheduler_running
    with _scheduler_lock:
        if _scheduler is not None and _scheduler_running:
            _scheduler.shutdown()
            _scheduler_running = False
    return {"status": "stopped"}


@router.post("/scheduler/run-now")
def scheduler_run_now(background_tasks: BackgroundTasks):
    """Trigger all scheduler jobs immediately (runs in background)."""
    if _scheduler is None or not _scheduler_running:
        # One-off: build a scheduler just for run_all_now
        svc = _get_service()
        temp = OnlineSearchScheduler(svc)
        background_tasks.add_task(temp.run_all_now)
    else:
        background_tasks.add_task(_scheduler.run_all_now)
    return {"status": "triggered"}


@router.put("/scheduler/config")
def update_scheduler_config(cfg: SchedulerConfigReq):
    """Apply new config — restarts the scheduler if it is running."""
    global _scheduler, _scheduler_running
    with _scheduler_lock:
        was_running = _scheduler_running
        if was_running and _scheduler is not None:
            _scheduler.shutdown()
            _scheduler_running = False

        svc = _get_service()
        kwargs = {k: v for k, v in cfg.model_dump().items() if v is not None}
        _scheduler = OnlineSearchScheduler(svc, **kwargs)

        if was_running:
            _scheduler.start()
            _scheduler_running = True

    return {
        "status": "restarted" if was_running else "config_saved",
        "config": _current_config(),
    }

"""
New FastAPI router mounted at /api/v1/online-search:

Method	Path	Purpose
POST	/web /news /academic /medical	Domain-specific search
POST	/url	Read & cache a URL
GET	/cache	Browse cached records (domain/category/limit filters)
GET/POST/POST/PUT	/scheduler/status /start /stop /run-now	Scheduler lifecycle
PUT	/scheduler/config	Update config + restart scheduler
"""
