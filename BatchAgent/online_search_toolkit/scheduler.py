from __future__ import annotations

import logging
from typing import List, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from .service import OnlineSearchService

logger = logging.getLogger(__name__)


class OnlineSearchScheduler:
    def __init__(
        self,
        service: OnlineSearchService,
        *,
        timezone: Optional[str] = None,
        breaking_interval_minutes: Optional[int] = None,
        daily_refresh_hour_1: Optional[int] = None,
        daily_refresh_hour_2: Optional[int] = None,
        news_refresh_queries: Optional[List[str]] = None,
        academic_refresh_queries: Optional[List[str]] = None,
        web_refresh_queries: Optional[List[str]] = None,
    ):
        self.service = service
        cfg = service.config.scheduler

        self.timezone = timezone or cfg.timezone
        self.breaking_interval_minutes = breaking_interval_minutes or cfg.breaking_interval_minutes
        self.daily_refresh_hour_1 = daily_refresh_hour_1 if daily_refresh_hour_1 is not None else cfg.daily_refresh_hour_1
        self.daily_refresh_hour_2 = daily_refresh_hour_2 if daily_refresh_hour_2 is not None else cfg.daily_refresh_hour_2

        self.news_refresh_queries = news_refresh_queries or [
            "OpenAI",
            "Anthropic",
            "Google DeepMind",
            "AI regulation",
            "robotics",
        ]
        self.academic_refresh_queries = academic_refresh_queries or [
            "large language models",
            "autonomous driving",
            "radar signal processing",
            "mental health treatment",
        ]
        self.web_refresh_queries = web_refresh_queries or [
            "latest python packaging changes",
            "NVIDIA Jetson news",
            "vLLM updates",
        ]

        self.scheduler = BackgroundScheduler(timezone=self.timezone)

    def start(self) -> None:
        self.scheduler.add_job(
            self._run_breaking_job,
            trigger=IntervalTrigger(minutes=self.breaking_interval_minutes),
            id="breaking_news_job",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )

        self.scheduler.add_job(
            self._run_news_refresh_job,
            trigger=CronTrigger(hour=self.daily_refresh_hour_1, minute=0),
            id="news_refresh_job_1",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )

        self.scheduler.add_job(
            self._run_academic_refresh_job,
            trigger=CronTrigger(hour=self.daily_refresh_hour_1, minute=20),
            id="academic_refresh_job_1",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )

        self.scheduler.add_job(
            self._run_web_refresh_job,
            trigger=CronTrigger(hour=self.daily_refresh_hour_2, minute=0),
            id="web_refresh_job_2",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )

        self.scheduler.start()
        logger.info("Online search scheduler started.")

    def shutdown(self) -> None:
        self.scheduler.shutdown(wait=False)
        logger.info("Online search scheduler stopped.")

    def _run_breaking_job(self) -> None:
        try:
            result = self.service.search_news(
                query="breaking news",
                limit=20,
                language="mixed",
                category="breaking",
            )
            logger.info("Breaking refresh collected %s records", result.count)
        except Exception:
            logger.exception("Breaking refresh job failed")

    def _run_news_refresh_job(self) -> None:
        try:
            result = self.service.refresh_domain_cache(
                domain="news",
                queries=self.news_refresh_queries,
                limit=10,
                language="mixed",
            )
            logger.info("News refresh completed: %s", result)
        except Exception:
            logger.exception("News refresh job failed")

    def _run_academic_refresh_job(self) -> None:
        try:
            result = self.service.refresh_domain_cache(
                domain="academic",
                queries=self.academic_refresh_queries,
                limit=10,
                language="en",
            )
            logger.info("Academic refresh completed: %s", result)
        except Exception:
            logger.exception("Academic refresh job failed")

    def _run_web_refresh_job(self) -> None:
        try:
            result = self.service.refresh_domain_cache(
                domain="general",
                queries=self.web_refresh_queries,
                limit=10,
                language="mixed",
            )
            logger.info("Web refresh completed: %s", result)
        except Exception:
            logger.exception("Web refresh job failed")