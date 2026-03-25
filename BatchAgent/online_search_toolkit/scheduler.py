from __future__ import annotations

import logging
from typing import List, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from .service import OnlineSearchService

logger = logging.getLogger(__name__)

# Default health/medical seed URLs crawled daily
DEFAULT_SEED_URLS: List[str] = [
    # NIH / NLM
    "https://www.nih.gov/health-information",
    "https://www.ncbi.nlm.nih.gov/books/NBK11828/",        # Dietary Reference Intakes
    "https://www.nhlbi.nih.gov/health/heart-healthy-living",
    # FDA
    "https://www.fda.gov/food/nutrition-food-labeling-and-critical-foods",
    "https://www.fda.gov/consumers/consumer-updates/dietary-supplements",
    # CDC
    "https://www.cdc.gov/nutrition/index.html",
    "https://www.cdc.gov/physicalactivity/basics/adults/index.htm",
    "https://www.cdc.gov/vaccines/schedules/index.html",
    # WHO
    "https://www.who.int/news-room/fact-sheets/detail/healthy-diet",
    "https://www.who.int/news-room/fact-sheets/detail/physical-activity",
    # Harvard HSPH
    "https://www.hsph.harvard.edu/nutritionsource/healthy-eating-plate/",
    "https://www.hsph.harvard.edu/nutritionsource/staying-active/",
    # Mayo Clinic
    "https://www.mayoclinic.org/healthy-lifestyle/nutrition-and-healthy-eating/basics/nutrition-basics/hlv-20049477",
    "https://www.mayoclinic.org/healthy-lifestyle/fitness/basics/fitness-basics/hlv-20049447",
    "https://www.mayoclinic.org/healthy-lifestyle/childrens-health/basics/childrens-health/hlv-20049425",
    # AAP / HealthyChildren
    "https://www.healthychildren.org/English/healthy-living/nutrition/Pages/default.aspx",
    "https://www.healthychildren.org/English/healthy-living/fitness/Pages/default.aspx",
    # USDA
    "https://www.dietaryguidelines.gov/resources/2020-2025-dietary-guidelines-online-materials",
]


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
        seed_urls: Optional[List[str]] = None,
        seed_crawl_hour: Optional[int] = None,
    ):
        self.service = service
        cfg = service.config.scheduler

        self.timezone = timezone or cfg.timezone
        self.breaking_interval_minutes = breaking_interval_minutes or cfg.breaking_interval_minutes
        self.daily_refresh_hour_1 = daily_refresh_hour_1 if daily_refresh_hour_1 is not None else cfg.daily_refresh_hour_1
        self.daily_refresh_hour_2 = daily_refresh_hour_2 if daily_refresh_hour_2 is not None else cfg.daily_refresh_hour_2
        self.seed_crawl_hour = seed_crawl_hour if seed_crawl_hour is not None else 3  # 3 AM by default

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
        self.seed_urls = seed_urls if seed_urls is not None else DEFAULT_SEED_URLS

        self.scheduler = BackgroundScheduler(timezone=self.timezone)

    # ------------------------------------------------------------------
    # Start / stop
    # ------------------------------------------------------------------

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
            id="news_refresh_job",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        self.scheduler.add_job(
            self._run_academic_refresh_job,
            trigger=CronTrigger(hour=self.daily_refresh_hour_1, minute=20),
            id="academic_refresh_job",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        self.scheduler.add_job(
            self._run_web_refresh_job,
            trigger=CronTrigger(hour=self.daily_refresh_hour_2, minute=0),
            id="web_refresh_job",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        self.scheduler.add_job(
            self._run_url_seed_job,
            trigger=CronTrigger(hour=self.seed_crawl_hour, minute=30),
            id="url_seed_job",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )

        self.scheduler.start()
        logger.info(
            "Scheduler started. Jobs: breaking every %dm, news/academic at %d:00/%d:20, "
            "web at %d:00, url-seed at %d:30.",
            self.breaking_interval_minutes,
            self.daily_refresh_hour_1, self.daily_refresh_hour_1,
            self.daily_refresh_hour_2,
            self.seed_crawl_hour,
        )

    def shutdown(self) -> None:
        self.scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped.")

    def run_all_now(self) -> None:
        """Run every job once immediately (useful for testing / first-run warm-up)."""
        logger.info("run_all_now: triggering all jobs immediately.")
        for job_fn, label in [
            (self._run_breaking_job,      "breaking_news"),
            (self._run_news_refresh_job,  "news_refresh"),
            (self._run_academic_refresh_job, "academic_refresh"),
            (self._run_web_refresh_job,   "web_refresh"),
            (self._run_url_seed_job,      "url_seed"),
        ]:
            logger.info("Running job: %s", label)
            job_fn()
        logger.info("run_all_now: all jobs complete.")

    # ------------------------------------------------------------------
    # Job implementations
    # ------------------------------------------------------------------

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

    def _run_url_seed_job(self) -> None:
        """Crawl the health/info seed URLs and persist them to cache."""
        ok, fail = 0, 0
        for url in self.seed_urls:
            try:
                # Infer domain from URL
                domain = "medical"
                if any(d in url for d in ("nih.gov", "fda.gov", "cdc.gov", "who.int",
                                          "mayoclinic.org", "healthychildren.org",
                                          "hsph.harvard.edu", "dietaryguidelines.gov")):
                    domain = "medical"
                record = self.service.read_url(
                    url,
                    domain=domain,
                    category="health_guidelines",
                    persist=True,
                    force_refresh=False,
                )
                logger.debug("Seeded: %s (%s)", record.title[:60], url)
                ok += 1
            except Exception:
                logger.warning("URL seed failed for %s", url, exc_info=False)
                fail += 1
        logger.info("URL seed job done: %d ok, %d failed out of %d URLs.", ok, fail, len(self.seed_urls))
