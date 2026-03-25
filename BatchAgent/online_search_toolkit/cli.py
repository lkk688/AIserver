from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from collections import Counter
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.text import Text

from .bootstrap import create_online_search_service
from .config import SearchConfig
from .models import SearchRequest
from .scheduler import OnlineSearchScheduler

console = Console()

# Color palette per source type
_SOURCE_COLORS = {
    "PubMed": "green",
    "MedlinePlus": "cyan",
    "NIMH": "blue",
    "arXiv": "magenta",
    "Semantic Scholar": "yellow",
    "Wikipedia": "white",
    "GNews": "bright_blue",
    "NewsData": "bright_cyan",
    "NewsAPI": "bright_green",
    "Serper": "bright_magenta",
    "Tavily": "bright_yellow",
    "Crawler": "dim white",
}

def _source_color(source: str) -> str:
    for key, color in _SOURCE_COLORS.items():
        if key.lower() in source.lower():
            return color
    return "white"


def _truncate_embeddings(obj: Any) -> None:
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if k == "embedding" and isinstance(v, list):
                obj[k] = f"[Embedding vector of size {len(v)}]"
            else:
                _truncate_embeddings(v)
    elif isinstance(obj, list):
        for item in obj:
            _truncate_embeddings(item)


def print_result(result: Any) -> None:
    """Print a SearchResult or SearchRecord with rich source breakdown + JSON."""
    # ── Source breakdown summary (only for SearchResult which has .items) ──
    items = getattr(result, "items", None)
    if items is not None:
        source_counts: Counter = Counter(r.source for r in items)
        domain = getattr(result, "domain", "")
        query = getattr(result, "query", "")

        # Header panel
        console.print(Panel(
            f"[bold]{query}[/bold]  [dim]domain={domain}  total={len(items)}[/dim]",
            title="[bold cyan]Search Results[/bold cyan]",
            border_style="cyan",
        ))

        # Source breakdown table
        table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
        table.add_column("Source", style="bold")
        table.add_column("Count", justify="right")
        table.add_column("Bar")

        max_count = max(source_counts.values(), default=1)
        for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
            color = _source_color(source)
            bar = "[{color}]{bar}[/{color}]".format(
                color=color,
                bar="\u2588" * max(1, round(count / max_count * 20)),
            )
            table.add_row(
                Text(source, style=color),
                str(count),
                Text.from_markup(bar),
            )

        console.print(table)

    # ── Full JSON dump ──
    data = result.model_dump(mode="json")
    _truncate_embeddings(data)
    print(json.dumps(data, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Online Search Toolkit CLI")
    parser.add_argument(
        "command",
        choices=[
            "search",
            "news",
            "academic",
            "medical",
            "read_url",
            "scheduler",
        ],
    )
    parser.add_argument("--query", type=str, default="")
    parser.add_argument("--url", type=str, default="")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--language", type=str, default="mixed")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--enable_youtube", action="store_true")

    parser.add_argument("--api_url", type=str, default="http://100.65.193.60:8081/v1")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--api_model", type=str, default="text-embeddings-inference")

    parser.add_argument("--postgres_enabled", action="store_true")
    parser.add_argument("--postgres_dsn", type=str, default=None)

    parser.add_argument("--force_refresh", action="store_true")
    parser.add_argument("--use_crawler", action="store_true")
    parser.add_argument("--no_cache", action="store_true", help="Skip local cache and fetch fresh results")
    parser.add_argument("--run_now", action="store_true", help="(scheduler) Run all jobs once immediately at startup")

    parser.add_argument("--log_level", type=str, default="INFO")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = SearchConfig()
    config.embedding.api_url = args.api_url
    config.embedding.api_key = args.api_key
    config.embedding.api_model = args.api_model

    if args.postgres_enabled:
        config.store.postgres_enabled = True
    if args.postgres_dsn:
        config.store.postgres_dsn = args.postgres_dsn

    service = create_online_search_service(config)

    def _run_with_spinner(label: str, fn):
        """Run fn() behind a rich spinner; return its result."""
        with console.status(f"[bold cyan]{label}[/bold cyan]", spinner="dots"):
            return fn()

    use_cache = not args.no_cache

    if args.command == "search":
        result = _run_with_spinner(
            f"Searching web for '{args.query}'...",
            lambda: service.search(SearchRequest(
                query=args.query,
                domain="general",
                limit=args.limit,
                language=args.language,
                category=args.category,
                enable_youtube=args.enable_youtube,
                use_cache=use_cache,
            )),
        )
        print_result(result)
        return 0

    if args.command == "news":
        result = _run_with_spinner(
            f"Searching news for '{args.query}'...",
            lambda: service.search(SearchRequest(
                query=args.query,
                domain="news",
                limit=args.limit,
                language=args.language,
                category=args.category,
                use_cache=use_cache,
            )),
        )
        print_result(result)
        return 0

    if args.command == "academic":
        result = _run_with_spinner(
            f"Searching academic sources for '{args.query}'...",
            lambda: service.search(SearchRequest(
                query=args.query,
                domain="academic",
                limit=args.limit,
                language="en",
                category=args.category,
                use_cache=use_cache,
            )),
        )
        print_result(result)
        return 0

    if args.command == "medical":
        result = _run_with_spinner(
            f"Searching medical sources for '{args.query}'...",
            lambda: service.search(SearchRequest(
                query=args.query,
                domain="medical",
                limit=args.limit,
                language="en",
                category=args.category,
                use_cache=use_cache,
            )),
        )
        print_result(result)
        return 0

    if args.command == "read_url":
        result = _run_with_spinner(
            f"Reading URL '{args.url}'...",
            lambda: service.read_url(
                url=args.url,
                domain="general",
                category=args.category or "general",
                persist=True,
                force_refresh=args.force_refresh,
                use_crawler=args.use_crawler,
            ),
        )
        print_result(result)
        return 0

    if args.command == "scheduler":
        scheduler = OnlineSearchScheduler(service)
        scheduler.start()

        if args.run_now:
            console.print("[bold yellow]--run_now: triggering all jobs immediately...[/bold yellow]")
            with console.status("[bold cyan]Running all scheduler jobs now...[/bold cyan]", spinner="dots"):
                scheduler.run_all_now()
            console.print("[bold green]All jobs done. Scheduler continuing on schedule.[/bold green]")

        job_table = Table(box=box.SIMPLE, show_header=False)
        job_table.add_row("[cyan]breaking_news[/cyan]",  f"every {service.config.scheduler.breaking_interval_minutes} min")
        job_table.add_row("[cyan]news_refresh[/cyan]",   f"daily at {service.config.scheduler.daily_refresh_hour_1}:00")
        job_table.add_row("[cyan]academic_refresh[/cyan]", f"daily at {service.config.scheduler.daily_refresh_hour_1}:20")
        job_table.add_row("[cyan]web_refresh[/cyan]",    f"daily at {service.config.scheduler.daily_refresh_hour_2}:00")
        job_table.add_row("[cyan]url_seed[/cyan]",       "daily at 03:30  (crawls health URLs → .cache/search/medical.json)")
        console.print(Panel(
            job_table,
            title="[bold cyan]Online Search Scheduler[/bold cyan]",
            subtitle="Press [bold]Ctrl+C[/bold] to stop",
            border_style="cyan",
        ))

        def _stop(*_):
            console.print("\n[bold red]Shutting down scheduler...[/bold red]")
            scheduler.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, _stop)
        signal.signal(signal.SIGTERM, _stop)

        while True:
            time.sleep(60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
# ─────────────────────────────────────────────────────────────
#  General web search
# ─────────────────────────────────────────────────────────────
python -m online_search_toolkit.cli search --query "vllm paged attention" --limit 5
python -m online_search_toolkit.cli search --query "Python packaging PEP 517" --limit 8
python -m online_search_toolkit.cli search --query "NVIDIA Jetson Orin benchmarks" --limit 5

# ─────────────────────────────────────────────────────────────
#  News search
# ─────────────────────────────────────────────────────────────
python -m online_search_toolkit.cli news --query "OpenAI" --language mixed --limit 5
python -m online_search_toolkit.cli news --query "AI regulation EU Act" --language en --limit 8
python -m online_search_toolkit.cli news --query "breaking news" --category breaking --limit 10

# ─────────────────────────────────────────────────────────────
#  Academic search  →  PubMed [✓] + arXiv [✓] + Semantic Scholar [✓]
# ─────────────────────────────────────────────────────────────
python -m online_search_toolkit.cli academic --query "large language model reasoning" --limit 5
python -m online_search_toolkit.cli academic --query "GLP-1 receptor agonist weight loss clinical trial" --limit 5 --no_cache
python -m online_search_toolkit.cli academic --query "Mediterranean diet cardiovascular outcomes meta-analysis" --limit 5 --no_cache
python -m online_search_toolkit.cli academic --query "HIIT high intensity interval training metabolic syndrome" --limit 5 --no_cache
python -m online_search_toolkit.cli academic --query "vitamin D supplementation immune function RCT" --limit 5 --no_cache

# ─────────────────────────────────────────────────────────────
#  Medical search  — SOURCE STATUS (as of 2026-03)
#
#  ✓ PubMed        — eutils.ncbi.nlm.nih.gov  (free JSON API, no key)
#  ✓ Europe PMC    — ebi.ac.uk/europepmc       (free JSON API; covers PubMed +
#                    PMC + WHO technical docs + preprints)
#  ✓ MedlinePlus   — wsearch.nlm.nih.gov       (free XML API; aggregates NIH,
#                    CDC, AAFP consumer health topics; short queries ≤4 words)
#  ✗ CDC search    — search.cdc.gov            (JS-rendered, no parseable API;
#                    use read_url for specific CDC pages, or Europe PMC)
#  ✗ WHO IRIS      — iris.who.int/rest/search  (403 Forbidden without auth;
#                    WHO guidelines available via Europe PMC)
#  ✗ NIMH          — nimh.nih.gov/search       (403 Forbidden)
#
#  Always use --no_cache to bypass stale cache and fetch live results.
# ─────────────────────────────────────────────────────────────

# Mental health
python -m online_search_toolkit.cli medical --query "depression treatment guidelines" --limit 5 --no_cache
python -m online_search_toolkit.cli medical --query "anxiety disorder CBT efficacy" --limit 5 --no_cache

# Diet & nutrition
python -m online_search_toolkit.cli medical --query "dietary guidelines protein intake" --limit 5 --no_cache
python -m online_search_toolkit.cli medical --query "omega-3 fatty acids heart health" --limit 5 --no_cache
python -m online_search_toolkit.cli medical --query "intermittent fasting metabolic health" --limit 5 --no_cache
python -m online_search_toolkit.cli medical --query "plant-based diet cancer prevention" --limit 5 --no_cache

# Fitness & exercise
python -m online_search_toolkit.cli medical --query "physical activity guidelines adults" --limit 5 --no_cache
python -m online_search_toolkit.cli medical --query "strength training sarcopenia prevention" --limit 5 --no_cache
python -m online_search_toolkit.cli medical --query "aerobic exercise diabetes management" --limit 5 --no_cache

# General health maintenance
python -m online_search_toolkit.cli medical --query "preventive health screenings age" --limit 5 --no_cache
python -m online_search_toolkit.cli medical --query "hypertension lifestyle intervention" --limit 5 --no_cache
python -m online_search_toolkit.cli medical --query "cholesterol statin guidelines" --limit 5 --no_cache
python -m online_search_toolkit.cli medical --query "vaccine schedule adults" --limit 5 --no_cache

# Child health  (AAP content found via Europe PMC + PubMed Pediatrics journal)
python -m online_search_toolkit.cli medical --query "child nutrition infant feeding" --limit 5 --no_cache
python -m online_search_toolkit.cli medical --query "childhood vaccine immunization schedule" --limit 5 --no_cache
python -m online_search_toolkit.cli medical --query "pediatric sleep recommendations" --limit 5 --no_cache
python -m online_search_toolkit.cli medical --query "child growth development milestones" --limit 5 --no_cache
python -m online_search_toolkit.cli medical --query "screen time children guidelines" --limit 5 --no_cache

# FDA drug & supplement safety
python -m online_search_toolkit.cli medical --query "dietary supplement safety regulations" --limit 5 --no_cache
python -m online_search_toolkit.cli medical --query "obesity medications approval" --limit 5 --no_cache

# ─────────────────────────────────────────────────────────────
#  URL reader — ingest specific pages into .cache/search/medical.json
#  (these are also crawled automatically by the scheduler's url_seed job)
# ─────────────────────────────────────────────────────────────
# Research papers
python -m online_search_toolkit.cli read_url --url "https://arxiv.org/pdf/1706.03762.pdf"
python -m online_search_toolkit.cli read_url --url "https://huggingface.co/Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled"

# NIH / NLM  [✓ static HTML, readable]
python -m online_search_toolkit.cli read_url --url "https://www.nih.gov/health-information"
python -m online_search_toolkit.cli read_url --url "https://www.ncbi.nlm.nih.gov/books/NBK11828/"
python -m online_search_toolkit.cli read_url --url "https://www.nhlbi.nih.gov/health/heart-healthy-living"

# FDA  [✓ static HTML, readable]
python -m online_search_toolkit.cli read_url --url "https://www.fda.gov/food/nutrition-food-labeling-and-critical-foods"
python -m online_search_toolkit.cli read_url --url "https://www.fda.gov/consumers/consumer-updates/dietary-supplements"

# CDC  [✓ topic pages are static HTML, readable via read_url]
python -m online_search_toolkit.cli read_url --url "https://www.cdc.gov/nutrition/index.html"
python -m online_search_toolkit.cli read_url --url "https://www.cdc.gov/physicalactivity/basics/adults/index.htm"
python -m online_search_toolkit.cli read_url --url "https://www.cdc.gov/vaccines/schedules/index.html"

# WHO fact sheets  [✓ static HTML, readable]
python -m online_search_toolkit.cli read_url --url "https://www.who.int/news-room/fact-sheets/detail/healthy-diet"
python -m online_search_toolkit.cli read_url --url "https://www.who.int/news-room/fact-sheets/detail/physical-activity"

# Harvard HSPH  [✓ static HTML, readable]
python -m online_search_toolkit.cli read_url --url "https://www.hsph.harvard.edu/nutritionsource/healthy-eating-plate/"
python -m online_search_toolkit.cli read_url --url "https://www.hsph.harvard.edu/nutritionsource/staying-active/"

# Mayo Clinic  [✓ static HTML, readable]
python -m online_search_toolkit.cli read_url --url "https://www.mayoclinic.org/healthy-lifestyle/nutrition-and-healthy-eating/basics/nutrition-basics/hlv-20049477"
python -m online_search_toolkit.cli read_url --url "https://www.mayoclinic.org/healthy-lifestyle/fitness/basics/fitness-basics/hlv-20049447"
python -m online_search_toolkit.cli read_url --url "https://www.mayoclinic.org/healthy-lifestyle/childrens-health/basics/childrens-health/hlv-20049425"

# AAP / HealthyChildren  [✓ static HTML, readable]
python -m online_search_toolkit.cli read_url --url "https://www.healthychildren.org/English/healthy-living/nutrition/Pages/default.aspx"
python -m online_search_toolkit.cli read_url --url "https://www.healthychildren.org/English/healthy-living/fitness/Pages/default.aspx"

# USDA  [✓ static HTML, readable]
python -m online_search_toolkit.cli read_url --url "https://www.dietaryguidelines.gov/resources/2020-2025-dietary-guidelines-online-materials"

# ─────────────────────────────────────────────────────────────
#  Scheduler — background periodic refresh
#  Cache location: .cache/search/{news,academic,medical,general}.json
#  Jobs:
#    breaking_news    every 30 min  → news.json
#    news_refresh     daily 07:00   → news.json
#    academic_refresh daily 07:20   → academic.json
#    web_refresh      daily 19:00   → general.json
#    url_seed         daily 03:30   → medical.json  (crawls health URLs above)
# ─────────────────────────────────────────────────────────────

# Start scheduler (normal mode — runs on schedule)
python -m online_search_toolkit.cli scheduler

# Start scheduler AND run all jobs immediately right now (useful for first-run / testing)
python -m online_search_toolkit.cli scheduler --run_now

# Verbose logging
python -m online_search_toolkit.cli scheduler --run_now --log_level DEBUG

# With postgres
python -m online_search_toolkit.cli scheduler --postgres_enabled --postgres_dsn "postgresql://user:pass@localhost:5432/search_db"

# ─────────────────────────────────────────────────────────────
#  Flags reference
# ─────────────────────────────────────────────────────────────
# --limit N              max results returned (default: 8)
# --language en|zh|mixed filter by language (default: mixed)
# --category CATEGORY    optional topic category tag
# --enable_youtube       include YouTube results in web search
# --no_cache             skip local cache; fetch fresh results from PubMed/arXiv/etc.
# --force_refresh        bypass cache and re-fetch the URL (read_url only)
# --use_crawler          use headless crawler for JS-rendered pages
# --api_url URL          embedding service base URL (default: http://localhost:8081/v1)
# --api_model MODEL      embedding model name
# --postgres_enabled     enable pgvector persistence
# --postgres_dsn DSN     PostgreSQL connection string
# --log_level LEVEL      logging verbosity: DEBUG|INFO|WARNING|ERROR
"""