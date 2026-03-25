from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time

from .bootstrap import create_online_search_service
from .config import SearchConfig
from .scheduler import OnlineSearchScheduler
from typing import Any


def print_result(result: Any) -> None:
    data = result.model_dump(mode="json")

    def truncate_embeddings(obj):
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                if k == "embedding" and isinstance(v, list):
                    obj[k] = f"[Embedding vector of size {len(v)}]"
                else:
                    truncate_embeddings(v)
        elif isinstance(obj, list):
            for item in obj:
                truncate_embeddings(item)

    truncate_embeddings(data)
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

    parser.add_argument("--api_url", type=str, default="http://localhost:8081/v1")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--api_model", type=str, default="text-embeddings-inference")

    parser.add_argument("--postgres_enabled", action="store_true")
    parser.add_argument("--postgres_dsn", type=str, default=None)

    parser.add_argument("--force_refresh", action="store_true")
    parser.add_argument("--use_crawler", action="store_true")

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

    if args.command == "search":
        result = service.search_web(
            query=args.query,
            limit=args.limit,
            language=args.language,
            category=args.category,
            enable_youtube=args.enable_youtube,
        )
        print_result(result)
        return 0

    if args.command == "news":
        result = service.search_news(
            query=args.query,
            limit=args.limit,
            language=args.language,
            category=args.category,
        )
        print_result(result)
        return 0

    if args.command == "academic":
        result = service.search_academic(
            query=args.query,
            limit=args.limit,
            language="en",
            category=args.category,
        )
        print_result(result)
        return 0

    if args.command == "medical":
        result = service.search_medical(
            query=args.query,
            limit=args.limit,
            language="en",
            category=args.category,
        )
        print_result(result)
        return 0

    if args.command == "read_url":
        result = service.read_url(
            url=args.url,
            domain="general",
            category=args.category or "general",
            persist=True,
            force_refresh=args.force_refresh,
            use_crawler=args.use_crawler,
        )
        print_result(result)
        return 0

    if args.command == "scheduler":
        scheduler = OnlineSearchScheduler(service)
        scheduler.start()

        def _stop(*_):
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
python -m online_search_toolkit.cli search --query "vllm paged attention" --limit 5

python -m online_search_toolkit.cli news --query "OpenAI" --language mixed --limit 5

python -m online_search_toolkit.cli academic --query "large language model reasoning" --limit 5

python -m online_search_toolkit.cli medical --query "depression treatment" --limit 5

python -m online_search_toolkit.cli read_url --url "https://example.com"

python -m online_search_toolkit.cli read_url --url "https://arxiv.org/pdf/1706.03762.pdf"
"""