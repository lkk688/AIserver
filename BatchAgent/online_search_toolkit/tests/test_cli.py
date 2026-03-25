"""
Unit tests for online_search_toolkit CLI.
"""
from __future__ import annotations

import argparse
import sys
from unittest.mock import patch, MagicMock

import pytest

from BatchAgent.online_search_toolkit.cli import build_parser, main
from BatchAgent.online_search_toolkit.models import SearchResult, URLReadResult, SearchRecord
from BatchAgent.online_search_toolkit.utils import utc_now


@pytest.fixture
def mock_service():
    with patch("BatchAgent.online_search_toolkit.cli.create_online_search_service") as mock_create:
        service_instance = MagicMock()
        mock_create.return_value = service_instance
        yield service_instance


def make_dummy_search_result(query: str) -> SearchResult:
    return SearchResult(
        query=query,
        domain="general",
        count=1,
        items=[
            SearchRecord(
                id="test_id",
                record_type="web_page",
                source_type="search_api",
                title="Test Title",
                summary="Test Summary",
                url="https://example.com",
                source="test",
                domain="general",
                language="en",
                category="general",
                fetched_at=utc_now(),
                metadata={},
                embedding=[0.1, 0.2, 0.3], # to test truncation implicitly
            )
        ]
    )


def test_build_parser():
    parser = build_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    args = parser.parse_args(["search", "--query", "test"])
    assert args.command == "search"
    assert args.query == "test"


def test_cli_search(mock_service, capsys):
    mock_service.search_web.return_value = make_dummy_search_result("vllm")
    
    test_args = ["cli.py", "search", "--query", "vllm", "--limit", "5"]
    with patch.object(sys, 'argv', test_args):
        ret = main()
        assert ret == 0
        mock_service.search_web.assert_called_once_with(
            query="vllm", limit=5, language="mixed", category=None, enable_youtube=False
        )
        
    captured = capsys.readouterr()
    assert "[Embedding vector of size 3]" in captured.out
    assert '"embedding": [' not in captured.out


def test_cli_news(mock_service, capsys):
    mock_service.search_news.return_value = make_dummy_search_result("OpenAI")
    
    test_args = ["cli.py", "news", "--query", "OpenAI", "--language", "mixed", "--limit", "3"]
    with patch.object(sys, 'argv', test_args):
        ret = main()
        assert ret == 0
        mock_service.search_news.assert_called_once_with(
            query="OpenAI", limit=3, language="mixed", category=None
        )


def test_cli_academic(mock_service, capsys):
    mock_service.search_academic.return_value = make_dummy_search_result("reasoning")
    
    test_args = ["cli.py", "academic", "--query", "reasoning", "--limit", "2"]
    with patch.object(sys, 'argv', test_args):
        ret = main()
        assert ret == 0
        mock_service.search_academic.assert_called_once_with(
            query="reasoning", limit=2, language="en", category=None
        )


def test_cli_medical(mock_service, capsys):
    mock_service.search_medical.return_value = make_dummy_search_result("depression")
    
    test_args = ["cli.py", "medical", "--query", "depression", "--limit", "5"]
    with patch.object(sys, 'argv', test_args):
        ret = main()
        assert ret == 0
        mock_service.search_medical.assert_called_once_with(
            query="depression", limit=5, language="en", category=None
        )


def test_cli_read_url(mock_service, capsys):
    record = SearchRecord(
        id="test_url_id",
        record_type="web_page",
        source_type="url_fetch",
        title="Test URL",
        summary="Summary of URL",
        url="https://example.com/page",
        source="test",
        domain="general",
        language="en",
        category="general",
        fetched_at=utc_now(),
        metadata={},
    )
    mock_service.read_url.return_value = URLReadResult(record=record, from_cache=False)
    
    test_args = ["cli.py", "read_url", "--url", "https://example.com/page"]
    with patch.object(sys, 'argv', test_args):
        ret = main()
        assert ret == 0
        mock_service.read_url.assert_called_once_with(
            url="https://example.com/page",
            domain="general",
            category="general",
            persist=True,
            force_refresh=False,
            use_crawler=False,
        )


def test_cli_scheduler(mock_service):
    with patch("BatchAgent.online_search_toolkit.cli.OnlineSearchScheduler") as mock_scheduler_cls:
        mock_scheduler_instance = mock_scheduler_cls.return_value
        test_args = ["cli.py", "scheduler"]
        
        # We need to mock time.sleep so the test doesn't actually wait
        with patch("time.sleep", side_effect=InterruptedError("stop loop")):
            with patch.object(sys, 'argv', test_args):
                with pytest.raises(InterruptedError, match="stop loop"):
                    main()
        
        mock_scheduler_cls.assert_called_once_with(mock_service)
        mock_scheduler_instance.start.assert_called_once()
