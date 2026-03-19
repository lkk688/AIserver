"""
Tests for document_agent_v1 pipeline.

Run with:
    cd /Developer/AIserver
    pytest BatchAgent/document_agent/test_document_agent.py -v
"""
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest
import numpy as np
from typing import List, Dict, Any
from unittest.mock import patch, MagicMock

# Pre-import the module so @patch can resolve the dotted path
import BatchAgent.document_agent.document_agent_v1 as dav1
from BatchAgent.document_agent.document_agent_v1 import (
    StructuredDocumentAgent,
    ChunkRAGIndex,
    get_embeddings,
    create_full_pipeline_from_pdf,
    register_pipeline,
    unregister_pipeline,
    get_active_chunk_index,
    get_active_section_agent,
    _active_pipelines,
)
from BatchAgent.document_agent.chunking import chunk_text_blocks

_PATCH_PREFIX = "BatchAgent.document_agent.document_agent_v1"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_blocks() -> List[Dict[str, Any]]:
    """Synthetic text blocks mimicking parser.extract_text_blocks() output."""
    return [
        {
            "page_no": 1, "block_index": 0,
            "text": "Abstract This paper introduces a novel framework for analyzing data using layout-aware extraction.",
            "bbox": {"x0": 72, "y0": 100, "x1": 540, "y1": 120},
            "metadata": {
                "first_line": "Abstract",
                "max_font_size": 16.0, "avg_font_size": 12.0, "min_font_size": 12.0,
                "bold_ratio": 1.0, "font_names": ["TimesNewRoman-Bold"],
                "line_count": 2, "char_count": 90,
            },
        },
        {
            "page_no": 1, "block_index": 1,
            "text": "Data analysis is a critical challenge in modern computing. We present a method that leverages layout-aware extraction combined with hierarchical indexing to enable efficient document understanding by large language models.",
            "bbox": {"x0": 72, "y0": 130, "x1": 540, "y1": 200},
            "metadata": {
                "first_line": "Data analysis is a critical challenge",
                "max_font_size": 11.0, "avg_font_size": 11.0, "min_font_size": 11.0,
                "bold_ratio": 0.0, "font_names": ["TimesNewRoman"],
                "line_count": 4, "char_count": 220,
            },
        },
        {
            "page_no": 2, "block_index": 0,
            "text": "Our methodology relies on hybrid extraction using bounding boxes and layout analysis to build hierarchical document structures for retrieval.",
            "bbox": {"x0": 72, "y0": 100, "x1": 540, "y1": 180},
            "metadata": {
                "first_line": "Our methodology relies on hybrid extraction",
                "max_font_size": 11.0, "avg_font_size": 11.0, "min_font_size": 11.0,
                "bold_ratio": 0.0, "font_names": ["TimesNewRoman"],
                "line_count": 3, "char_count": 140,
            },
        },
        {
            "page_no": 2, "block_index": 1,
            "text": "Results show significant improvement in retrieval accuracy compared to naive full-text search approaches across multiple benchmark datasets.",
            "bbox": {"x0": 72, "y0": 190, "x1": 540, "y1": 240},
            "metadata": {
                "first_line": "Results show significant improvement",
                "max_font_size": 11.0, "avg_font_size": 11.0, "min_font_size": 11.0,
                "bold_ratio": 0.0, "font_names": ["TimesNewRoman"],
                "line_count": 2, "char_count": 140,
            },
        },
    ]


@pytest.fixture
def sample_markdown() -> str:
    """Sample markdown with page markers and sections."""
    return """
# Abstract
This paper introduces a novel framework for analyzing data.

[PAGE: 1]
# 1. Introduction
Data analysis is hard. We present a new method.
## 1.1 Background
Historically, people used abacuses.
## 1.2 Motivation
We need faster tools because data is growing.

[PAGE: 2]
# 2. Methodology
Our method relies on hybrid extraction.
## 2.1 The OpenDataLoader Approach
We use bounding boxes and layout analysis.
## 2.2 The PaperIndex Structure
We build trees out of documents so LLMs can read them.

# 3. Results
It works very well.
    """


def _mock_sentence_transformer():
    """Create a mock SentenceTransformer that returns deterministic embeddings."""
    mock_model = MagicMock()
    def encode_side_effect(texts):
        np.random.seed(42)
        return np.random.rand(len(texts), 384)
    mock_model.encode.side_effect = encode_side_effect
    return mock_model


# ---------------------------------------------------------------------------
# Test Group 1: Parser block structure
# ---------------------------------------------------------------------------

class TestParserExtraction:
    """Validate the expected structure of text blocks."""

    def test_blocks_have_required_keys(self, sample_blocks):
        required = {"page_no", "block_index", "text", "bbox", "metadata"}
        for block in sample_blocks:
            assert required.issubset(block.keys())

    def test_metadata_has_font_info(self, sample_blocks):
        for block in sample_blocks:
            meta = block["metadata"]
            assert "max_font_size" in meta
            assert "bold_ratio" in meta
            assert isinstance(meta["bold_ratio"], (int, float))

    def test_bbox_has_coordinates(self, sample_blocks):
        for block in sample_blocks:
            bbox = block["bbox"]
            for key in ("x0", "y0", "x1", "y1"):
                assert key in bbox


# ---------------------------------------------------------------------------
# Test Group 2: Chunking
# ---------------------------------------------------------------------------

class TestChunking:
    """Test chunking.chunk_text_blocks."""

    def test_produces_chunks(self, sample_blocks):
        chunks = chunk_text_blocks(sample_blocks, target_size=200, overlap=50)
        assert len(chunks) > 0
        for chunk in chunks:
            assert "text" in chunk
            assert "page_no" in chunk
            assert "metadata" in chunk
            assert len(chunk["text"]) > 0

    def test_empty_input(self):
        assert chunk_text_blocks([]) == []

    def test_single_block(self):
        blocks = [{"page_no": 1, "block_index": 0, "text": "Hello world.",
                    "bbox": {"x0": 0, "y0": 0, "x1": 100, "y1": 20},
                    "metadata": {}}]
        chunks = chunk_text_blocks(blocks, target_size=500, min_chunk_size=5)
        assert len(chunks) == 1
        assert chunks[0]["text"] == "Hello world."

    def test_large_block_is_split(self):
        long_text = "This is a sentence. " * 100  # ~2000 chars
        blocks = [{"page_no": 1, "block_index": 0, "text": long_text,
                    "bbox": {"x0": 0, "y0": 0, "x1": 500, "y1": 500},
                    "metadata": {}}]
        chunks = chunk_text_blocks(blocks, target_size=200, overlap=50, min_chunk_size=50)
        assert len(chunks) > 1

    def test_chunk_metadata_type(self, sample_blocks):
        chunks = chunk_text_blocks(sample_blocks, target_size=200, overlap=50)
        for chunk in chunks:
            assert "chunk_type" in chunk["metadata"]
            assert chunk["metadata"]["chunk_type"] in ("combined", "single", "split")


# ---------------------------------------------------------------------------
# Test Group 3: StructuredDocumentAgent
# ---------------------------------------------------------------------------

class TestStructuredDocumentAgent:
    """Test the section-level agent with mock embeddings."""

    @patch(f"{_PATCH_PREFIX}.SentenceTransformer")
    def test_get_overview_has_sections(self, mock_st_cls, sample_markdown):
        mock_st_cls.return_value = _mock_sentence_transformer()
        agent = StructuredDocumentAgent(sample_markdown, embedding_type="local")

        overview = agent.get_overview()
        assert "Abstract" in overview
        assert "Methodology" in overview
        assert "sec_" in overview

    @patch(f"{_PATCH_PREFIX}.SentenceTransformer")
    def test_read_details_returns_content(self, mock_st_cls, sample_markdown):
        mock_st_cls.return_value = _mock_sentence_transformer()
        agent = StructuredDocumentAgent(sample_markdown, embedding_type="local")

        result = agent.read_details("sec_1")
        assert "SECTION" in result
        assert "Page" in result

    @patch(f"{_PATCH_PREFIX}.SentenceTransformer")
    def test_read_details_invalid_id(self, mock_st_cls, sample_markdown):
        mock_st_cls.return_value = _mock_sentence_transformer()
        agent = StructuredDocumentAgent(sample_markdown, embedding_type="local")

        result = agent.read_details("nonexistent")
        assert "Error" in result

    @patch(f"{_PATCH_PREFIX}.SentenceTransformer")
    def test_search_returns_ranked_results(self, mock_st_cls, sample_markdown):
        mock_st_cls.return_value = _mock_sentence_transformer()
        agent = StructuredDocumentAgent(sample_markdown, embedding_type="local")

        result = agent.search("bounding boxes")
        assert "Search Results" in result
        assert "Score:" in result

    @patch(f"{_PATCH_PREFIX}.SentenceTransformer")
    def test_search_top_k(self, mock_st_cls, sample_markdown):
        mock_st_cls.return_value = _mock_sentence_transformer()
        agent = StructuredDocumentAgent(sample_markdown, embedding_type="local")

        result = agent.search("methodology", top_k=2)
        assert "1." in result
        assert "2." in result


# ---------------------------------------------------------------------------
# Test Group 4: ChunkRAGIndex
# ---------------------------------------------------------------------------

class TestChunkRAGIndex:
    """Test chunk-level RAG index with mock embeddings."""

    @patch(f"{_PATCH_PREFIX}.SentenceTransformer")
    def test_creates_chunks(self, mock_st_cls, sample_blocks):
        mock_st_cls.return_value = _mock_sentence_transformer()
        index = ChunkRAGIndex(sample_blocks, embedding_type="local", target_chunk_size=200)
        assert len(index.chunks) > 0

    @patch(f"{_PATCH_PREFIX}.SentenceTransformer")
    def test_search_returns_list(self, mock_st_cls, sample_blocks):
        mock_st_cls.return_value = _mock_sentence_transformer()
        index = ChunkRAGIndex(sample_blocks, embedding_type="local", target_chunk_size=200)

        results = index.search("novel framework")
        assert isinstance(results, list)
        if results:
            assert "text" in results[0]
            assert "score" in results[0]
            assert "page_no" in results[0]

    @patch(f"{_PATCH_PREFIX}.SentenceTransformer")
    def test_search_formatted_output(self, mock_st_cls, sample_blocks):
        mock_st_cls.return_value = _mock_sentence_transformer()
        index = ChunkRAGIndex(sample_blocks, embedding_type="local", target_chunk_size=200)

        output = index.search_formatted("bounding boxes")
        assert "Chunk Search Results" in output
        assert "Score:" in output

    @patch(f"{_PATCH_PREFIX}.SentenceTransformer")
    def test_empty_blocks(self, mock_st_cls):
        mock_st_cls.return_value = _mock_sentence_transformer()
        index = ChunkRAGIndex([], embedding_type="local", target_chunk_size=200)
        assert index.chunks == []
        assert index.search("anything") == []


# ---------------------------------------------------------------------------
# Test Group 5: Full pipeline (mocked extraction)
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """Test create_full_pipeline_from_pdf with mocked PDF extraction."""

    @patch(f"{_PATCH_PREFIX}.SentenceTransformer")
    @patch(f"{_PATCH_PREFIX}.equations")
    @patch(f"{_PATCH_PREFIX}.tables")
    @patch(f"{_PATCH_PREFIX}.figures")
    @patch(f"{_PATCH_PREFIX}.parser")
    def test_pipeline_returns_all_keys(self, mock_parser, mock_figures, mock_tables,
                                        mock_equations, mock_st_cls, tmp_path):
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_text("fake pdf content")

        mock_parser.extract_text_blocks.return_value = [
            {"page_no": 1, "block_index": 0,
             "text": "This is the abstract of the paper about novel methods.",
             "bbox": {"x0": 72, "y0": 100, "x1": 540, "y1": 120},
             "metadata": {"max_font_size": 16.0, "avg_font_size": 12.0, "bold_ratio": 1.0}},
            {"page_no": 1, "block_index": 1,
             "text": "We present a new approach to document understanding.",
             "bbox": {"x0": 72, "y0": 130, "x1": 540, "y1": 200},
             "metadata": {"max_font_size": 11.0, "avg_font_size": 11.0, "bold_ratio": 0.0}},
        ]
        mock_figures.extract_and_store_paper_figures.return_value = {}
        mock_figures.load_paper_figure_manifest.return_value = {"images": []}
        mock_tables.extract_and_store_paper_tables.return_value = {}
        mock_equations.extract_and_store_paper_equations.return_value = {}
        mock_st_cls.return_value = _mock_sentence_transformer()

        result = create_full_pipeline_from_pdf(str(fake_pdf))

        assert "section_agent" in result
        assert "chunk_index" in result
        assert "markdown" in result
        assert "blocks" in result
        assert "chunks" in result
        assert len(result["blocks"]) == 2

    def test_pipeline_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            create_full_pipeline_from_pdf("/nonexistent/path.pdf")


# ---------------------------------------------------------------------------
# Test Group 6: Shared embedding helpers
# ---------------------------------------------------------------------------

class TestEmbeddingHelpers:
    """Test module-level embedding functions."""

    def test_get_embeddings_empty(self):
        result = get_embeddings([])
        assert len(result) == 0

    def test_get_embeddings_local(self):
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(2, 384)
        result = get_embeddings(["a", "b"], embedding_type="local", local_model=mock_model)
        assert result.shape == (2, 384)
        mock_model.encode.assert_called_once_with(["a", "b"])

    def test_get_embeddings_api_raises_on_failure(self):
        """API mode should raise RuntimeError instead of falling back to mock."""
        with pytest.raises(RuntimeError, match="Embedding API failed"):
            get_embeddings(
                ["test"],
                embedding_type="api",
                base_url="http://localhost:1/v1",
                api_key="EMPTY",
                api_model="test-model",
                max_retries=1,
                timeout=1.0,
            )

    @patch("requests.post")
    def test_get_embeddings_api_batching(self, mock_post):
        """API mode should batch large inputs."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        # Return 2 embeddings per call (batch_size=2)
        mock_response.json.return_value = {
            "data": [
                {"index": 0, "embedding": [0.1] * 8},
                {"index": 1, "embedding": [0.2] * 8},
            ]
        }
        mock_post.return_value = mock_response

        result = get_embeddings(
            ["a", "b", "c", "d"],
            embedding_type="api",
            base_url="http://localhost:8003/v1",
            api_key="EMPTY",
            api_model="test",
            batch_size=2,
        )
        assert result.shape == (4, 8)
        assert mock_post.call_count == 2  # 4 texts / batch_size 2 = 2 calls


# ---------------------------------------------------------------------------
# Test Group 7: Pipeline Registry
# ---------------------------------------------------------------------------

class TestPipelineRegistry:
    """Test the active pipeline registry."""

    def setup_method(self):
        _active_pipelines.clear()

    def teardown_method(self):
        _active_pipelines.clear()

    def test_register_and_retrieve_section_agent(self):
        mock_agent = MagicMock()
        register_pipeline("test_doc", {"section_agent": mock_agent, "chunk_index": None})
        assert get_active_section_agent() is mock_agent

    def test_register_and_retrieve_chunk_index(self):
        mock_index = MagicMock()
        register_pipeline("test_doc", {"section_agent": None, "chunk_index": mock_index})
        assert get_active_chunk_index() is mock_index

    def test_unregister_pipeline(self):
        register_pipeline("test_doc", {"section_agent": MagicMock(), "chunk_index": MagicMock()})
        unregister_pipeline("test_doc")
        assert get_active_section_agent() is None
        assert get_active_chunk_index() is None

    def test_no_active_pipelines(self):
        assert get_active_section_agent() is None
        assert get_active_chunk_index() is None


# ---------------------------------------------------------------------------
# Test Group 8: Optional real PDF tests
# ---------------------------------------------------------------------------

REAL_PDF_PATH = Path("/Developer/AIserver/data/pdftest/NVIDIA-Nemotron-3-Super-Technical-Report.pdf")


@pytest.mark.skipif(not REAL_PDF_PATH.exists(), reason="Real PDF not available")
class TestRealPDF:
    """Integration tests with a real PDF (skipped if PDF not found)."""

    def test_parser_extract_text_blocks(self):
        from BatchAgent.document_agent.parser import extract_text_blocks
        blocks = extract_text_blocks(str(REAL_PDF_PATH))
        assert len(blocks) > 10
        assert blocks[0]["page_no"] >= 1

    @patch(f"{_PATCH_PREFIX}.SentenceTransformer")
    @patch(f"{_PATCH_PREFIX}.equations")
    @patch(f"{_PATCH_PREFIX}.tables")
    @patch(f"{_PATCH_PREFIX}.figures")
    def test_full_pipeline_real_pdf(self, mock_figures, mock_tables,
                                     mock_equations, mock_st_cls):
        """Run full pipeline on real PDF with mocked multi-modal extractors."""
        mock_figures.extract_and_store_paper_figures.return_value = {}
        mock_figures.load_paper_figure_manifest.return_value = {"images": []}
        mock_tables.extract_and_store_paper_tables.return_value = {}
        mock_equations.extract_and_store_paper_equations.return_value = {}
        mock_st_cls.return_value = _mock_sentence_transformer()

        result = create_full_pipeline_from_pdf(str(REAL_PDF_PATH))

        assert len(result["blocks"]) > 10
        assert len(result["chunks"]) > 0
        assert "section_agent" in result

        # Verify search works
        agent = result["section_agent"]
        overview = agent.get_overview()
        assert len(overview) > 50

        chunk_results = result["chunk_index"].search("training methodology")
        assert isinstance(chunk_results, list)
