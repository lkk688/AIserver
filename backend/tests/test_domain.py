import pytest
from uuid import uuid4
from datetime import datetime
from backend.app.domain.models import Source, Document, Chunk, Job, JobType, JobStatus

def test_source_model():
    s = Source(name="test", path="/tmp")
    assert s.id is not None
    assert isinstance(s.created_at, datetime)
    assert s.name == "test"

def test_document_model():
    doc = Document(
        source_id=uuid4(),
        uri="file:///tmp/test.txt",
        title="Test Doc"
    )
    assert doc.status == "new"
    assert doc.title == "Test Doc"

def test_chunk_model():
    chunk = Chunk(
        doc_id=uuid4(),
        chunk_index=0,
        text="hello world",
        start_offset=0,
        end_offset=11,
        chunk_hash="abc"
    )
    assert chunk.text == "hello world"

def test_job_model():
    job = Job(type=JobType.SCAN_SOURCE)
    assert job.status == JobStatus.PENDING
    assert job.progress == 0.0

def test_job_enum_validation():
    with pytest.raises(ValueError):
        Job(type="invalid_type")
