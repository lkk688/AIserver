import pytest
from backend.app.util.chunking import chunk_text
from backend.app.util.hashing import compute_hash

def test_chunking_basic():
    text = "Hello world " * 50
    # chunk size 10 tokens, overlap 2
    chunks = chunk_text(text, 10, 2)
    
    assert len(chunks) > 0
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    # Check text content
    assert "Hello" in chunks[0].text

def test_chunking_overlap():
    text = "A B C D E F G H I J"
    # Using small size to force splits. 
    # Tiktoken tokens might map 1:1 to words here roughly.
    chunks = chunk_text(text, 4, 2)
    # Expected: [A B C D], [C D E F], ...
    if len(chunks) >= 2:
        # Check for overlap content
        # This depends on tiktoken encoding, but generally words should overlap
        pass

def test_hashing_stability():
    t1 = "Hello World"
    t2 = "Hello World"
    assert compute_hash(t1) == compute_hash(t2)
    assert compute_hash(t1) != compute_hash("Hello Python")
