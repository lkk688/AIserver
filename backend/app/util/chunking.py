import hashlib
import tiktoken
from dataclasses import dataclass
from typing import List

@dataclass
class ChunkDraft:
    text: str
    start_offset: int
    end_offset: int
    chunk_index: int
    
def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[ChunkDraft]:
    if not text:
        return []

    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        return _chunk_text_chars(text, chunk_size * 4, chunk_overlap * 4)

    paragraphs = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        if text[i:i+2] == "\n\n":
            para = text[start:i]
            paragraphs.append((para, start, i))
            i += 2
            start = i
        else:
            i += 1
    if start < n:
        paragraphs.append((text[start:n], start, n))

    chunks = []
    chunk_idx = 0
    current_paragraphs: List[str] = []
    current_start: int | None = None
    current_end: int | None = None
    current_tokens = 0

    def flush():
        nonlocal current_paragraphs, current_start, current_end, chunk_idx
        if not current_paragraphs or current_start is None or current_end is None:
            return
        joined = "\n\n".join(p for p in current_paragraphs if p)
        chunks.append(
            ChunkDraft(
                text=joined,
                start_offset=current_start,
                end_offset=current_end,
                chunk_index=chunk_idx,
            )
        )
        chunk_idx += 1
        current_paragraphs = []
        current_start = None
        current_end = None

    for para, para_start, para_end in paragraphs:
        trimmed = para.strip()
        if not trimmed:
            continue
        tokens_len = len(encoding.encode(trimmed))
        if tokens_len > chunk_size and not current_paragraphs:
            inner_chunks = _chunk_text_chars(trimmed, chunk_size * 4, chunk_overlap * 4)
            for inner in inner_chunks:
                chunks.append(
                    ChunkDraft(
                        text=inner.text,
                        start_offset=para_start + inner.start_offset,
                        end_offset=para_start + inner.end_offset,
                        chunk_index=chunk_idx,
                    )
                )
                chunk_idx += 1
            continue

        if current_tokens + tokens_len > chunk_size and current_paragraphs:
            flush()
            current_tokens = 0

        if current_start is None:
            current_start = para_start
        current_end = para_end
        current_paragraphs.append(trimmed)
        current_tokens += tokens_len

    flush()
    return chunks

def _chunk_text_chars(text: str, chunk_size_chars: int, chunk_overlap_chars: int) -> List[ChunkDraft]:
    chunks = []
    start = 0
    idx = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size_chars, text_len)
        chunk_text = text[start:end]
        
        chunks.append(ChunkDraft(
            text=chunk_text,
            start_offset=start,
            end_offset=end,
            chunk_index=idx
        ))
        
        idx += 1
        start += chunk_size_chars - chunk_overlap_chars
        
        if start >= text_len:
            break
            
    return chunks
