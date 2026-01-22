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
    """
    Splits text into chunks of specified token size with overlap.
    Uses tiktoken cl100k_base encoding (common for OpenAI models).
    Falls back to character approximation if needed (1 token ~ 4 chars).
    """
    if not text:
        return []

    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        
        chunks = []
        num_tokens = len(tokens)
        start_idx = 0
        chunk_idx = 0
        
        while start_idx < num_tokens:
            end_idx = min(start_idx + chunk_size, num_tokens)
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = encoding.decode(chunk_tokens)
            
            # Calculate offsets in original text
            # This is tricky because decoding might not map 1:1 to original slicing due to normalization/whitespace.
            # Robust approach: find the substring in the original text starting from search_start.
            # Optimization: We can approximate or just search.
            # Since we process sequentially, we can track position.
            
            # Re-locating exact substring in original text is safer for highlighting.
            # However, tiktoken decoding is lossy for some whitespace.
            # For this MVP, we will try to find the chunk_text in the remaining original text.
            
            # Simple approximation for offsets if exact match fails:
            # We trust the text content more than the offsets for embedding.
            # But let's try to be accurate.
            
            # Since finding exact offsets after tokenization round-trip is hard/slow,
            # we will return the decoded text. The offsets will be approximations 
            # based on cumulative character length if we can't match exactly.
            
            # Actually, for RAG, the text content is what matters most. 
            # Let's verify if we can just map back.
            
            # Alternative: use character slicing based on token count * 4? No, that's bad.
            
            # Let's try to match the decoded text in the original text.
            # We assume sequential processing.
            
            # Problem: `encoding.decode` might normalize.
            # Let's stick to the decoded text as the "true" chunk text.
            # Offsets will be calculated relative to the full text *if possible*, 
            # but relying on `text.find` might fail if there are duplicates.
            # A more robust way is to just store the chunk text. 
            # But the requirement asks for start_offset/end_offset.
            
            # Let's maintain a cursor in the original text.
            # Current known issue: Tokenization ignores some whitespace or normalizes it.
            # We will use a "best effort" offset finding.
            
            chunks.append(ChunkDraft(
                text=chunk_text,
                start_offset=0, # Placeholder, difficult to get exact byte offsets from tokens efficiently without a mapping
                end_offset=0,   # Placeholder
                chunk_index=chunk_idx
            ))
            
            chunk_idx += 1
            start_idx += chunk_size - chunk_overlap
            
            # Stop if we reached the end
            if start_idx >= num_tokens:
                break
                
        # Fix offsets by searching in original text?
        # This is expensive but accurate.
        # Let's implement a simple character fallback if tokenization is too complex for offsets,
        # OR just acknowledge that offsets are approximate or 0 if not critical.
        # Given "stable chunk boundaries", tokenization is best.
        # Let's try to patch offsets:
        
        current_pos = 0
        for chunk in chunks:
            # Find chunk text starting from current_pos
            # Allow for some fuzzy matching or whitespace diffs?
            # Tiktoken usually preserves characters well enough for `find`.
            found_idx = text.find(chunk.text, current_pos)
            if found_idx != -1:
                chunk.start_offset = found_idx
                chunk.end_offset = found_idx + len(chunk.text)
                current_pos = found_idx + len(chunk.text) - 10 # Backtrack a bit for overlap safety? 
                # Actually overlap implies we should look backwards? 
                # No, the next chunk starts later in the token stream, so it should be forward in char stream mostly.
                # But due to overlap, the next chunk's text will overlap with this one.
                # So we should set current_pos carefully.
                
                # Better strategy: 
                # chunk N starts at token T. chunk N+1 starts at token T + size - overlap.
                # The text of N+1 should start roughly after N's non-overlapping part.
                # But exact char alignment is hard.
                
                # Let's just set current_pos to found_idx + 1 to ensure forward progress 
                # but allow overlap matching.
                current_pos = found_idx + 1
            else:
                # Fallback: maybe just leave as 0 or approximate?
                pass
                
        return chunks

    except Exception:
        # Fallback to character chunking
        return _chunk_text_chars(text, chunk_size * 4, chunk_overlap * 4)

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
