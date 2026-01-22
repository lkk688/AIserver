import hashlib

def compute_hash(text: str) -> str:
    """
    Computes a stable hash of the text using SHA-256.
    Returns the hex digest.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
