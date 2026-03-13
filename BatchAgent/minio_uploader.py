"""MinIO file uploader using the minio Python SDK."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional


_client = None
_bucket: str = ""
_endpoint: str = ""


def init_minio(endpoint: str, access_key: str, secret_key: str, bucket: str, secure: bool = False):
    """Initialize the global MinIO client. Call once at startup."""
    global _client, _bucket, _endpoint
    try:
        from minio import Minio
        _endpoint = endpoint.replace("http://", "").replace("https://", "").rstrip("/")
        _client = Minio(_endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        _bucket = bucket
        # Ensure bucket exists
        if not _client.bucket_exists(bucket):
            _client.make_bucket(bucket)
    except Exception as exc:
        print(f"[MinIO] Init failed (uploads disabled): {exc}")
        _client = None


def upload_file(local_path: str, object_name: Optional[str] = None) -> Optional[str]:
    """
    Upload a local file to MinIO. Returns the public HTTP URL or None on failure.
    object_name defaults to the file's basename.
    """
    if _client is None:
        return None
    try:
        name = object_name or Path(local_path).name
        _client.fput_object(_bucket, name, local_path)
        scheme = "https" if getattr(_client, "_secure", False) else "http"
        return f"{scheme}://{_endpoint}/{_bucket}/{name}"
    except Exception as exc:
        print(f"[MinIO] Upload failed for {local_path}: {exc}")
        return None
