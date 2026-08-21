"""File security, MIME type verification, and path traversal protection."""
import hashlib
import os
import re
from typing import List, Set, Optional
from packages.configuration.settings import get_settings

SAFE_FILENAME_REGEX = re.compile(r'^[a-zA-Z0-9_\-\.]+$')

def sanitize_filename(filename: str) -> str:
    """Strip directory paths and unsafe characters from uploaded filename."""
    basename = os.path.basename(filename)
    clean_name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', basename)
    return clean_name or "uploaded_file"

def sanitize_path(base_dir: str, user_path: str) -> str:
    """Prevent path traversal attacks by ensuring resolved path stays within base_dir."""
    abs_base = os.path.abspath(base_dir)
    target = os.path.abspath(os.path.join(base_dir, user_path))
    if not target.startswith(abs_base):
        raise ValueError(f"Path traversal detected: {user_path}")
    return target

def validate_file_extension(filename: str, allowed_extensions: Optional[List[str]] = None) -> bool:
    """Check if file extension is permitted."""
    if allowed_extensions is None:
        allowed_extensions = get_settings().ALLOWED_UPLOAD_EXTENSIONS
    
    ext = os.path.splitext(filename)[1].lower()
    return ext in allowed_extensions

def validate_mime_type(content_sample: bytes, expected_format: str) -> bool:
    """Sniff header bytes to ensure content matches file extension."""
    if expected_format.lower() in [".parquet", "parquet"]:
        # Parquet files begin with magic bytes 'PAR1'
        return content_sample.startswith(b"PAR1")
    elif expected_format.lower() in [".json", "json"]:
        sample = content_sample.strip()
        return sample.startswith((b"{", b"["))
    elif expected_format.lower() in [".csv", "csv"]:
        try:
            decoded = content_sample[:1024].decode('utf-8')
            return ("\n" in decoded) or ("," in decoded) or (";" in decoded)
        except Exception:
            return False
    return True

def calculate_file_sha256(filepath: str) -> str:
    """Compute cryptographic SHA-256 hash of file content."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()
