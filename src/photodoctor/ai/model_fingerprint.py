from __future__ import annotations

import hashlib
from pathlib import Path


def sampled_file_digest(path: str | Path, block_size: int = 16 * 1024) -> str:
    """Cheap content fingerprint for model/cache invalidation.

    Full SHA-256 verification remains the authority before a model is accepted for
    execution.  This helper only makes metadata caches notice ordinary manual file
    replacement without hashing hundreds of megabytes on every photo open.  Small
    files are read completely; large files sample the beginning, middle and end.
    """
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.blake2b(digest_size=12)
    with p.open("rb") as f:
        if size <= block_size * 3:
            h.update(f.read())
        else:
            offsets = (0, max(0, size // 2 - block_size // 2), max(0, size - block_size))
            for offset in offsets:
                f.seek(offset)
                h.update(f.read(block_size))
    h.update(str(size).encode("ascii"))
    return h.hexdigest()
