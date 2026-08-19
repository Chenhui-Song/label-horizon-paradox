"""Local parquet cache utilities.

Cache key by (table, date range); on a hit, read locally directly to avoid
repeated DDB queries.

The cache is **shared read infrastructure** (the same market/fundamental data
reused across experiments) and defaults to the in-package ``data/cache/``
directory. If you need to move the cache outside the package (e.g. read-only
install, CI environment), set the environment variable ``BT_CACHE_DIR`` to
override the directory.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd

__all__ = ["cache_path", "read_cache", "write_cache", "cache_key", "cache_root"]


def cache_root() -> Path:
    """Cache root directory.

    Prefers the environment variable ``BT_CACHE_DIR``; otherwise defaults to
    the in-package ``data/cache/``.
    """
    env = os.environ.get("BT_CACHE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parent / "cache"


def cache_key(*parts) -> str:
    """Generate a cache key (for the filename), joining arbitrary args into a stable short string."""
    raw = "|".join(str(p) for p in parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def cache_path(name: str, key: str, ext: str = "parquet") -> Path:
    """Return the cache file path. name is a business name (e.g. calendar), key is from cache_key()."""
    return cache_root() / f"{name}_{key}.{ext}"


def read_cache(name: str, key: str) -> pd.DataFrame | None:
    """Read the cache; returns None if it does not exist."""
    p = cache_path(name, key)
    if p.exists():
        return pd.read_parquet(p)
    return None


def write_cache(name: str, key: str, df: pd.DataFrame) -> None:
    """Write the cache (snappy compression)."""
    p = cache_path(name, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, compression="snappy")
