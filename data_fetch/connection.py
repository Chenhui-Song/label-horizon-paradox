"""Unified data connection: DolphinDB (market data).

Credentials are centralized here; the connection is lazy and closed after use.
The upper layer only touches the ``ddb()`` entry point.

Usage:
    from data_fetch.connection import dc
    s = dc.ddb()                   # DolphinDB session; call dc.close_ddb() when done

Credentials are read from environment variables (``DDB_HOST``, ``DDB_PORT``,
``DDB_USER``, ``DDB_PASSWORD``); set them before running the build scripts.
This module is provided for reference only — the internal DDB instance it was
originally written against is not accessible. Adapt it to your own data source
(see docs/data.md).
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass

import dolphindb as ddb

__all__ = ["DataConnection", "dc"]


@dataclass
class DDBConfig:
    """DolphinDB connection config (read from environment variables)."""

    host: str = os.environ.get("DDB_HOST", "")
    port: int = int(os.environ.get("DDB_PORT", "0"))
    user: str = os.environ.get("DDB_USER", "")
    password: str = os.environ.get("DDB_PASSWORD", "")


class DataConnection:
    """Unified data connection manager (thread-safe, lazy connection).

    DolphinDB: a single reused session; call close_ddb() to release.
    """

    def __init__(self, ddb_cfg: DDBConfig | None = None):
        self.ddb_cfg = ddb_cfg or DDBConfig()
        self._ddb_session: ddb.session | None = None
        self._lock = threading.Lock()

    def ddb(self) -> ddb.session:
        """Get the DolphinDB session (lazy, reused). Thread-safe."""
        if self.ddb_cfg.host == "":
            raise RuntimeError(
                "DolphinDB connection not configured. Set DDB_HOST/DDB_PORT/"
                "DDB_USER/DDB_PASSWORD environment variables, or adapt "
                "data_fetch/connection.py to your own data source."
            )
        if self._ddb_session is None:
            with self._lock:
                if self._ddb_session is None:
                    c = self.ddb_cfg
                    self._ddb_session = ddb.session(c.host, c.port, c.user, c.password)
        return self._ddb_session

    def close_ddb(self) -> None:
        """Close the DolphinDB session."""
        if self._ddb_session is not None:
            try:
                self._ddb_session.close()
            finally:
                self._ddb_session = None

    # ---------- Convenience self-check ----------

    def ping(self) -> dict:
        """Connectivity self-check; returns the DolphinDB status."""
        out = {}
        try:
            s = self.ddb()
            out["ddb"] = {"ok": True, "version": str(s.run("version()"))}
        except Exception as e:
            out["ddb"] = {"ok": False, "error": str(e)}
        return out


# Global default instance; use `from data_fetch.connection import dc`
dc = DataConnection()
