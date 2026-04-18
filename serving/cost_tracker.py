"""
SQLite-based cost tracker for the tweet scoring serving layer.

Tracks per-request latency, composite score, and estimated AWS g5.xlarge cost.

Cost formula: g5.xlarge on-demand rate ($1.006/hr) pro-rated by inference time.
  estimated_cost_usd = latency_ms / 1000 * (1.006 / 3600)
"""

import sqlite3
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path("data/cost_tracker.db")

# AWS g5.xlarge on-demand rate (USD/hr) as of 2024
_G5_XLARGE_HOURLY = 1.006

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS requests (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ts               TEXT    NOT NULL,
    latency_ms       REAL    NOT NULL,
    composite_score  REAL    NOT NULL,
    model_version    TEXT    NOT NULL,
    estimated_cost_usd REAL  NOT NULL
);
"""


def _get_conn() -> sqlite3.Connection:
    """Open (and initialise) the SQLite database, creating parent dirs if needed."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(CREATE_TABLE_SQL)
    conn.commit()
    return conn


def log_request(
    latency_ms: float,
    composite_score: float,
    model_version: str = "tweet-scorer",
) -> None:
    """Insert a single inference record into the cost tracker database."""
    estimated_cost_usd = latency_ms / 1000 * (_G5_XLARGE_HOURLY / 3600)
    ts = datetime.now(tz=timezone.utc).isoformat()
    try:
        conn = _get_conn()
        conn.execute(
            """
            INSERT INTO requests (ts, latency_ms, composite_score, model_version, estimated_cost_usd)
            VALUES (?, ?, ?, ?, ?)
            """,
            (ts, latency_ms, composite_score, model_version, estimated_cost_usd),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        # Cost tracking is best-effort — never let a DB error surface to the caller.
        logger.error("cost_tracker.log_request failed: %r", exc)


def weekly_summary() -> None:
    """Print a 7-day cost and quality summary to stdout."""
    since = (datetime.now(tz=timezone.utc) - timedelta(days=7)).isoformat()
    try:
        conn = _get_conn()
        row = conn.execute(
            """
            SELECT
                COUNT(*)                   AS total_requests,
                AVG(latency_ms)            AS avg_latency_ms,
                SUM(estimated_cost_usd)    AS total_cost_usd,
                AVG(composite_score)       AS avg_composite_score
            FROM requests
            WHERE ts >= ?
            """,
            (since,),
        ).fetchone()
        conn.close()

        total_requests, avg_latency_ms, total_cost_usd, avg_composite_score = row
        print("=== Weekly Summary (last 7 days) ===")
        print(f"  Total requests    : {total_requests or 0}")
        print(f"  Avg latency       : {avg_latency_ms or 0:.1f} ms")
        print(f"  Total cost (est.) : ${total_cost_usd or 0:.6f}")
        print(f"  Avg composite     : {avg_composite_score or 0:.2f}")
    except Exception as exc:
        logger.error("cost_tracker.weekly_summary failed: %r", exc)
