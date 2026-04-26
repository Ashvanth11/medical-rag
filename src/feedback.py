import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "feedback.db"


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            ts            TEXT    NOT NULL,
            turn_id       TEXT    NOT NULL UNIQUE,
            session_title TEXT,
            question      TEXT    NOT NULL,
            answer        TEXT    NOT NULL,
            rating        INTEGER NOT NULL   -- 1 = thumbs up, -1 = thumbs down
        )
    """)
    # migrate existing rows that predate the session_title column
    cols = {row[1] for row in conn.execute("PRAGMA table_info(feedback)")}
    if "session_title" not in cols:
        conn.execute("ALTER TABLE feedback ADD COLUMN session_title TEXT")
    conn.commit()
    return conn


def log_feedback(
    turn_id: str,
    question: str,
    answer: str,
    rating: int,
    session_title: Optional[str] = None,
) -> None:
    with _get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO feedback (ts, turn_id, session_title, question, answer, rating) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), turn_id, session_title, question, answer, rating),
        )
