"""Operational data written during conversations: tickets and failed order lookups.

It lives apart from the store database, which ingestion replaces, and from the tools'
read-only connections. In production this would be a CRM integration.
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app.db import connect_writable

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id INTEGER PRIMARY KEY AUTOINCREMENT,
    protocol TEXT UNIQUE,
    thread_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    summary TEXT NOT NULL,
    contact TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tickets_thread ON tickets (thread_id, reason);
CREATE TABLE IF NOT EXISTS order_lookup_failures (
    thread_id TEXT PRIMARY KEY,
    failures INTEGER NOT NULL
);
"""


@dataclass(frozen=True)
class Ticket:
    protocol: str
    reason: str
    summary: str
    contact: str | None


class RuntimeStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        with connect_writable(path) as connection:
            connection.executescript(SCHEMA)

    def open_ticket(
        self,
        *,
        thread_id: str,
        reason: str,
        summary: str,
        contact: str | None,
        now: datetime,
    ) -> Ticket:
        """Create a ticket, or return the conversation's open one for the same reason.

        A regenerated answer may call the tool again, which must not duplicate the ticket. A
        contact given later is saved, because the customer is told the team will use it.
        """
        with connect_writable(self._path) as connection:
            existing = connection.execute(
                "SELECT protocol, reason, summary, contact FROM tickets"
                " WHERE thread_id = ? AND reason = ?",
                (thread_id, reason),
            ).fetchone()
            if existing:
                if contact and contact != existing["contact"]:
                    connection.execute(
                        "UPDATE tickets SET contact = ? WHERE protocol = ?",
                        (contact, existing["protocol"]),
                    )
                return Ticket(
                    protocol=existing["protocol"],
                    reason=existing["reason"],
                    summary=existing["summary"],
                    contact=contact or existing["contact"],
                )
            cursor = connection.execute(
                "INSERT INTO tickets (thread_id, reason, summary, contact, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (thread_id, reason, summary, contact, now.isoformat()),
            )
            protocol = f"EM-{now:%Y%m%d}-{cursor.lastrowid:04d}"
            connection.execute(
                "UPDATE tickets SET protocol = ? WHERE ticket_id = ?",
                (protocol, cursor.lastrowid),
            )
        return Ticket(protocol=protocol, reason=reason, summary=summary, contact=contact)

    def list_tickets(self) -> list[sqlite3.Row]:
        with connect_writable(self._path) as connection:
            return connection.execute("SELECT * FROM tickets ORDER BY ticket_id").fetchall()

    def failed_lookups(self, thread_id: str) -> int:
        with connect_writable(self._path) as connection:
            row = connection.execute(
                "SELECT failures FROM order_lookup_failures WHERE thread_id = ?", (thread_id,)
            ).fetchone()
        return row["failures"] if row else 0

    def record_failed_lookup(self, thread_id: str) -> None:
        with connect_writable(self._path) as connection:
            connection.execute(
                """
                INSERT INTO order_lookup_failures (thread_id, failures) VALUES (?, 1)
                ON CONFLICT (thread_id) DO UPDATE SET failures = failures + 1
                """,
                (thread_id,),
            )
