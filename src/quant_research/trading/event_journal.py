"""Durable local SQLite journal for order lifecycle events.

Single-writer transactions and full replay deliberately favor correctness over
throughput. This is not a broker gateway, reconciliation system, or exactly-once
execution guarantee. Never infer live broker state solely from this journal.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from quant_research.trading.order_lifecycle import (
    OrderEvent,
    OrderEventKind,
    OrderState,
    apply_order_event,
)


def _encode(event: OrderEvent) -> str:
    return json.dumps(
        {
            "event_id": event.event_id,
            "order_id": event.order_id,
            "occurred_at": event.occurred_at.isoformat(),
            "kind": event.kind.value,
            "fill_quantity": event.fill_quantity,
            "fill_price": str(event.fill_price) if event.fill_price is not None else None,
            "broker_execution_id": event.broker_execution_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _decode(payload: str) -> OrderEvent:
    data = json.loads(payload)
    return OrderEvent(
        event_id=data["event_id"],
        order_id=data["order_id"],
        occurred_at=datetime.fromisoformat(data["occurred_at"]),
        kind=OrderEventKind(data["kind"]),
        fill_quantity=data["fill_quantity"],
        fill_price=Decimal(data["fill_price"]) if data["fill_price"] is not None else None,
        broker_execution_id=data["broker_execution_id"],
    )


class OrderEventJournal:
    """Local durable order journal. Call `close()` when finished.

    One connection per instance. SQLite serializes writers across processes.
    Operations are transactional; state is reconstructed from journal rows.
    """

    def __init__(self, path: str | Path) -> None:
        db_path = Path(path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, isolation_level=None, timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                requested_quantity INTEGER NOT NULL CHECK(requested_quantity > 0)
            );
            CREATE TABLE IF NOT EXISTS order_events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                order_id TEXT NOT NULL REFERENCES orders(order_id),
                broker_execution_id TEXT UNIQUE,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_events_order
                ON order_events(order_id, sequence);
            """
        )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> OrderEventJournal:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def create_order(self, order_id: str, requested_quantity: int) -> OrderState:
        initial = OrderState(order_id=order_id, requested_quantity=requested_quantity)
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            existing = self._conn.execute(
                "SELECT requested_quantity FROM orders WHERE order_id=?", (order_id,)
            ).fetchone()
            if existing is None:
                self._conn.execute(
                    "INSERT INTO orders(order_id, requested_quantity) VALUES (?, ?)",
                    (order_id, requested_quantity),
                )
            elif existing[0] != requested_quantity:
                raise ValueError("conflicting order definition")
            self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise
        return initial if existing is None else self.load_order(order_id)

    def load_order(self, order_id: str) -> OrderState:
        row = self._conn.execute(
            "SELECT requested_quantity FROM orders WHERE order_id=?", (order_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown order: {order_id}")
        state = OrderState(order_id=order_id, requested_quantity=int(row[0]))
        events = self._conn.execute(
            "SELECT payload FROM order_events WHERE order_id=? ORDER BY sequence", (order_id,)
        )
        for (payload,) in events:
            state = apply_order_event(state, _decode(payload))
        return state

    def append(self, event: OrderEvent) -> OrderState:
        """Atomically validate and record an event, then return rebuilt state.

        Repeated event_id with identical content is idempotent. Conflicting
        event IDs and duplicate broker execution IDs are rejected, not ignored.
        """
        payload = _encode(event)
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute(
                "SELECT payload FROM order_events WHERE event_id=?", (event.event_id,)
            ).fetchone()
            if row is not None:
                if row[0] != payload:
                    raise ValueError("conflicting event_id payload")
                state = self.load_order(event.order_id)
                self._conn.commit()
                return state
            if event.broker_execution_id is not None:
                seen = self._conn.execute(
                    "SELECT event_id FROM order_events WHERE broker_execution_id=?",
                    (event.broker_execution_id,),
                ).fetchone()
                if seen is not None:
                    raise ValueError("duplicate broker_execution_id")
            state = apply_order_event(self.load_order(event.order_id), event)
            self._conn.execute(
                "INSERT INTO order_events(event_id, order_id, broker_execution_id, payload) "
                "VALUES (?, ?, ?, ?)",
                (event.event_id, event.order_id, event.broker_execution_id, payload),
            )
            self._conn.commit()
            return state
        except BaseException:
            self._conn.rollback()
            raise

    def orders(self) -> tuple[OrderState, ...]:
        ids = self._conn.execute("SELECT order_id FROM orders ORDER BY order_id").fetchall()
        return tuple(self.load_order(str(order_id)) for (order_id,) in ids)
