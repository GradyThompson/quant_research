"""Load a separately pinned future session for a rolling barrier signal stream."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import exchange_calendars as xcals
import pandas as pd
import pyarrow as pa

from quant_research.core import DatasetId, DatasetVersion
from quant_research.data.schemas import BarFrequency, ohlcv_schema
from quant_research.data.storage import DatasetStore
from quant_research.model.barrier import BarrierModelInput


def load_following_minute_session(
    *,
    store: DatasetStore,
    dataset_id: DatasetId,
    version: DatasetVersion,
    initial: BarrierModelInput,
    session: date,
) -> pa.Table:
    """Validate an immutable future session before feeding bars to the model.

    The returned table retains its original canonical schema. Only bars from
    the specified subsequent exchange session are accepted.
    """
    if initial.decision_at.tzinfo is None or initial.decision_at.utcoffset() != timedelta(0):
        raise ValueError("Initial decision must be UTC-aware")
    calendar = xcals.get_calendar("XNYS")
    target = pd.Timestamp(session)
    if not calendar.is_session(target):
        raise ValueError("Requested date is not an exchange session")
    opening = calendar.session_open(target).to_pydatetime()
    closing = calendar.session_close(target).to_pydatetime()
    if opening <= initial.decision_at:
        raise ValueError("The update session must start after initialization")

    table = store.read(dataset_id, version)
    if table.schema != ohlcv_schema(frequency=BarFrequency.MINUTE):
        raise ValueError("Update dataset does not use the canonical minute schema")
    if table.num_rows == 0:
        raise ValueError("Update session is empty")

    previous_stamp: datetime | None = None
    previous_availability = initial.decision_at
    for row in table.select(["security_id", "timestamp", "available_at"]).to_pylist():
        stamp = row["timestamp"]
        available = row["available_at"]
        if row["security_id"] != str(initial.security_id):
            raise ValueError("Update dataset contains a different security")
        if not isinstance(stamp, datetime) or not isinstance(available, datetime):
            raise ValueError("Null or invalid update timestamp")
        if stamp.tzinfo is None or stamp.utcoffset() != timedelta(0):
            raise ValueError("Update timestamps must be UTC")
        if available.tzinfo is None or available.utcoffset() != timedelta(0):
            raise ValueError("Update availability timestamps must be UTC")
        if not opening < stamp <= closing:
            raise ValueError("Update bar is outside the requested session")
        if previous_stamp is not None and stamp <= previous_stamp:
            raise ValueError("Update timestamps must be strictly increasing")
        if available <= previous_availability or available < stamp:
            raise ValueError("Update availability must be increasing and not precede bar end")
        previous_stamp = stamp
        previous_availability = available
    return table
