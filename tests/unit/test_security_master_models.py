"""Tests for security master contracts, not identity resolution."""

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta, timezone

import pytest

from quant_research.core import SecurityId
from quant_research.security_master import (
    IdentifierRecord,
    IdentifierType,
    ListingRecord,
    SecurityRecord,
    SecurityType,
    ValidityPeriod,
)

START = datetime(2020, 1, 1, tzinfo=UTC)
END = datetime(2021, 1, 1, tzinfo=UTC)
RECORDED = datetime(2021, 2, 1, tzinfo=UTC)
SECURITY = SecurityId("SEC-000001")


def test_half_open_validity():
    interval = ValidityPeriod(START, END)
    assert interval.contains(START)
    assert interval.contains(END - timedelta(microseconds=1))
    assert not interval.contains(END)


def test_open_ended_validity():
    assert ValidityPeriod(START).contains(datetime(2035, 1, 1, tzinfo=UTC))


@pytest.mark.parametrize("end", [START, START - timedelta(seconds=1)])
def test_invalid_interval(end):
    with pytest.raises(ValueError, match="valid_to"):
        ValidityPeriod(START, end)


def test_naive_time_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        ValidityPeriod(datetime(2020, 1, 1))


def test_non_utc_offset_rejected():
    with pytest.raises(ValueError, match="UTC"):
        ValidityPeriod(datetime(2020, 1, 1, tzinfo=timezone(timedelta(hours=-5))))


def test_security_record():
    record = SecurityRecord(SECURITY, SecurityType.COMMON_STOCK, START, issuer_id="ISS-001")
    assert record.security_id == SECURITY
    with pytest.raises(FrozenInstanceError):
        record.issuer_id = "ISS-002"


def test_security_record_rejects_blank_issuer():
    with pytest.raises(ValueError, match="issuer_id"):
        SecurityRecord(SECURITY, SecurityType.COMMON_STOCK, START, issuer_id=" ")


def test_identifier_record_can_be_backfilled():
    # Later recorded_at than the historical interval is legitimate.
    record = IdentifierRecord(
        SECURITY,
        IdentifierType.ISIN,
        "US0378331005",
        ValidityPeriod(START, END),
        RECORDED,
        "provider_x",
    )
    assert record.validity.contains(START)
    assert not record.validity.contains(END)
    assert record.recorded_at > END


def test_identifier_rejects_blank_value():
    with pytest.raises(ValueError, match="identifier_value"):
        IdentifierRecord(
            SECURITY, IdentifierType.ISIN, " ", ValidityPeriod(START), RECORDED, "provider_x"
        )


def test_listing_ticker_change_can_use_two_periods():
    old = ListingRecord(SECURITY, "OLD", "XNYS", ValidityPeriod(START, END), RECORDED, "provider_x")
    new = ListingRecord(SECURITY, "NEW", "XNYS", ValidityPeriod(END), RECORDED, "provider_x")
    assert old.validity.contains(START)
    assert not old.validity.contains(END)
    assert new.validity.contains(END)


@pytest.mark.parametrize("mic", ["nyse", "XY", "ABCDE", "XN Y", "ÉNYS"])
def test_invalid_mic(mic):
    with pytest.raises(ValueError, match="mic"):
        ListingRecord(SECURITY, "ABC", mic, ValidityPeriod(START), RECORDED, "provider_x")


def test_listings_are_immutable():
    record = ListingRecord(SECURITY, "ABC", "XNYS", ValidityPeriod(START), RECORDED, "provider_x")
    with pytest.raises(FrozenInstanceError):
        record.ticker = "DEF"


def test_recorded_at_is_required_to_be_aware():
    with pytest.raises(ValueError, match="timezone-aware"):
        ListingRecord(
            SECURITY, "ABC", "XNYS", ValidityPeriod(START), datetime(2021, 2, 1), "provider_x"
        )
