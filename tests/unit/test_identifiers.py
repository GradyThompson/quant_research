from dataclasses import FrozenInstanceError
from uuid import uuid4

import pytest

from quant_research.core import (
    DatasetId,
    DatasetVersion,
    SecurityId,
)


def test_valid_security_id():
    security_id = SecurityId("SEC-000001")

    assert security_id.value == "SEC-000001"
    assert str(security_id) == "SEC-000001"


@pytest.mark.parametrize(
    "value",
    [
        "",
        "AAPL",
        "SEC-123",
        "SEC-ABCDEF",
        "sec-000001",
        "SEC-000001 ",
    ],
)
def test_invalid_security_id(value):
    with pytest.raises(ValueError):
        SecurityId(value)


def test_security_id_is_immutable():
    security_id = SecurityId("SEC-000001")

    with pytest.raises(FrozenInstanceError):
        security_id.value = "SEC-000002"


def test_security_id_value_equality():
    first = SecurityId("SEC-000001")
    second = SecurityId("SEC-000001")

    assert first == second
    assert hash(first) == hash(second)


def test_valid_dataset_id():
    dataset = DatasetId("market.ohlcv.daily")

    assert str(dataset) == "market.ohlcv.daily"


@pytest.mark.parametrize(
    "value",
    [
        "",
        "ohlcv",
        "Market.OHLCV",
        ".market.ohlcv",
        "market.",
        "market..daily",
        "market/ohlcv",
    ],
)
def test_invalid_dataset_id(value):
    with pytest.raises(ValueError):
        DatasetId(value)


def test_dataset_version():
    value = uuid4()
    version = DatasetVersion(value)

    assert version.value == value
    assert str(version) == str(value)


def test_dataset_versions_are_distinct():
    first = DatasetVersion(uuid4())
    second = DatasetVersion(uuid4())

    assert first != second
