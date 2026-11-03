"""Offline tests for deterministic minute pilot planning."""

from datetime import date

import pytest

from quant_research.security_master.minute_pilot import minute_session_plan, select_minute_pilot


def member(ticker: str, volume: float, mic: str = "XNAS") -> dict[str, object]:
    return {
        "security_id": f"SEC-{ticker}",
        "ticker": ticker,
        "mic": mic,
        "mean_dollar_volume": volume,
    }


def test_select_ranks_independently_of_input_order() -> None:
    items = [member("BBB", 20), member("AAA", 20), member("CCC", 30)]
    first, excluded = select_minute_pilot(items, count=2)
    second, _ = select_minute_pilot(list(reversed(items)), count=2)
    assert [item["ticker"] for item in first] == ["CCC", "AAA"]
    assert first == second
    assert excluded == {"unsupported_mic": 0, "unsupported_ticker": 0}


def test_select_filters_adapter_incompatible_names() -> None:
    items = [member("BRK.B", 100), member("AAA", 50), member("BBB", 200, "ARCX")]
    selected, excluded = select_minute_pilot(items, count=1)
    assert selected[0]["ticker"] == "AAA"
    assert excluded == {"unsupported_mic": 1, "unsupported_ticker": 1}


def test_select_refuses_duplicate_and_insufficient_candidates() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        select_minute_pilot([member("AAA", 10), member("AAA", 20)], count=1)
    with pytest.raises(ValueError, match="Only 1"):
        select_minute_pilot([member("AAA", 10)], count=2)
    with pytest.raises(ValueError, match="count"):
        select_minute_pilot([member("AAA", 10)], count=0)


@pytest.mark.parametrize("volume", [float("nan"), float("inf"), -1.0, True])
def test_invalid_volume_is_rejected(volume: float) -> None:
    with pytest.raises(ValueError, match="mean_dollar_volume"):
        select_minute_pilot([member("AAA", volume)], count=1)


def test_minute_session_plan_excludes_holiday_and_weekend() -> None:
    assert minute_session_plan(evaluation_date=date(2025, 1, 13)) == (
        date(2025, 1, 3),
        date(2025, 1, 6),
        date(2025, 1, 7),
        date(2025, 1, 8),
        date(2025, 1, 10),
        date(2025, 1, 13),
    )


def test_minute_session_plan_rejects_closed_day() -> None:
    with pytest.raises(ValueError, match="not an exchange session"):
        minute_session_plan(evaluation_date=date(2025, 1, 11))
