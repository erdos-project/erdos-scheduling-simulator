import sys

import pytest

from utils import EventTime


def test_time_construction_success():
    """Test that EventTime can be successfully constructed."""

    test_time_ms = EventTime(100, EventTime.Unit.MS)
    assert (
        test_time_ms.time == 100 and test_time_ms.unit == EventTime.Unit.MS
    ), "Incorrect time or unit retrieved."

    test_time_us = EventTime(100, EventTime.Unit.US)
    assert (
        test_time_us.time == 100 and test_time_us.unit == EventTime.Unit.US
    ), "Incorrect time or unit retrieved."

    test_time_s = EventTime(100, EventTime.Unit.S)
    assert (
        test_time_s.time == 100 and test_time_s.unit == EventTime.Unit.S
    ), "Incorrect time or unit retrieved."


def test_time_construction_fail():
    """Test that EventTime fails construction with the wrong unit or wrong time type."""
    with pytest.raises(ValueError):
        test_time_ns = EventTime(100, "ns")

    with pytest.raises(ValueError):
        test_time_ms = EventTime("100.0", EventTime.Unit.MS)


def test_time_unit_conversions():
    """Test that the unit conversions work correctly, and raise errors if we go from
    lower to higher granularity when using the `to` method."""

    # Test microsecond conversions.
    test_time_us = EventTime(200, EventTime.Unit.US)
    with pytest.raises(ValueError):
        test_time_us.to(EventTime.Unit.MS)
    with pytest.raises(ValueError):
        test_time_us.to(EventTime.Unit.S)

    # Test millisecond conversions.
    test_time_ms = EventTime(200, EventTime.Unit.MS)
    test_time_us = test_time_ms.to(EventTime.Unit.US)
    with pytest.raises(ValueError):
        test_time_ms.to(EventTime.Unit.S)
    assert (
        test_time_us.time == int(200 * 1e3) and test_time_us.unit == EventTime.Unit.US
    ), "Incorrect time or unit retrieved."

    # Test second conversions.
    test_time_s = EventTime(200, EventTime.Unit.S)
    test_time_ms = test_time_s.to(EventTime.Unit.MS)
    test_time_us = test_time_s.to(EventTime.Unit.US)
    assert (
        test_time_ms.time == int(200 * 1e3) and test_time_ms.unit == EventTime.Unit.MS
    ), "Incorrect time or unit retrieved."
    assert (
        test_time_us.time == int(200 * 1e6) and test_time_us.unit == EventTime.Unit.US
    ), "Incorrect time or unit retrieved."


def test_time_unchecked_conversions():
    """Test that the unchecked time conversions work correctly."""
    test_time_us = EventTime(200, EventTime.Unit.US)
    test_time_ms, test_time_ms_unit = test_time_us.to_unchecked(EventTime.Unit.MS)
    test_time_s, test_time_s_unit = test_time_us.to_unchecked(EventTime.Unit.S)
    assert (
        abs(test_time_ms - 200 * 1e-3) < sys.float_info.epsilon
        and test_time_ms_unit == EventTime.Unit.MS
    ), "Incorrect time retrieved."
    assert (
        abs(test_time_s - 200 * 1e-6) < sys.float_info.epsilon
        and test_time_s_unit == EventTime.Unit.S
    ), "Incorrect time retrieved."


def test_time_addition():
    """Test that the time addition works correctly."""
    test_time_us = EventTime(200, EventTime.Unit.US)
    test_time_double_us = test_time_us + EventTime(200, EventTime.Unit.US)
    assert (
        test_time_double_us.time == 400
        and test_time_double_us.unit == EventTime.Unit.US
    ), "Incorrect time or unit retrieved."

    test_time_ms = EventTime(200, EventTime.Unit.MS)
    test_time_ms_us = test_time_us + test_time_ms
    assert (
        test_time_ms_us.time == 200 + int(200 * 1e3)
        and test_time_ms_us.unit == EventTime.Unit.US
    ), "Incorrect time or unit retrieved."

    test_time_s = EventTime(200, EventTime.Unit.S)
    test_time_ms_s = test_time_s + test_time_ms
    assert (
        test_time_ms_s.time == 200 + int(200 * 1e3)
        and test_time_ms_s.unit == EventTime.Unit.MS
    ), "Incorrect time or unit retrieved."


def test_time_subtraction():
    """Test that the time subtraction works correctly."""
    test_time_us = EventTime(200, EventTime.Unit.US)
    test_time_zero_us = test_time_us - EventTime(200, EventTime.Unit.US)
    assert (
        test_time_zero_us.time == 0 and test_time_zero_us.unit == EventTime.Unit.US
    ), "Incorrect time or unit retrieved."

    test_time_ms = EventTime(200, EventTime.Unit.MS)
    test_time_ms_us = test_time_ms - test_time_us
    assert (
        test_time_ms_us.time == int(200 * 1e3) - 200
        and test_time_ms_us.unit == EventTime.Unit.US
    ), "Incorrect time or unit retrieved."

    test_time_s = EventTime(200, EventTime.Unit.S)
    test_time_ms_s = test_time_s - test_time_ms
    assert (
        test_time_ms_s.time == int(200 * 1e3) - 200
        and test_time_ms_s.unit == EventTime.Unit.MS
    ), "Incorrect time or unit retrieved."


def test_time_equivalence():
    """Test that EventTime checks correct equivalence."""
    test_time_ms = EventTime(200, EventTime.Unit.MS)
    test_time_us = EventTime(200000, EventTime.Unit.US)
    assert test_time_ms == EventTime(200, EventTime.Unit.MS), "Incorrect equality."
    assert test_time_ms == test_time_us, "Incorrect equality."
    assert test_time_ms != EventTime(200, EventTime.Unit.US), "Incorrect equality."

def test_time_comparisons():
    """Test that the comparison methods work correctly."""
    test_time_us = EventTime(200, EventTime.Unit.US)
    test_time_ms = EventTime(2, EventTime.Unit.MS) 

    assert test_time_us < test_time_ms, "Incorrect < comparison."
    assert test_time_ms > test_time_us, "Incorrect > comparison."
    assert test_time_ms != test_time_us, "Incorrect != comparison."
