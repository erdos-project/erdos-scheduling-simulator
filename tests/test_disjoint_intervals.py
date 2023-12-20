import pytest

from utils import DisjointedIntervals


def test_add_non_overlapping_intervals():
    di = DisjointedIntervals()
    di.add((1, 3))
    di.add((5, 7))

    assert di._intervals == [(1, 3), (5, 7)]


def test_add_overlapping_intervals_raises_error():
    di = DisjointedIntervals()
    di.add((3, 5))
    di.add((10, 13))

    with pytest.raises(ValueError) as exc_info:
        di.add((2, 4))
    assert "Overlap detected for (2, 4)" in str(exc_info.value)


def test_overlap_with_existing_intervals():
    di = DisjointedIntervals()
    di.add((3, 5))
    di.add((10, 13))

    assert di.overlap((2, 4)) is True
    assert di.overlap((8, 11)) is True


def test_no_overlap_with_existing_intervals():
    di = DisjointedIntervals()
    di.add((3, 5))
    di.add((10, 13))

    assert di.overlap((1, 2)) is False
    assert di.overlap((6, 9)) is False
    assert di.overlap((14, 20)) is False


def test_overlap_at_boundaries():
    di = DisjointedIntervals()
    di.add((3, 5))
    di.add((10, 13))

    assert di.overlap((1, 3)) is True
    assert di.overlap((5, 7)) is True
    assert di.overlap((8, 10)) is True
    assert di.overlap((13, 15)) is True


def test_no_overlap_empty_intervals():
    di = DisjointedIntervals()

    assert di.overlap((1, 3)) is False
    assert di.overlap((5, 7)) is False


def test_overlap_more_than_one_intervals():
    di = DisjointedIntervals()
    di.add((3, 5))
    di.add((10, 13))

    assert di.overlap((3, 10)) is True
    assert di.overlap((4, 11)) is True
    assert di.overlap((2, 14)) is True


def test_placement_gap_with_left_interval():
    di = DisjointedIntervals()
    di.add((3, 5))
    di.add((10, 13))

    assert di.placement_gap_with_left_interval((1, 2)) == 0
    assert di.placement_gap_with_left_interval((6, 7)) == 1
    assert di.placement_gap_with_left_interval((7, 7)) == 2
    assert di.placement_gap_with_left_interval((8, 9)) == 3
    assert di.placement_gap_with_left_interval((15, 18)) == 2
