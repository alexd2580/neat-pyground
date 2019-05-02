"""Tests for the `utils` module."""
import pytest
from hypothesis import given
from hypothesis.strategies import integers, lists

from neatnn.utils import chunks, randbool


def test_randbool():
    """`randbool` generates random booleans.

    The probability of generating a particular sequence of length 20
    is equal to `9.5367431640625e-07`. Therefore in this test we expect
    the function to generate both `True` and `False`.
    """
    randoms = [randbool() for _ in range(20)]
    assert any(randoms)
    assert not all(randoms)


@given(data=lists(integers()), chunk_size=integers(min_value=1, max_value=5))
def test_chunks(data, chunk_size):
    """`chunks` iterates over all elements in chunks of size `chunk_size`."""
    data_iter = iter(data)
    chunked_iter = chunks(data, chunk_size)

    last_chunk_had_full_size = True
    for chunk in chunked_iter:
        # All chunks except the last one should be full size.
        assert last_chunk_had_full_size
        last_chunk_had_full_size = len(chunk) == chunk_size

        for value in chunk:
            assert next(data_iter, "Unexpected") == value

    assert next(data_iter, "End") == "End"


def test_chunks_too_small():
    """`chunks` raises an exception if the chunk_size is less than 1."""
    with pytest.raises(Exception):
        next(chunks([], 0))
