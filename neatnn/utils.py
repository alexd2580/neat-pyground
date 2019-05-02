"""Utility module."""
import random


def randbool():
    """Generate a random boolean."""
    return random.random() > 0.5


def chunks(data, chunk_size):
    """Iterate over `data` in chunks of size `chunk_size`.

    The last chunk may be less than `chunk_size` if the length of `data` is not
    evenly divisible by `chunk_size`.
    """
    if chunk_size < 1:
        raise Exception("`chunk_size` may not be less than 1")

    accum = []
    for d in data:
        accum.append(d)
        if len(accum) == chunk_size:
            yield accum
            accum = []

    if accum:
        yield accum
