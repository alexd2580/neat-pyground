"""Tests for the `nn` module."""
from neatnn.nn import NN, Gene


class TestNN:
    """Tests for the `NN` class."""

    def test_pythonic_evaluator(self):
        """Check the pythonic evaluator representation of a NN."""
        NN.setup(1, 1)
        genes = [Gene(0, 1, 1, True)]
        nn = NN(genes)
        pythonic = nn.to_python_function()

        assert pythonic([4]) == 4
