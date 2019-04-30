"""Tests for the `nn` module."""
import math

from hypothesis import given
from hypothesis.strategies import booleans, floats, integers

from neatnn.nn import NN, Gene


class TestGene:
    """Tests for the Gene class."""

    def test_next_index(self):
        """Test that the gene index increases incrementally."""
        assert Gene.next_gene_index() != Gene.next_gene_index()

    @given(start=integers(), end=integers(), weight=floats(allow_nan=False), active=booleans())
    def test_init(self, start, end, weight, active):
        """`__init__` constructs a gene properly."""
        next_index = Gene.next_gene_index()
        gene = Gene(start, end, weight, active)

        assert gene.index == (next_index + 1)
        assert gene.start == start
        assert gene.end == end
        assert gene.weight == weight
        assert gene.active == active

    @given(active=booleans())
    def test_toggle(self, active):
        """`toggle` flips the `active` flag."""
        gene = Gene(0, 0, 0, active)

        gene.toggle()
        assert gene.active == (not active)

        gene.toggle()
        assert gene.active == active

    @given(weight=floats(min_value=-100, max_value=100))
    def test_shift_weight(self, weight):
        """`shift_weight` multiplies the weight by `[-2 .. +2]`."""
        gene = Gene(0, 0, weight, True)
        gene.shift_weight()
        assert -math.fabs(2.0 * weight) <= gene.weight <= math.fabs(2.0 * weight)

    @given(weight=floats())
    def test_randomize_weight(self, weight):
        """`randomize_weight` sets a random weight."""
        gene = Gene(0, 0, weight, True)
        gene.randomize_weight()
        assert -2.0 <= gene.weight <= 2.0


class TestNN:
    """Tests for the `NN` class."""

    @given(floats(min_value=-100.0, max_value=100.0))
    def test_simple_pythonic_evaluator(self, value):
        """Check the pythonic evaluator representation of a NN."""
        NN.setup(1, 1)
        genes = [Gene(0, 1, 1, True)]
        nn = NN(genes)
        pythonic = nn.to_python_function()

        assert (pythonic([value])[0] > 0.5) == (value > 0.0)
