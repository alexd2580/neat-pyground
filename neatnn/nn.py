"""Simple neural network structure classes."""

import logging
import math
import random

import tensorflow as tf

logger = logging.getLogger(__name__)


class Gene:
    """Represents a single connection in a neural network."""

    _next_gene_index = 0
    _genes = []

    _MAX_PERTURB_STEP = 0.1

    @staticmethod
    def next_gene_index():
        """Sequentially increasing integer."""
        index = Gene._next_gene_index
        Gene._next_gene_index = Gene._next_gene_index + 1
        return index

    def __init__(self, start, end, active, weight=None, index=None):
        """Create a new `Gene` and add it to the global registry."""
        self.index = index if index is not None else Gene.next_gene_index()
        self.start = start
        self.end = end
        self.weight = weight if weight is not None else random.random() * 4 - 2
        self.active = active

        if index is None:
            Gene._genes.append(self)

    def clone(self):
        """Create a copy of this gene."""
        return Gene(self.start, self.end, self.active, weight=self.weight, index=self.index)

    def __repr__(self):
        """Generate string representation."""
        link = f"{self.start}-{self.end}"
        active = "on" if self.active else "off"
        return f"Gene({self.index}: {link}, {self.weight:.2f}, {active})"

    @staticmethod
    def exists(start, end):
        """Check whether a link between `start` and `end` already exists."""
        return any(gene.start == start and gene.end == end for gene in Gene._genes)

    def toggle(self):
        """Flip the `active` flag."""
        self.active = not self.active

    def shift_weight(self):
        """Shift the weight by a random value."""
        self.weight = self.weight + (random.random() - 0.5) * 2 * Gene._MAX_PERTURB_STEP

    def randomize_weight(self):
        """Set the weight to a random value."""
        self.weight = random.uniform(-2, 2)


class NN:
    """Represents a neural network built of `Gene`s."""

    _num_inputs = None
    _num_outputs = None

    _next_node_index = None

    @staticmethod
    @property
    def next_node_index():
        index = NN._next_node_index
        NN._next_node_index = Gene._next_node_index + 1
        return index

    @staticmethod
    def setup(num_in, num_out):
        NN._num_inputs = num_in + 1
        NN._num_outputs = num_out

        NN._next_node_index = NN._num_inputs + NN._num_outputs

    def __init__(self, links=None):
        self._links = [link.clone() for link in (links or [])]

    def to_tensorflow_network(self):
        nodes = {}

        # Create input nodes.
        for index in range(NN._num_inputs):
            nodes[index] = tf.Variable(0.0, name=f"input{index}")

        # Create temporary placeholders.
        for index in range(NN._num_outputs):
            nodes[NN._num_inputs + index] = tf.Variable(0.0, name=f"out{index}")

        # Sort links by their target.
        links_by_end = {}
        for link in self._links:
            prev = links_by_end.get(link.end) or []
            prev.append(link)
            links_by_end[link.end] = prev

        while links_by_end:
            creatable_end_nodes = {
                index: links
                for index, links in links_by_end.items()
                if all(start in nodes for _, start, _, _, _ in links)
            }

            for index, links in creatable_end_nodes.items():
                # `links` is all links required for a certain end node.
                # Filter active links.
                links = [c for c in links if c.active]

                end_node = tf.constant(0.0)
                if links:
                    for _, start, _, _, weight in links:
                        end_node = tf.add(end_node, tf.multiply(nodes[start], weight))
                    end_node = tf.nn.sigmoid(end_node)

                nodes[index] = end_node
                del links_by_end[index]

        inputs = [nodes[index] for index in range(NN._num_inputs)]
        outputs = [nodes[index + NN._num_inputs] for index in range(NN._num_outputs)]
        return inputs, outputs


class RecurrentEvaluator:
    """A recurrent evaluator."""

    def __init__(self, nn):
        """Generate a pure python evaluator of the network."""
        self._inputs = {}
        self._links = {}
        self._outputs = {}

        self._num_inputs = nn._num_inputs
        self._num_outputs = nn._num_outputs

        # Initialize output values.
        for index in range(nn._num_outputs):
            self._inputs[nn._num_inputs + index] = 0.0
            self._outputs[nn._num_inputs + index] = 0.0

        # Initialize node values.
        for gene in nn._links:
            self._inputs[gene.start] = 0.0
            self._outputs[gene.start] = 0.0

        # Sort links by their target.
        # self._links :: Map End [Gene]
        for link in nn._links:
            if link.active:
                prev = self._links.get(link.end) or []
                prev.append(link)
                self._links[link.end] = prev

    def __call__(self, inputs):
        """Evaluate the network."""
        # Swap the buffers.
        self._inputs, self._outputs = self._outputs, self._inputs

        # Copy inputs over.
        for i in range(self._num_inputs - 1):
            self._inputs[i] = inputs[i]

        # Global BIAS input node.
        self._inputs[self._num_inputs - 1] = 1.0

        for end, genes in self._links.items():
            weighted = [self._inputs[gene.start] * gene.weight for gene in genes]
            epower = math.e ** (5 * sum(weighted))
            self._outputs[end] = epower / (1 + epower)

        return (self._outputs[self._num_inputs + i] for i in range(self._num_outputs))
