"""Simple neural network structure classes."""

import math
import random

import tensorflow as tf


class Gene:
    """Represents a single connection in a neural network."""

    _next_gene_index = 0

    @staticmethod
    def next_gene_index():
        index = Gene._next_gene_index
        Gene._next_gene_index = Gene._next_gene_index + 1
        return index

    def __init__(self, start, end, weight, active, index=None):
        self.index = index if index else Gene.next_gene_index()
        self.start = start
        self.end = end
        self.weight = weight
        self.active = active

    def toggle(self):
        self.active = not self.active

    def shift_weight(self):
        self.weight = self.weight * random.random() * 2

    def randomize_weight(self):
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
        NN._num_inputs = num_in
        NN._num_outputs = num_out

        NN._next_node_index = num_in + num_out

    def __init__(self, links=None):
        self._links = links or []

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

    def to_python_function(self):
        """Generate a pure python evaluator of the network."""
        # Incrementally generate expressions for computing the output nodes.
        nodes = {}

        def default_zero(inputs):
            """Default evaluator, which is used as a placeholder."""
            return 0.0

        def mk_input_access(index):
            """Input node accessor."""

            def input_access(inputs):
                return inputs[index]

            return input_access

        def mk_sigmoid_sum_weighted(links):
            """Compute a weighted sum of the given `links`.

            Uses a sigmoid activation function.
            """
            input_nodes = [(nodes[link.start], link.weight) for link in links]

            def sigmoid_sum_weighted(inputs):
                summed = sum([node(inputs) * weight for node, weight in input_nodes])
                epower = math.e ** summed
                return epower / (1 + epower)

            return sigmoid_sum_weighted

        # Create input nodes.
        for index in range(NN._num_inputs):
            nodes[index] = mk_input_access(index)

        # Create temporary output placeholders (In case noone uses those).
        for index in range(NN._num_outputs):
            nodes[NN._num_inputs + index] = default_zero

        # Sort links by their target.
        # links_by_end :: Map End [Gene]
        links_by_end = {}
        for link in self._links:
            prev = links_by_end.get(link.end) or []
            prev.append(link)
            links_by_end[link.end] = prev

        # Read link definitions from this dict and insert them into nodes.
        # Terminate when all links have been created.
        while links_by_end:
            creatable_end_nodes = {
                index: links
                for index, links in links_by_end.items()
                if all(link.start in nodes for link in links)
            }

            for index, links in creatable_end_nodes.items():
                # `links` is all links required for a certain end node.
                # Filter active links.
                links = [c for c in links if c.active]
                nodes[index] = mk_sigmoid_sum_weighted(links) if links else default_zero
                del links_by_end[index]

        # Gather the outputs.
        output_nodes = [nodes[index + NN._num_inputs] for index in range(NN._num_outputs)]
        return lambda inputs: [node(inputs) for node in output_nodes]
