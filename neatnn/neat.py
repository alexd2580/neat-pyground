import functools
import math
import random
import sys

import numpy as np
import pygame
import tensorflow as tf

from neatnn import NN, Gene


class NeatNN(NN)
    _delta_weights_factor = 0.0
    _delta_disjoint_factor = 1.0
    _delta_excess_factor = 1.0
    _species_delta_treshold = 6

    thanos = 0.5

    @staticmethod
    def align_genes(a, b):
        a_iter = iter(a._connections)
        b_iter = iter(b._connections)

        a, b = next(a_iter, None), next(b_iter, None)
        while a or b:
            if a and b and a[0] == b[0]:
                yield (a, b)
                a, b = next(a_iter, None), next(b_iter, None)
            elif b and (not a or a[0] > b[0]):
                yield (None, b)
                b = next(b_iter, None)
            elif a and (not b or a[0] < b[0]):
                yield (a, None)
                a = next(a_iter, None)

    @staticmethod
    def delta(a, b):
        matching_weight_delta = 0.0
        num_disjoint = 0
        num_excess = 0

        aligned_genes = NN.align_genes(a, b)
        a, b = next(aligned_genes, (None, None))
        while a and b:
            matching_weight_delta = matching_weight_delta + math.fabs(a[4] - b[4])
            a, b = next(aligned_genes, (None, None))

        had_a = True
        while a or b:
            num_excess = num_excess + 1
            if not had_a and a:
                # These are disjoint therefore added to the gene pool.
                num_disjoint = num_disjoint + num_excess
                num_excess = 0
            had_a = bool(a)
            a, b = next(aligned_genes, (None, None))

        return (
            matching_weight_delta * NN._delta_weights_factor
            + num_disjoint * NN._delta_disjoint_factor
            + num_excess * NN._delta_excess_factor
        )

    @staticmethod
    def is_same_species(a, b):
        return NN.delta(a, b) < NN._species_delta_treshold

    @staticmethod
    def breed(a, b):
        new_connections = []

        aligned_genes = NN.align_genes(a, b)
        a, b = next(aligned_genes, (None, None))
        while a and b:
            new_connections.append(a if random_bool() else b)
            a, b = next(aligned_genes, (None, None))

        excess_genes = []
        had_a = True
        while a or b:
            excess_genes.append(a or b)
            if not had_a and a:
                # These are disjoint therefore added to the gene pool.
                new_connections.extend(excess_genes)
                excess_genes = []
            had_a = bool(a)
            a, b = next(aligned_genes, (None, None))

        return NN(connections=new_connections)

    @staticmethod
    def add_link(connections):
        start_nodes = list(set([*[start for _, start, _, _, _ in connections], *range(NN._num_in)]))
        end_nodes = list(
            set(
                [
                    *[end for _, _, end, _, _ in connections],
                    *[index + NN._num_in for index in range(NN._num_out)],
                ]
            )
        )

        gene_index = NN._next_gene_index
        NN._next_gene_index = NN._next_gene_index + 1
        weight = random.random() * 4 - 2

        # Repeat until not cyclic.
        def is_parent_of(a, b):
            if a == b:
                return True
            return any([is_parent_of(end, b) for _, start, end, _, _ in connections if start == a])

        start_node, end_node = 0, 0
        while is_parent_of(end_node, start_node):
            start_node = start_nodes[random_int(len(start_nodes))]
            end_node = end_nodes[random_int(len(end_nodes))]

        connections.append((gene_index, start_node, end_node, True, weight))

    @staticmethod
    def split_link(connections):
        if len(connections) == 0:
            return NN.add_link(connections)

        random_connection_index = random_int(len(connections))
        index, start, end, _, weight = connections[random_connection_index]
        connections[random_connection_index] = (index, start, end, False, weight)

        gene_index = NN._next_gene_index
        NN._next_gene_index = NN._next_gene_index + 2

        node_index = NN._next_node_index
        NN._next_node_index = NN._next_node_index + 1

        connections.extend(
            [
                (gene_index, start, node_index, True, 1.0),
                (gene_index + 1, node_index, end, True, weight),
            ]
        )
        return connections

    @staticmethod
    def shift_link(connections):
        if len(connections) == 0:
            return NN.add_link(connections)
        random_connection_index = random_int(len(connections))
        index, start, end, active, weight = connections[random_connection_index]
        connections[random_connection_index] = (
            index,
            start,
            end,
            active,
            weight * random.random() * 4 - 2,
        )
        return connections

    @staticmethod
    def randomize(connections):
        if len(connections) == 0:
            return NN.add_link(connections)
        random_connection_index = random_int(len(connections))
        index, start, end, active, weight = connections[random_connection_index]
        connections[random_connection_index] = (index, start, end, active, random.random() * 4 - 2)
        return connections

    @staticmethod
    def mutate(a):
        possible_mutations = [NN.add_link, NN.split_link, NN.shift_link, NN.randomize]
        mutation_f = possible_mutations[random_int(len(possible_mutations))]
        connections = [*a._connections]
        mutation_f(connections)
        return NN(connections)
