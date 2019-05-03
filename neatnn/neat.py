import functools
import math
import random
import sys

import numpy as np
import pygame
import tensorflow as tf

from neatnn import NN, Gene
from neatnn.utils import randbool


class NeatNN(NN):
    _DELTA_WEIGHTS_FACTOR = 0.4
    _DELTA_DISJOINT_FACTOR = 2.0
    _DELTA_TRESHOLD = 1.0

    _CREATE_LINK_CHANCE = 2.0
    _SPLIT_LINK_CHANCE = 0.50
    _DISABLE_LINK_CHANCE = 0.4
    _ENABLE_LINK_CHANCE = 0.2

    def __init__(self, links=None):
        super(NN, links=links)
        self._mutations = [
            ("create_link", NeatNN._CREATE_LINK_CHANCE),
            ("split_link", NeatNN._SPLIT_LINK_CHANCE),
        ]

    @staticmethod
    def align_genes(a, b):
        """Iterate over matching pairs of genes."""
        a_iter = iter(a._links)
        b_iter = iter(b._links)

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
        """Compute the delta between two genomes."""
        num_matching = 0
        matching_weight_delta = 0.0
        num_disjoint = 0

        aligned_genes = NN.align_genes(a, b)
        a, b = next(aligned_genes, (None, None))
        while a and b:
            num_matching = num_matching + 1
            matching_weight_delta = matching_weight_delta + math.fabs(a[4] - b[4])
            a, b = next(aligned_genes, (None, None))

        while a or b:
            num_disjoint = num_disjoint + 1
            a, b = next(aligned_genes, (None, None))

        max_total = max(len(a._links), len(b._links))

        return (
            (matching_weight_delta / num_matching) * NeatNN._DELTA_WEIGHTS_FACTOR
            + (num_disjoint / max_total) * NeatNN._DELTA_DISJOINT_FACTOR
        )

    @staticmethod
    def is_same_species(a, b):
        """Check if both organisms belong to the same species."""
        return NeatNN.delta(a, b) < NN._DELTA_TRESHOLD

    @staticmethod
    def breed(a, b):
        """Breed two organisms."""
        new_links = []

        aligned_genes = NN.align_genes(a, b)
        a, b = next(aligned_genes, (None, None))
        while a and b:
            new_links.append(a if randbool() else b)
            a, b = next(aligned_genes, (None, None))

        excess_genes = []
        had_a = True
        while a or b:
            excess_genes.append(a or b)
            if not had_a and a:
                # These are disjoint therefore added to the gene pool.
                new_links.extend(excess_genes)
                excess_genes = []
            had_a = bool(a)
            a, b = next(aligned_genes, (None, None))

        return NN(links=new_links)

    @staticmethod
    def create_link(links):
        start_nodes = list(set([*[start for _, start, _, _, _ in links], *range(NN._num_in)]))
        end_nodes = list(
            set(
                [
                    *[end for _, _, end, _, _ in links],
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
            return any([is_parent_of(end, b) for _, start, end, _, _ in links if start == a])

        start_node, end_node = 0, 0
        while is_parent_of(end_node, start_node):
            start_node = start_nodes[random_int(len(start_nodes))]
            end_node = end_nodes[random_int(len(end_nodes))]

        links.append((gene_index, start_node, end_node, True, weight))

    @staticmethod
    def split_link(links):
        random_link_index = randint(0, len(links) - 1)
        index, start, end, _, weight = links[random_link_index]
        links[random_link_index] = (index, start, end, False, weight)

        gene_index = NN._next_gene_index
        NN._next_gene_index = NN._next_gene_index + 2

        node_index = NN._next_node_index
        NN._next_node_index = NN._next_node_index + 1

        links.extend(
            [
                (gene_index, start, node_index, True, 1.0),
                (gene_index + 1, node_index, end, True, weight),
            ]
        )
        return links

    @staticmethod
    def shift_link(links):
        if len(links) == 0:
            return NN.add_link(links)
        random_link_index = random_int(len(links))
        index, start, end, active, weight = links[random_link_index]
        links[random_link_index] = (
            index,
            start,
            end,
            active,
            weight * random.random() * 4 - 2,
        )
        return links

    @staticmethod
    def randomize(links):
        if len(links) == 0:
            return NN.add_link(links)
        random_link_index = random_int(len(links))
        index, start, end, active, weight = links[random_link_index]
        links[random_link_index] = (index, start, end, active, random.random() * 4 - 2)
        return links

    def mutate(self):
        """Create a new individual by mutating an existing one."""
        # randomly increase or decrease the chance of all mutations respectively by 5%
        # Max one "link" mutation
        new_links = list(self._links)
        for mutation, chance in self._mutations:
            if random.random() < chance:
                new_links = getattr(NeatNN, mutation)(new_links)

        return NeatNN(links=new_links)


class Species():
    """Groups individuals with similar behavior."""

    self._MAX_STALENESS = 15
    self._CROSSOVER_CHANCE = 0.75

    def __init__(self, adam):
        """Create a new species with `adam` as it's founder."""
        self._is_sorted_by_score = False
        self._individuals = [adam]

        self._max_score = 0
        self._staleness = 0

    @property
    def representative(self):
        """Select the representative of this species.

        TODO: Does this have to be the best-performing?
        """
        return self._individuals[0]

    @property
    def max_score(self):
        """Get this species' max score."""
        return self._max_score

    def add(self, individual):
        """Add `individual` to this species.

        Does not check whether the individual really belongs to this species."""
        self._individuals.append(individual)
        self._is_sorted_by_score = False

    def sort_species_by_score(self):
        """Orders the individuals of this species by score."""
        if not self._is_sorted_by_score:
            self._individuals.sort(reverse=True, key=lambda a: a.score)
            self._is_sorted_by_score = True

    def opinionated_thanos(self):
        """Kill the worst performing 50% of organisms."""
        self.sort_species_by_score()
        # Keep at least one organism in each species.
        self._individuals = self._individuals[:math.ceil(len(organisms) / 2)]

    def sudden_death(self):
        """Kill everyone except the best one."""
        self.sort_species_by_score()
        self._individuals = [self._individuals[0]]

    def update_staleness(self):
        """Checks whether this species' performance has improved.

        If the max score did not increase since the last check, increase the staleness counter.
        """
        current_max = max([i.score for i in self._individuals])
        if current_max > self._max_score:
            self._max_score = current_max
            self._staleness = 0
        else:
            self._staleness = self._staleness + 1

    @property
    def is_stale(self):
        """Check whether the current staleness has reached critical levels."""
        return self._staleness > Species._MAX_STALENESS

    def breed_child(self):
        """Create a new child from the ones in this species."""
        if random.random() < Species._CROSSOVER_CHANCE:
            return NeatNN.breed(random.choice(self._individuals), random.choice(self._individuals))
        else:
            return random.choice(self._individuals).mutate()


class Population():
    """Groups actions on populations."""

    def __init__(self, population=300):
        """Create a new population of size `population`."""
        self._population = population
        self._species = []
        for _ in range(self._population):
            new_organism = NeatNN()
            new_organism = new_organism.mutate()
            self.add_to_respective_species(new_organism)

    def add_to_respective_species(self, individual):
        """Add `individual` to his respective species."""
        for species in self._species:
            if individual.belongs_to(species):
                species.add(individual)
                return
        self._species.append(Species(individual))

    def assign_average_global_rank(self):
        """Assign the global position from the back to each individual."""
        everybody = [individual for species in self._species for individual in species._individuals]
        everybody.sort(key=lambda i: i.score)
        for index, individual in enumerate(everybody):
            # Rank is one-based to simplify future calculations.
            individual.global_rank = index + 1

        for species in self._species:
            species.average_global_rank = np.mean([i.global_rank for i in species._individuals])

    def next_generation(self):
        """Create next generation."""
        # Remove bottom ceil 50% of each species.
        for species in self._species:
            species.opinionated_thanos()
            species.update_staleness()

        self._species.sort(reverse=True, key=lambda s: s.max_score)
        self._species = [s for rank, s in enumerate(self._species) if not s.is_stale or rank == 0]

        # Calculate average (adjusted?) fitness of each species.
        self.assign_average_global_rank()

        # If this species will produce no children, then drop it.
        total_average_global_rank = sum([s.average_global_rank for s in self._species])
        children_per_rank = self._population / total_average_global_rank
        self._species = [
            species for species in self._species
            if species.average_glbal_rank * children_per_rank >= 1
        ]

        # Breed children.
        total_average_global_rank = sum([s.average_global_rank for s in self._species])
        children_per_rank = self._population / total_average_global_rank
        children = []
        for species in self._species:
            # One child is the top-scorer from the previous generation.
            num_children = math.floor(species.average_glbal_rank * children_per_rank) - 1
            children.extend([species.breed_child() for _ in range(num_children)])

        for species in self._species:
            species.sudden_death()

        num_missing = self._population - len(self._species) - len(children)
        for _ in range(num_missing):
            children.append(random.choice(self._species).breed_child())

        for child in children:
            self.add_to_respective_species(child)






MutateConnectionsChance = 0.25
linkMutate
    PerturbChance = 0.90
    StepSize = 0.1
  # For each node
  # with a chance of `perturbChance` add random in range [-step, +step] to weight
  # otherwise assign a random value to connection [-2, 2]
