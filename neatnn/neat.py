"""Implementation of NeuroEvolution of Augmented Topologies."""

import logging
import math
import random

import numpy as np
from neatnn.nn import NN, Gene
from neatnn.utils import make_set, randbool

logger = logging.getLogger(__name__)


class NeatNN(NN):
    """Neural network class with methods for running NEAT."""

    _DELTA_WEIGHTS_FACTOR = 0.4
    _DELTA_DISJOINT_FACTOR = 2.0
    _DELTA_TRESHOLD = 1.0

    _CREATE_LINK_CHANCE = 2.0
    _SPLIT_LINK_CHANCE = 0.5
    _MUTATE_LINKS_CHANCE = 0.25
    _MUTATE_LINK_PERTURB_CHANCE = 0.9
    _DISABLE_LINK_CHANCE = 0.4
    _ENABLE_LINK_CHANCE = 0.2

    def __init__(self, links=None, clone=None):
        """Create a new neat-backed NN."""
        if links is not None and clone:
            raise Exception("Can't specify both `links` and `clone`.")

        if clone:
            super().__init__(links=clone._links)
            self._mutations = list(clone._mutations)
        else:
            super().__init__(links=links if links is not None else [])
            self._mutations = [
                ("create_link", NeatNN._CREATE_LINK_CHANCE),
                ("split_link", NeatNN._SPLIT_LINK_CHANCE),
                ("mutate_links", NeatNN._MUTATE_LINKS_CHANCE),
                ("disable_link", NeatNN._DISABLE_LINK_CHANCE),
                ("enable_link", NeatNN._ENABLE_LINK_CHANCE),
            ]

    @staticmethod
    def align_genes(a, b):
        """Iterate over matching pairs of genes."""
        a_iter = iter(a._links)
        b_iter = iter(b._links)

        a, b = next(a_iter, None), next(b_iter, None)
        while a or b:
            if a and b and a.index == b.index:
                yield (a, b)
                a, b = next(a_iter, None), next(b_iter, None)
            elif b and (not a or a.index > b.index):
                yield (None, b)
                b = next(b_iter, None)
            elif a and (not b or a.index < b.index):
                yield (a, None)
                a = next(a_iter, None)

    @staticmethod
    def delta(individual_a, individual_b):
        """Compute the delta between two genomes."""
        num_matching = 0
        matching_weight_delta = 0.0
        num_disjoint = 0

        aligned_genes = NeatNN.align_genes(individual_a, individual_b)
        a, b = next(aligned_genes, (None, None))
        while a and b:
            num_matching = num_matching + 1
            matching_weight_delta = matching_weight_delta + math.fabs(a.weight - b.weight)
            a, b = next(aligned_genes, (None, None))

        while a or b:
            num_disjoint = num_disjoint + 1
            a, b = next(aligned_genes, (None, None))

        max_total = max(len(individual_a._links), len(individual_b._links))

        weight_delta = matching_weight_delta / num_matching if num_matching else 0
        weight_delta = weight_delta * NeatNN._DELTA_WEIGHTS_FACTOR
        disjoint_delta = num_disjoint / max_total * NeatNN._DELTA_DISJOINT_FACTOR
        return weight_delta + disjoint_delta

    def belongs_to(self, species):
        """Check if `self` belongs to `species`."""
        return NeatNN.delta(self, species.representative) < NeatNN._DELTA_TRESHOLD

    def set_score(self, score):
        """Store the score."""
        self._score = score

    @staticmethod
    def crossover(a, b):
        """Breed two individuals."""
        new_links = []

        aligned_genes = NeatNN.align_genes(a, b)
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

        return NeatNN(links=new_links)

    def create_link(self):
        """Join two nodes with a link."""
        start_nodes = range(NN._num_inputs)
        start_nodes = make_set(*[gene.start for gene in self._links], *start_nodes)
        end_nodes = [index + NN._num_inputs for index in range(NN._num_outputs)]
        end_nodes = make_set(*[gene.end for gene in self._links], *end_nodes)

        start_node = random.choice(start_nodes)
        end_node = random.choice(end_nodes)

        if any(link.start == start_node and link.end == end_node for link in self._links):
            logger.debug(f"Skipping mutation, link between {start_node} and {end_node} exists.")
            return

        # Check globally or locally? Use global ID if link exists globally?
        # if Gene.exists(start_node, end_node):

        self._links.append(Gene(start_node, end_node, True))

    def split_link(self):
        """Split a link and create a node in between."""
        assert self._links
        link = random.choice(self._links)
        if not link.active:
            return

        link.toggle()
        node_index = NN._next_node_index
        self._links.extend(
            [
                Gene(link.start, node_index, True, weight=1.0),
                Gene(node_index, link.end, True, weight=link.weight),
            ]
        )

    def mutate_links(self):
        """Mutate the weights on all links."""
        for link in self._links:
            if random.random() < NeatNN._MUTATE_LINK_PERTURB_CHANCE:
                link.randomize_weight()
            else:
                link.shift_weight()

    def disable_link(self):
        """Disable an enabled link."""
        candidates = [link for link in self._links if link.active]
        if candidates:
            random.choice(candidates).toggle()

    def enable_link(self):
        """Enable a disabled link."""
        candidates = [link for link in self._links if not link.active]
        if candidates:
            random.choice(candidates).toggle()

    def mutate(self):
        """Create a new individual by mutating an existing one."""
        # randomly increase or decrease the chance of all mutations respectively by 5%
        # Max one "link" mutation
        logger.debug(f"Mutating {self}")
        clone = NeatNN(clone=self)
        for mutation, chance in clone._mutations:
            if random.random() < chance:
                logger.debug(f"Applying {mutation} mutation with a chance of {chance:.2f}")
                getattr(clone, mutation)()

        return clone


class Species:
    """Groups individuals with similar behavior."""

    _species_stats = {}
    _next_species_index = 0

    _MAX_STALENESS = 15
    _CROSSOVER_CHANCE = 0.75

    def __init__(self, adam):
        """Create a new species with `adam` as it's founder."""
        self._individuals = [adam]

        self._max_score = 0
        self._staleness = 0

        self._index = self._next_species_index
        Species._next_species_index = Species._next_species_index + 1

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

        Does not check whether the individual really belongs to this species.
        """
        self._individuals.append(individual)

    def sort_by_score(self):
        """Orders the individuals of this species by score."""
        self._individuals.sort(reverse=True, key=lambda a: a._score)

    def opinionated_thanos(self):
        """Kill the latter 50% of organisms."""
        # Keep at least one organism in each species.
        self._individuals = self._individuals[: math.ceil(len(self._individuals) / 2)]

    def sudden_death(self):
        """Kill everyone except the first."""
        self._individuals = [self._individuals[0]]

    def update_staleness(self):
        """Check whether this species' performance has improved.

        If the max score did not increase since the last check, increase the staleness counter.
        """
        current_max = self._individuals[0]._score
        if current_max > self._max_score or current_max > 100000:
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
            return NeatNN.crossover(
                random.choice(self._individuals), random.choice(self._individuals)
            )
        else:
            return random.choice(self._individuals).mutate()


class Population:
    """Groups actions on populations."""

    def __init__(self, population=300):
        """Create a new population of size `population`."""
        self._population = population
        self._generation = 0
        self._species = []
        for _ in range(self._population):
            new_organism = NeatNN()
            new_organism = new_organism.mutate()
            self.add_to_respective_species(new_organism)

    def everybody(self):
        """Iterate through every individual."""
        for species in self._species:
            for individual in species._individuals:
                yield individual

    def add_to_respective_species(self, individual):
        """Add `individual` to his respective species."""
        for species in self._species:
            if individual.belongs_to(species):
                species.add(individual)
                return
        self._species.append(Species(individual))

    def assign_average_global_rank(self):
        """Assign the global position from the back to each individual."""
        everybody = list(self.everybody())
        everybody.sort(key=lambda i: i._score)
        for index, individual in enumerate(everybody):
            # Rank is one-based to simplify future calculations.
            individual.global_rank = index + 1

        for species in self._species:
            species.average_global_rank = np.mean([i.global_rank for i in species._individuals])

    def next_generation(self):
        """Create next generation."""
        self._generation = self._generation + 1

        logger.info(f"Generation {self._generation}")

        for species in self._species:
            species.sort_by_score()
            species.update_staleness()

        self._species.sort(reverse=True, key=lambda s: s.max_score)

        n = min(5, len(self._species))
        logger.info(f"Best performing {n} species:")
        for index in range(n):
            species = self._species[index]
            rank = f"\t#{index + 1}\t{species._index}:"
            size = f"\tSize = {len(species._individuals)}"
            max_score = f"\tMax score = {species.max_score}"
            staleness = f"\tStaleness = {species._staleness}{' RIP' if species.is_stale else ''}"
            logger.info(f"{rank}{size}{max_score}{staleness}")

        for species in self._species:
            species.opinionated_thanos()

        self._species = [s for rank, s in enumerate(self._species) if not s.is_stale or rank == 0]

        # Calculate average (adjusted?) fitness of each species.
        self.assign_average_global_rank()

        n = min(5, len(self._species))
        logger.info(f"Best performing {n} species after selection purge:")
        for index in range(n):
            species = self._species[index]
            rank = f"\t#{index + 1}\t{species._index}:"
            size = f"\tSize = {len(species._individuals)}"
            max_score = f"\tMax score = {species.max_score}"
            average_global_rank = f"\tAverage global rank = {species.average_global_rank:.1f} [{' '.join(f'{i._score:n}' for i in species._individuals[:10])}]"
            logger.info(f"{rank}{size}{max_score}{average_global_rank}")

        # If this species will produce no children, then drop it.
        total_average_global_rank = sum([s.average_global_rank for s in self._species])
        children_per_rank = self._population / total_average_global_rank
        self._species = [
            species
            for species in self._species
            if (species.average_global_rank * children_per_rank) >= 1
        ]

        n = min(5, len(self._species))
        logger.info(f"Best performing {n} species after reproduction purge:")
        for index in range(n):
            species = self._species[index]
            rank = f"\t#{index + 1}\t{species._index}:"
            size = f"\tSize = {len(species._individuals)}"
            max_score = f"\tMax score = {species.max_score}"
            average_global_rank = f"\tAverage global rank = {species.average_global_rank:.1f}"
            logger.info(f"{rank}{max_score}{average_global_rank}")

        # Breed children.
        total_average_global_rank = sum([s.average_global_rank for s in self._species])
        children_per_rank = self._population / total_average_global_rank
        children = []
        for species in self._species:
            # One child is the top-scorer from the previous generation.
            num_children = math.floor(species.average_global_rank * children_per_rank) - 1
            children.extend([species.breed_child() for _ in range(num_children)])

        for species in self._species:
            species.sudden_death()

        n = min(5, len(self._species))
        logger.info(f"Best performing {n} species after sudden death:")
        for index in range(n):
            species = self._species[index]
            rank = f"\t#{index + 1}\t{species._index}:"
            score = f"\tScore = {species._individuals[0]._score:n}"
            logger.info(f"{rank}{score}")

        num_missing = self._population - len(self._species) - len(children)
        for _ in range(num_missing):
            children.append(random.choice(self._species).breed_child())

        for child in children:
            self.add_to_respective_species(child)
