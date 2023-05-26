#
# GA primitives, implemented for your convenience.
#

import numpy as np
from typing import List, Union
import numpy.typing as npt

from .problem import Problem, Solution


class Initialization:
    def initialize(self, rng: np.random.Generator, population: List[Solution]):
        raise NotImplementedError()


class RandomUniformInitialization(Initialization):
    """
    Initialize uniformly with continuous values.
    """

    def __init__(self, length: int, low: Union[npt.NDArray[np.float64], float] = 0.0, high: Union[npt.NDArray[np.float64], float] = 1.0):
        self.length = length
        self.low = low
        self.high = high

    def initialize(self, rng: np.random.Generator, population: List[Solution]):
        for solution in population:
            solution.e = rng.uniform(self.low, self.high, size=self.length)


class RandomPermutationInitialization(Initialization):
    """
    Initialize with random permutations
    """

    def __init__(self, length: int):
        self.length = length

    def initialize(self, rng: np.random.Generator, population: List[Solution]):
        for solution in population:
            solution.e = rng.permutation(self.length)


class Selection:
    def select(
        self, rng: np.random.Generator, population: List[Solution], num_to_select: int
    ) -> List[Solution]:
        raise NotImplementedError()


def single_tournament(subset: List[Solution], o: int = 1) -> List[Solution]:
    """
    :param subset: The subset of solution on which to perform a tournament
    :param s: The number of solutions surviving this tournament (usually 1)
    """
    subset.sort(key=(lambda x: x.f))
    return subset[:o]


def tournament_selection(
    rng: np.random.Generator,
    population: List[Solution],
    to_select: int,
    s: int = 4,
    o: int = 1,
    shuffle: bool = True,
) -> List[Solution]:
    """
    :param rng: the random number generator to use
    :param population: solutions on which to perform tournament selection
    :param to_select: number of solutions to select
    :param s: the tournament size
    :param o: the number of solutions to select for each tournament
    """
    selected: List[Solution] = []
    idx = 0
    if not shuffle:
        p = np.arange(len(population))
    else:
        p = rng.permutation(len(population))

    while len(selected) < to_select:
        # If we are out of indices, generate a new permutation
        if idx + s >= len(population) and shuffle:
            p = rng.permutation(len(population))
            idx = 0
        #
        selected += single_tournament([population[i] for i in p[idx : idx + s]], o)
        idx += s
    # Ensure the population & selection are exactly the same size.
    # i.e. remove any extras, if necessary.
    return selected[:to_select]


class TournamentSelection(Selection):
    def __init__(self, s: int = 4, o: int = 1, shuffle=True):
        self.s = s
        self.o = o
        self.shuffle = shuffle

    def select(
        self, rng: np.random.Generator, population: List[Solution], num_to_select: int
    ) -> List[Solution]:
        return tournament_selection(
            rng, population, num_to_select, self.s, self.o, self.shuffle
        )


class SequentialSelector(Selection):
    def __init__(self, shuffle=True):
        self.shuffle = shuffle
        self.ordering = np.zeros(0)
        self.position = 0

    def select(
        self, rng: np.random.Generator, population: List[Solution], num_to_select: int
    ) -> List[Solution]:
        if len(population) != len(self.ordering):
            self.ordering = np.arange(len(population))
            if self.shuffle:
                rng.shuffle(self.ordering)
            self.position = 0

        def next_sample():
            r = self.ordering[self.position]
            self.position += 1

            if self.position >= len(population):
                rng.shuffle(self.ordering)
                self.position = 0

            return population[r]

        return [next_sample() for _ in range(num_to_select)]


class UniformSamplingSelector(Selection):
    def __init__(self):
        pass

    def select(
        self, rng: np.random.Generator, population: List[Solution], num_to_select: int
    ) -> List[Solution]:
        return list(rng.choice(population, size=num_to_select)) # type: ignore


class Recombinator:
    def recombine(
        self, rng: np.random.Generator, population: List[Solution]
    ) -> List[Solution]:
        raise NotImplementedError()


def get_de_mask(
    rng: np.random.Generator,
    cr: float,
    l: int,
):
    # Define which variables to replace
    mask = rng.choice((True, False), size=l, p=(1 - cr, cr))
    # Select a variable to always replace (otherwise we could potentially end up changing nothing!)
    mask[rng.choice(l)] = True
    return mask


def crossover_de(
    # The current solution (i.e. to alter)
    x: Solution,
    # Solutions used for recombination, traditionally selected at random
    base_r0: Solution,
    r1: Solution,
    r2: Solution,
    # Which variables to replace
    mask: npt.NDArray[np.bool_],
    # Scale parameter
    f: float,
) -> Solution:
    assert x.e is not None, "Ensure solution x is initialized before use."
    assert base_r0.e is not None, "Ensure solution base_r0 is are initialized before use."
    assert r1.e is not None, "Ensure solution r1 is initialized before use."
    assert r2.e is not None, "Ensure solution r2 is initialized before use."
    all_d = base_r0.e + f * (r1.e - r2.e)
    return Solution(np.where(mask, x.e, all_d))


class DifferentialEvolutionRecombinator(Recombinator):
    def __init__(self, cr: float, f: float):
        """
        Recombine according to the methodology used in Differential Evolution

        See the following works:
        -   Price, Kenneth V., Rainer M. Storn, and Jouni A. Lampinen. 2005.
            Differential Evolution: A Practical Approach to Global Optimization.
            Natural Computing Series. Berlin; New York: Springer.
        -   Storn, Rainer, and Kenneth Price. 1997.
            'Differential Evolution - A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces'.
            Journal of Global Optimization 11 (4): 341-59. https://doi.org/10.1023/A:1008202821328.
        """
        self.cr = cr
        self.f = f

    def recombine(
        self, rng: np.random.Generator, population: List[Solution]
    ) -> List[Solution]:
        offspring = []
        for solution in population:
            assert solution.e is not None, "Ensure solution is initialized before use."
            # Select other solutions uniformly at random.
            base_r0 = rng.choice(population) # type: ignore
            r1 = rng.choice(population) # type: ignore
            r2 = rng.choice(population) # type: ignore

            # Compute the mask
            l = len(solution.e)
            mask = get_de_mask(rng, self.cr, l)

            # Provide both the original solution and the resulting solution such that
            # we can use non-shuffling tournament selection of size 2, returning 1 to
            # perform the original selection procedure.
            offspring += [
                solution,
                crossover_de(solution, base_r0, r1, r2, mask, self.f),
            ]
        return offspring


def invperm(permutation):
    """
    Invert a permutation

    Via https://stackoverflow.com/a/55737198
    """
    inv = np.empty_like(permutation)
    inv[permutation] = np.arange(len(inv), dtype=inv.dtype)
    return inv

def generate_uniform_mask(rng: np.random.Generator, l: int, p: float, at_least_one: bool = True) -> npt.NDArray[np.bool_]:
    """
    Generate a mask uniformly with probability p.
    """
    mask = rng.choice((True, False), size=l, p=(p, 1-p))
    if at_least_one:
        mask[rng.integers(0, l, endpoint=False)] = True
    return mask

def generate_uniform_indices(rng: np.random.Generator, l: int, p: float, at_least_one: bool = True)  -> npt.NDArray[np.int_]:
    """
    Generate indices uniformly with probability p.
    """
    return np.where(generate_uniform_mask(rng, l, p, at_least_one))[0]

def generate_sequential_mask(rng: np.random.Generator, l: int) -> npt.NDArray[np.bool_]:
    """
    Generate a mask such that the positions between a and b are True, and others false.

    a and b are picked uniformly, and reordered.
    """
    r = np.full(l, False)
    x0, x1 = rng.integers(0, l, size=2, endpoint=False)
    a, b = min(x0, x1), max(x0, x1)
    r[a:b] = True
    return r

def generate_sequential_indices(rng: np.random.Generator, l: int) -> npt.NDArray[np.int_]:
    """
    Generate indices such that the positions between a and b are True, and others false.

    a and b are picked uniformly, and reordered.
    """
    return np.where(generate_sequential_mask(rng, l))[0]


def generate_sequential_wrapping_mask(rng: np.random.Generator, l: int) -> npt.NDArray[np.bool_]:
    """
    Generate a mask such that the positions between a and b are True, and others false.
    Wraps around when sampled a and b, a > b.
    """
    r = np.full(l, False)
    a, b = rng.integers(0, l, size=2, endpoint=False)
    if b >= a:
        r[a:b] = True
    else:
        r[a:] = True
        r[:b] = True
    return r

def generate_sequential_wrapping_indices(rng: np.random.Generator, l: int) -> npt.NDArray[np.int_]:
    """
    Generate indices such that the positions between a and b are True, and others false.
    Wraps around when sampled a and b, a > b.
    """
    return np.where(generate_sequential_wrapping_mask(rng, l))[0]

def crossover_pmx(indices, s0: Solution, s1: Solution):
    assert s0.e is not None, "Ensure solution s0 is initialized before use."
    assert s1.e is not None, "Ensure solution s1 is initialized before use."

    # Prepare copies, and the inverse to perform lookups on.
    r0 = np.copy(s0.e)
    r0inv = invperm(r0)
    r1 = np.copy(s1.e)
    r1inv = invperm(r1)

    for i in indices:
        # We want r0[i], r1[i] = r1[i], r0[i], but that could invalidate the permutation.
        # We know that r1[i] == r0[r0inv[r1[i]]] (other way around is similar.) by definition
        #   of the inverse permutation
        # Therefore we can get r0[i] = r1[i], while perserving uniqueness by swapping
        # r0[i] and r0[r0inv[r1[i]]]
        o = r0inv[r1[i]]
        r_o, r_i = r0[o], r0[i]
        r0[i], r0[o] = r_o, r_i
        # To update r0inv, a similar swap should be performed on r0inv:
        r0inv[r_i], r0inv[r_o] = r0inv[r_o], r0inv[r_i]
        
        # Equivalently for r1 - perform swap
        o = r1inv[r0[o]]
        r_o, r_i = r1[o], r1[i]
        r1[i], r1[o] = r_o, r_i
        # To update r1inv, a similar swap should be performed on r0inv:
        r1inv[r_i], r1inv[r_o] = r1inv[r_o], r1inv[r_i]

    return [Solution(r0), Solution(r1)]


def crossover_cx(indices, s0: Solution, s1: Solution):
    assert s0.e is not None, "Ensure solution s0 is initialized before use."
    assert s1.e is not None, "Ensure solution s1 is initialized before use."

    # Prepare copies, and the inverse to perform lookups on.
    r0 = np.copy(s0.e)
    r0inv = invperm(r0)
    r1 = np.copy(s1.e)

    # Note: it is potentially better to keep the number of indices low for this operator,
    #       as it copies over multiple anyways (possibly covering the entire permutation)
    for s in indices:
        # Cycle crossover copies over more indices in order to preserve uniqueness.
        # Specifically, it copies over a cycle. Example:
        # Given two solutions a, b:
        #  a:  [0, 1, 2, 3, 4]
        #  b:  [4, 2, 3, 1, 0]
        # (idx) 0  1  2  3  4
        # First of all, we choose to cross over index 1, leading to a duplicate 2.
        #  a:  [0, 2, 2, 3, 4]
        #  b:  [4, 1, 3, 1, 0]
        # (idx) 0  1  2  3  4
        # In order to avoid duplicates, we swap the original position of two as well.
        # Leading to a duplicate 3.
        #  a:  [0, 2, 3, 3, 4]
        #  b:  [4, 1, 2, 1, 0]
        # (idx) 0  1  2  3  4
        # Again, swapping the position with the original 3, we swap the position that originally had a one
        # Which was actually our starting point! (hence: does not exist, was replaced!)
        #  a:  [0, 2, 3, 1, 4]
        #  b:  [4, 1, 2, 3, 0]
        # (idx) 0  1  2  3  4
        # and finally back to the start: where a 1 occurred in a (no change neccessary: already OK).
        # We end up with a valid solution again.

        r0[s], r1[s] = r1[s], r0[s]
        invalidated_idx = s
        # Also keep track of the index that was removed, this one should be reintroduced at the end
        # of a cycle
        r0inv[r1[s]] = -1
        # 
        while r0inv[r0[invalidated_idx]] != -1:
            # We have created a duplicate of the element at r0[invalidated_idx]
            v = r0[invalidated_idx]
            # The original position of this is located at r0inv[r0[invalidated_idx]]
            # Let us decide that this one is invalid instead.
            o = invalidated_idx
            invalidated_idx = r0inv[v]
            # This swap will likely reintroduce an invalid element.
            r0[invalidated_idx], r1[invalidated_idx] = r1[invalidated_idx], r0[invalidated_idx]
            r0inv[v] = o
        # As this value was absent, we can simply update the inverse:
        r0inv[r0[invalidated_idx]] = invalidated_idx
        # np.testing.assert_array_equal(r0inv, invperm(r0))

    return [Solution(r0), Solution(r1)]


def crossover_ox_neg(not_indices, s0: Solution, s1: Solution):
    assert s0.e is not None, "Ensure solution s0 is initialized before use."
    assert s1.e is not None, "Ensure solution s1 is initialized before use."

    # not_indices are the indices that shouldn't be copied over and preserve their exact position
    r0 = np.copy(s1.e)
    r0inv = invperm(s0.e)
    r0[not_indices] = r0[not_indices][np.argsort(r0inv[r0[not_indices]])]

    r1 = np.copy(s0.e)
    r1inv = invperm(s1.e)
    r1[not_indices] = r1[not_indices][np.argsort(r1inv[r1[not_indices]])]

    return [Solution(r0), Solution(r1)]

def crossover_ox(indices, s0: Solution, s1: Solution):
    not_indices = list(set(range(len(s1.e))) - set(indices))
    return crossover_ox_neg(not_indices, s0, s1)

class FunctionBasedRecombinator(Recombinator):
    """
    A simple recombinator that utilizes a particular function to create new individuals
    from a set of indices and two individuals
    """

    def __init__(
        self,
        indices_function,
        crossover_function,
        parent_selection: Selection,
        num_offspring: int,
        include_what=None,
    ):
        self.indices_function = indices_function
        self.crossover_function = crossover_function
        self.parent_selection = parent_selection
        self.num_offspring = num_offspring
        self.include_what = include_what

    def recombine(
        self, rng: np.random.Generator, population: List[Solution]
    ) -> List[Solution]:
        offspring = []
        if self.include_what == "population":
            offspring += population
        while len(offspring) < self.num_offspring:
            parents = self.parent_selection.select(rng, population, 2)
            if self.include_what == "parents":
                offspring += parents
            offspring += self.crossover_function(
                self.indices_function(), parents[0], parents[1]
            )
        return offspring


class ConfigurableGA:
    def __init__(
        self,
        seed: int,
        population_size: int,
        problem: Problem,
        initialization: Initialization,
        recombinator: Recombinator,
        selection: Selection,
    ):
        # Create solution containers
        self.population = [Solution(None) for _ in range(population_size)]
        # Store variables
        self.problem = problem
        self.initialization = initialization
        self.recombinator = recombinator
        self.selection = selection
        # Initialize the RNG
        self.rng = np.random.default_rng(seed=seed)
        # We have not yet initialized the population - should occur upon first generation
        self.initialized = False

    def initialize(self):
        # Use initializer to set solution values
        self.initialization.initialize(self.rng, self.population)
        # Evaluate all initial solutions
        for solution in self.population:
            self.problem.evaluate(solution)

    def create_offspring_and_select(self):
        # Create offspring (potentially)
        offspring = self.recombinator.recombine(self.rng, self.population)
        for solution in offspring:
            self.problem.evaluate(solution)
        self.population = self.selection.select(self.rng, offspring, len(self.population))

    def generation(self):
        if not self.initialized:
            # First: initialize the population
            self.initialize()
            self.initialized = True
        else:
            # Perform normal generation
            self.create_offspring_and_select()
