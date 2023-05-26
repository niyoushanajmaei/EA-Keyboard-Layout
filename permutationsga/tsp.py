# a previous version of the assignment had to option to do TSP.
import numpy as np
# Documentation see https://tsplib95.readthedocs.io/en/stable/
import tsplib95 as tsp # type: ignore

from .problem import Problem, Solution


class TSP(Problem):
    def __init__(self, problem: tsp.models.Problem):
        self.problem = problem

    def get_length(self):
        return self.problem.dimension

    def evaluate(self, sol: Solution):
        """
        Evaluate a permutation against a TSP problem instance.
        """
        if sol.evaluated:
            return sol.s

        assert sol.s is not None, "Ensure the solution has been decoded, if no decoding is needed, use identity."

        if len(np.unique(sol.s)) != len(sol.s):
            # Solution is not a valid permutation
            return np.inf

        # Note, TSPLIB95 uses 1-based Permutations / indexing, while numpy (and we) use zero based (indexed) permutations.
        # Convert by adding one to all the elements.
        f = self.problem.trace_tours([sol.s + 1])[0]
        # Set & return fitness
        sol.f = f
        sol.evaluated = True
        return f
