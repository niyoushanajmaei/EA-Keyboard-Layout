from typing import Optional
import numpy as np

class Solution:
    """
    Dataclass for containing the current solution (permutation) & corresponding fitness (if evaluated)
    """

    def __init__(self, e: Optional[np.ndarray]):
        self.evaluated = False
        self.e = e # encoded format
        self.s: Optional[np.ndarray] = None # actual solution: decoded format (for this assignment: always a permutation)
        self.f = np.inf # fitness


class Problem:
    def get_length(self):
        return 0

    def evaluate(self, solution: Solution):
        return 0.0


class IdenticalDecoder(Problem):
    """
    Encoded solution is identical to the format the problem expects for evaluation.
    """

    def __init__(self, problem: Problem):
        self.problem = problem

    def get_length(self):
        return self.problem.get_length()

    def evaluate(self, sol: Solution):
        if sol.evaluated:
            return sol.s

        sol.s = sol.e
        return self.problem.evaluate(sol)

def invperm(permutation):
    """
    Invert a permutation

    Via https://stackoverflow.com/a/55737198
    """
    inv = np.empty_like(permutation)
    inv[permutation] = np.arange(len(inv), dtype=inv.dtype)
    return inv

class InvPermDecoder(Problem):
    """
    Encoded solution is the inverse permutation of the actual solution.
    (Or: the actual solution is the inverse of the encoded solution)
    """

    def __init__(self, problem: Problem):
        self.problem = problem

    def get_length(self):
        return self.problem.get_length()

    def evaluate(self, sol: Solution):
        if sol.evaluated:
            return sol.s

        sol.s = invperm(sol.e)
        return self.problem.evaluate(sol)

class RandomKeysDecoder(Problem):
    """
    Solution is encoded in random keys, decode first, then evaluate.
    """

    def __init__(self, problem: Problem):
        self.problem = problem

    def get_length(self):
        return self.problem.get_length()

    def evaluate(self, sol: Solution):
        if sol.evaluated:
            return sol.s

        assert sol.e is not None, "Ensure solution sol is initialized before use."

        sol.s = np.argsort(sol.e)
        return self.problem.evaluate(sol)
