import numpy as np
import numpy.typing as npt
from .problem import Problem, Solution


def evaluate_qap(l: int, A: np.matrix, B: np.matrix, s: npt.NDArray[np.int_]):
    f = sum(sum(A[i, j] * B[s[i], s[j]] for i in range(l)) for j in range(l))
    return f

class QAP(Problem):
    def __init__(self, l: int, A: np.matrix, B: np.matrix):
        assert A.shape[0] == l, "QAP matrices must have the right size"
        assert A.shape[1] == l, "QAP matrices must have the right size"
        assert B.shape[0] == l, "QAP matrices must have the right size"
        assert B.shape[1] == l, "QAP matrices must have the right size"
        self.l = l
        self.A = A
        self.B = B

    def get_length(self):
        return self.l

    def evaluate(self, sol: Solution):
        if sol.evaluated:
            return sol.s

        assert sol.s is not None, "Ensure the solution has been decoded, if no decoding is needed, use identity."

        if len(np.unique(sol.s)) != len(sol.s):
            # Solution is not a valid permutation
            return np.inf

        f = evaluate_qap(self.l, self.A, self.B, sol.s)
        sol.f = f
        sol.evaluated = True
        return f


def read_qaplib(filename):
    with open(filename, "r") as f:
        # Read first line: dimensionality
        l = int(f.readline())
        # Skip empty line
        f.readline()
        # Load first matrix
        A = np.loadtxt((f.readline() for _ in range(l)))
        # Skip empty line
        f.readline()
        # Load second matrix
        B = np.loadtxt((f.readline() for _ in range(l)))
        return l, A, B
