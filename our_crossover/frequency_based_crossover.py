import numpy as np
from permutationsga.problem import Solution
from permutationsga.qap import QAP, read_qaplib

class FrequencyBasedCrossover():

    def crossover_freq(indices, s0: Solution, s1: Solution):
        assert s0.e is not None, "Ensure solution s0 is initialized before use."
        assert s1.e is not None, "Ensure solution s1 is initialized before use."

        # Prepare copies, and the inverse to perform lookups on.
        r0 = np.copy(s0.e)
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
          

        return [Solution(r0), Solution(r1)]

    def __init__(self, qap_problem: QAP, shuffle_factor: float):
        """
        Performs an initialization of the individual
        Assigns high frequency characters to the better region with probability 1-p and to the inferior region with probability p.
        Assign the rest randomly
        choose p to be small

        type: 0 for the typewrite and 1 for the digital keyboard (First one is the typewriter)
        high_frequency: A list of 9 indices of character with the highest frequency in the language
        """
        bigram = qap_problem.B
        print(bigram)
        self.type = type


    def initialize(self, rng: np.random.Generator, population: list[Solution]):
        for solution in population:
            self.frequency_restricted_initialization(rng, solution, self.p, self.type, self.high_frequency)

    
    @staticmethod
    def frequency_restricted_initialization(rng, individual: Solution, p: float, type : int, high_frequency : list[int]):
        """
        Performs an initialization of the individual
        Assigns high frequency characters to the better region with probability 1-p and to the inferior region with probability p.
        Assign the rest randomly
        choose p to be small
        
        type: 0 for the typewrite and 1 for the digital keyboard (First one is the typewriter)
        high_frequency: A list of 9 indices of character with the highest frequency in the language
        """
        
        individual.e = np.array([-1 for i in range(26)])

        # Assuming middle row is always better. Both in the digital keyboard and the typewriter
        if type == 0:
            better_region_indices = [i for i in range(10, 19)]
        else:
            better_region_indices = [i for i in range(10, 19)] 

        worse_region_indices = [i for i in range(26) if i not in better_region_indices]

        for char in high_frequency:
            if rng.random() < p:
                # assign to an inferior region
                while(True):
                    rand_ind = rng.integers(low=0, high=len(worse_region_indices), size=1)[0]
                    loc = worse_region_indices[rand_ind]
                    if individual.e[loc] == -1:
                        individual.e[loc] = char
                        break
            else:
                # assign to the better_region
                while True:
                    rand_ind = rng.integers(low=0, high=len(better_region_indices), size=1)[0]
                    loc = better_region_indices[rand_ind]
                    if individual.e[loc] == -1:
                        individual.e[loc] = char
                        break

        for char in [i for i in range(26) if i not in high_frequency]:
            while True:
                loc = rng.integers(low=0, high=26, size=1)[0]
                if individual.e[loc] == -1:
                    individual.e[loc] = char
                    break

        return 
        

if __name__ == "__main__":
    problem = QAP(*read_qaplib("./instances/qap/bur26a.dat"))
    cr = FrequencyBasedCrossover(problem,0.2)
