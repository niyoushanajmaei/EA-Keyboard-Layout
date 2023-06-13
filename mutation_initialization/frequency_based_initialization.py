from permutationsga.ga import Initialization
import numpy as np
from permutationsga.problem import Solution

class FrequencyRestrictedInitialization(Initialization):

    def __init__(self, p:int, type: int, high_frequency: list[int], better_region : list[int]):
        """
        Performs an initialization of the individual
        Assigns high frequency characters to the better region with probability 1-p and to the inferior region with probability p.
        Assign the rest randomly
        choose p to be small

        type: 0 for the typewrite and 1 for the digital keyboard (First one is the typewriter)
        high_frequency: A list of 9 indices of character with the highest frequency in the language
        """
        self.high_frequency = high_frequency
        self.p = p
        self.type = type
        self.better_region = better_region


    def initialize(self, rng: np.random.Generator, population: list[Solution]):
        for solution in population:
            self.frequency_restricted_initialization(rng, solution, self.p, self.type, self.high_frequency, self.better_region)

    @staticmethod
    def frequency_restricted_initialization(rng, individual: Solution, p: float, type : int, high_frequency : list[int], better_region : list[int] ):
        """
        Performs an initialization of the individual
        Assigns high frequency characters to the better region with probability 1-p and to the inferior region with probability p.
        Assign the rest randomly
        choose p to be small
        
        type: 0 for the typewrite and 1 for the digital keyboard (First one is the typewriter)
        high_frequency: A list of 9 indices of character with the highest frequency in the language
        """
        
        individual.e = np.array([-1 for i in range(26)])

        better_region_indices = better_region

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
        