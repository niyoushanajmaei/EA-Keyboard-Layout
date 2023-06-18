import numpy as np
from permutationsga.problem import Solution
from permutationsga.qap import QAP
import logging


class RegionBasedCrossover():

    def __init__(self):
        logging.basicConfig(filename="cross.log", level=logging.DEBUG, filemode='w')
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Initialized RegionBasedCrossover")
        self.logger.disabled = False

    def region_based_crossover(self, matrices, parent1: Solution, parent2: Solution):
        """
        Performs a crossover between two parents
        Copies each of the six regions of the keyboard from one randomly selected parent 
        Uses repair operations to fix clashes

        matrices = [frequency_matrix, delay_matrix]
        """
       

        self.logger.debug("Entering region_based_crossover")

        assert parent1.e is not None, "Ensure solution s0 is initialized before use."
        assert parent2.e is not None, "Ensure solution s1 is initialized before use."

        children = [np.array([-1 for i in range(26)]), np.array([-1 for i in range(26)])]
        parents = [parent1.e, parent2.e]

        regions = [np.array(list(range(0, 5))), np.array(list(range(5, 10))), np.array(list(range(10, 15))), np.array(list(range(15, 19))), np.array(list(range(19, 24))), np.array(list(range(24, 26)))]
        
        for i in range(len(regions)):
            region = regions[i]
            p_rand = np.random.randint(0, 2)
            for j in range(len(region)):
                children[0][region[j]] = parents[p_rand][region[j]]
                children[1][region[j]] = parents[1-p_rand][region[j]]
            
        self.logger.debug(f"Children before repair: {children}")
        if not self.is_valid(children[0]):
            children[0] = self.repair(children[0])
        if not self.is_valid(children[1]):
            children[1] = self.repair(children[1])

        self.logger.debug(f"Children after repair: {children}")

        if not self.is_valid(children[0]):
            self.logger.debug("Child 0 is not valid")
        if not self.is_valid(children[1]):
            self.logger.debug("Child 1 is not valid")


        self.logger.debug("Exiting linkage_based_crossover")
        return [Solution(children[0]), Solution(children[1])]



    def is_valid(self, child):
        """
        Returns True if all characters are unique and there is no -1 in the child
        """
        if -1 in child:
            return False
        if len(set(child)) != 26:
            return False
        return True
    

    def repair(self, child):
        """
        take out one of the duplicate characters and then place the characters that are not included randomly
        """
        self.logger.debug(f"Repairing child {child}")
        duplicates = self.find_duplicates(child)
        self.logger.debug(f"Found duplicate {duplicates}")
        child = self.remove_duplicates(child, duplicates)
        self.logger.debug(f"Removed duplicate {duplicates}")
        child = self.fill_in(child)
        self.logger.debug(f"Filled in child {child}")
        return child
    

    def find_duplicates(self, child):
        """
        Returns a list of duplicate characters in the child
        """
        duplicates = []
        for i in range(26):
            if child[i] in child[i+1:]:
                duplicates.append(child[i])
        return duplicates
    

    def remove_duplicates(self, child, duplicates):
        """
        Removes one of the duplicate characters from the child
        """
        for duplicate in duplicates:
            i = np.random.randint(0, 2)
            child[np.where(child == duplicate)[0][i]] = -1
        return child
    

    def fill_in(self, child):
        """
        Fills in the child with characters that are not included
        """
        missing = [i for i in range(26) if i not in child]
        np.random.shuffle(missing)
        for i in range(26):
            if child[i] == -1:
                child[i] = missing[0]
                missing = np.delete(missing, 0)
        return child
    
    

