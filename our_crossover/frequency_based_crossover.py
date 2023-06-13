import numpy as np
import matplotlib.pyplot as plt
from permutationsga.problem import Solution
from permutationsga.qap import QAP, read_qaplib


def count_letter_bigram_frequencies(bigram):
    # We don't care about which letter comes first, therefore we construct a symmetric matrix
    return bigram + bigram.T


def count_letter_frequencies(bigram):
    # We don't care about which letter comes first, therefore we construct a symmetric matrix
    symmetric_bigram = bigram + bigram.T

    # Let's see how the frequencies look like
    plt.imshow(symmetric_bigram)
    plt.show()

    res = np.zeros(len(symmetric_bigram))
    for column in range(len(symmetric_bigram)):
        res[column] = sum(symmetric_bigram[column]) / 2
    return res


class FrequencyBasedCrossover():

    def greedy_crossover(self, indices, s0: Solution, s1: Solution):
        """
        This crossover doesn't work because it returns invalid solutions.

        Idea: Letter combinations that are often used together should come from the same parent.
        This algorithm tries to greedily maximize the total bigram scores for both of the children.
        """
        assert s0.e is not None, "Ensure solution s0 is initialized before use."
        assert s1.e is not None, "Ensure solution s1 is initialized before use."

        # Define children
        children_a = np.zeros_like(s0.e, dtype=int)
        children_b = np.zeros_like(s1.e, dtype=int)

        # Shows which elements of the children_a are coming from which parent
        children_a_ingredients = {
            0: [],
            1: []
        }

        # Shows which elements of the children_b are coming from which parent
        children_b_ingredients = {
            0: [],
            1: []
        }

        # Find the optimal order for performing swaps
        order = self.find_order(s0.e, s1.e)

        # For each location in genotype either swap it or not
        for i in order:
            e0, e1 = s0.e[i], s1.e[i]
            # Get scores for swapping vs not swapping
            not_swap_1, swap_1 = self.calculate_score_increase(children_a_ingredients, (e0, e1))
            not_swap_2, swap_2 = self.calculate_score_increase(children_b_ingredients, (e0, e1))

            if not_swap_1 + not_swap_2 > swap_1 + swap_2:
                # Not swap elements because not-swapping has higher score
                children_a_ingredients[0].append(e0)
                children_b_ingredients[1].append(e1)
                children_a[i] = e0
                children_b[i] = e1
            else:
                # Swap elements because swapping has higher score
                children_a_ingredients[1].append(e1)
                children_b_ingredients[0].append(e0)
                children_a[i] = e1
                children_b[i] = e0
        print("r0:", children_a)
        print("r1:",children_b)
        return [Solution(children_a), Solution(children_b)]



    def
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

        self.bi_frequencies = count_letter_bigram_frequencies(bigram)
        self.frequencies = count_letter_frequencies(bigram)

        self.type = type

    def find_order(self, e0, e1):
        """
        Try to find the best order of considering the elements by sorting indices among these two children genomes
        that place highly frequent letter locations first.
         """

        multi = np.add(self.frequencies[e0], self.frequencies[e1])
        sorted_indices = np.argsort(multi)[::-1]
        return sorted_indices

    def calculate_score_increase(self, children_ingredients, new_pair):
        """
        We want to maximize the total frequency of letters that are coming from the same parent

        Score increase is calculated by summing the letter frequencies with all elements of the children (that have
        already been selected)
        """
        # Score increase if we don't swap the elements
        score_no_swap = 0
        score_with_swap = 0
        for element in children_ingredients[0]:
            # Calculate the score for the elements in children coming from the first parent
            score_no_swap += self.bi_frequencies[element][new_pair[0]]
            score_with_swap += self.bi_frequencies[element][new_pair[1]]
        for element in children_ingredients[1]:
            # Calculate the score for the elements in children coming from the second parent
            score_no_swap += self.bi_frequencies[element][new_pair[1]]
            score_with_swap += self.bi_frequencies[element][new_pair[0]]
        return score_no_swap, score_with_swap


if __name__ == "__main__":
    problem = QAP(*read_qaplib("../instances/qap/bur26a.dat"))
    cr = FrequencyBasedCrossover(problem, 0.2)
    x = Solution(np.arange(26))
    y = Solution(np.arange(26)[::-1])
    indices = [0, 1, 2]
    for i in range(10):
        x, y = cr.greedy_crossover(indices, x, y)
