
# We decided to not use this function and turn it into an initialization method instead

def frequency_restricted_mutation(individual: Solution, p: float, k: float):
    """
    
    DON'T USE
    
    Performs a mutation on the individual, with a probability of p or kp, depending on the region and the frequency of the character in English.
    choose 0 < k < 1, 0 < p < 1
    """
    middle_row_indices = [i for i in range(10, 19)]
    high_frequency = [5, 1, 14, 18, 9, 15, 20, 19, 12]  # chosen from marginal frequencies of the bigram analysis
    for loc in range(len(individual.e)):
        swapped_indices = []
        char = individual.e[loc]
        if (loc in middle_row_indices and char in high_frequency) or (loc not in middle_row_indices and char in high_frequency):
            bound = p*k
        else:
            bound = p
        if rng.random() < bound:
            while True:
                swap_index = rng.integers(0, len(individual.e), size=1)
                if swap_index != loc and swap_index not in swapped_indices:
                    break
            individual.e[loc], individual.e[swap_index] = individual.e[swap_index], individual.e[loc]
            swapped_indices.append(swap_index)
    return 
    