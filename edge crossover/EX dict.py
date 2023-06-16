# Imports
import gzip # as some instance files may have been compressed

# Re-import dependencies (in case earlier import was skipped)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from permutationsga.ga import (
    ConfigurableGA,

    RandomPermutationInitialization,
    crossover_ox,
    crossover_cx,
    crossover_pmx,
    TournamentSelection,
    FunctionBasedRecombinator,
    SequentialSelector,
    generate_uniform_indices,
    generate_sequential_indices,
    generate_sequential_wrapping_indices,

    RandomUniformInitialization,
    DifferentialEvolutionRecombinator,
    
    
)
from permutationsga.problem import IdenticalDecoder, InvPermDecoder, RandomKeysDecoder

from permutationsga.qap import QAP, read_qaplib


from permutationsga.problem import Solution
from string import ascii_lowercase as letters

import json
from typing import Dict

epsilon = 10e-6


# The bur* instances are those that we will be using - note that we are only using 26 keys in this case.
problem = QAP(*read_qaplib("./instances/qap/bur26a.dat"))



def print_layout(l):
    l = [letters[i] for i in l]
    print(" ".join(l[:10]))
    print(" ".join(l[10:19]))
    print(" ".join(l[19:]))


# Add the decoder - permutation encoding
problem = IdenticalDecoder(problem)   # Identity, if using the permutation directly
# problem = InvPermDecoder(problem)     # Inverse, if you want to reverse the direction in which the mapping occurs
# problem = RandomKeysDecoder(problem)            # Random Keys decoder, if our representation uses random keys




# GA - Permutation
seed = 42
population_size = 2**10
rng = np.random.default_rng(seed=seed + 1)
l = problem.get_length()


results         = {}
results_avgs    = {}



def crossover_ex(indices, s0: Solution, s1: Solution):
    global l

    assert s0.e is not None, "Ensure solution s0 is initialized before use."
    assert s1.e is not None, "Ensure solution s1 is initialized before use."

    adjacency_matrix: Dict[int, set] = {}

    # init adjacency matrix
    for i in range(l):
        adjacency_matrix[i] = set()

        for e in [s0.e, s1.e]:
            idx = np.where(e == i)[0][0]

            for new_idx in [idx -1, idx +1]:
                if 0 <= new_idx < l:
                    adjacency_matrix[i].add(e[new_idx])
    

    def shortest_adjacency_len(idx):
        shortest = []
        best_len = float("+inf")

        for new_idx in adjacency_matrix[idx]:
            element_len = len(adjacency_matrix[new_idx])

            if element_len == 0:
                continue

            if element_len < best_len:
                shortest = [new_idx]
                best_len = element_len

            elif element_len == best_len:
                shortest.append(new_idx)
        
        if not shortest:
            return
        
        return np.random.choice(shortest)


    idx     = np.random.randint(l)
    r0      = [idx]
    r0_set  = set()

    try:
        while adjacency_matrix != {}:

            new_idx = shortest_adjacency_len(idx)

            if new_idx is None:
                break

            r0.append(new_idx)
            r0_set.add(new_idx)

            for s_idx in list(adjacency_matrix.keys()):
                s: set = adjacency_matrix[s_idx]
                if idx in s:
                    s.remove(idx)
                
                    if len(s) == 0 and s_idx not in r0_set:
                        r0.append(s_idx)
                        r0_set.add(s_idx)
            
            idx = new_idx

    
    except ValueError:
        for l in adjacency_matrix.values():
            if l:
                raise Exception()
            
    assert len(np.unique(r0)) == 26

    return [Solution(np.array(r0)), Solution(np.array(r0))]


setups = {
    # "PMX sequential indices"    : (crossover_pmx,   lambda: generate_sequential_indices(rng, l)),
    # "PMX uniform indices"       : (crossover_pmx,   lambda: generate_uniform_indices(rng, l, 0.5)),
    # "OX sequential indices"     : (crossover_ox,    lambda: generate_sequential_indices(rng, l)),
    # "CX rng indices"            : (crossover_cx,    lambda: rng.integers(0, l - 1, size=1)),
    # "CX uniform indices"        : (crossover_cx,    lambda: generate_uniform_indices(rng, l, 0.05)),
    "EX"                        : (crossover_ex,    lambda: rng.integers(0, l - 1, size=1))
}




# crossover_fn = crossover_pmx; indices_gen = lambda: generate_sequential_indices(rng, l)
# crossover_fn = crossover_pmx; indices_gen = lambda: generate_uniform_indices(rng, l, 0.5)
# crossover_fn = crossover_ox; indices_gen = lambda: generate_sequential_indices(rng, l)
# crossover_fn = crossover_cx; indices_gen = lambda: rng.integers(0, l - 1, size=1)
# crossover_fn = crossover_cx; indices_gen = lambda: generate_uniform_indices(rng, l, 0.05)

for setup_name, (crossover_fn, indices_gen) in setups.items():
    print(setup_name)

    for n_repeats in range(5):

        initialization = RandomPermutationInitialization(l)
        parent_selection = SequentialSelector()
        recombinator = FunctionBasedRecombinator(
            indices_gen,
            crossover_fn,
            parent_selection,
            population_size * 2, # Note: double as we are including the previous population
            include_what="population"
        )
        selection = TournamentSelection()
        ga = ConfigurableGA(
            seed, population_size, problem, initialization, recombinator, selection
        )



        # ga.generation()

        # print_layout(ga.population[0].e)
        # print()
        # print_layout(ga.population[1].e)


        for i in range(50):
            print(" " * 30, end="\r")
            print(f"\t{n_repeats + 1} {i + 1}", end="\r")
            ga.generation()

            worst   = max(s.f for s in ga.population)
            best    = min(s.f for s in ga.population)

            # # Current best & worst
            # print("Worst:", worst)
            # print("Best: ", best)


            if abs(worst - best) < epsilon:
                break

        if setup_name not in results:
            results[setup_name] = {
                "n_iter"    : [],
                "best"      : [],
                "worst"     : [],
            }

        results[setup_name]["n_iter"].append(i)
        results[setup_name]["best"].append(best)
        results[setup_name]["worst"].append(worst)
    

    worst   = round(np.mean(results[setup_name]['worst']), 2)
    best    = round(np.mean(results[setup_name]['best']), 2)


    results_avgs[setup_name] = {
        "n_iter"    : np.mean(results[setup_name]['n_iter']),
        "best"      : best,
        "worst"     : worst,
        "diff"      : round(worst - best, 2)
    }


    print(f"\tFinished after {results_avgs[setup_name]['n_iter']} generations")

    # Current best & worst
    print("\tWorst:", results_avgs[setup_name]['worst'])
    print("\tBest: ", results_avgs[setup_name]['best'])

    print()

print(json.dumps(results_avgs, indent=4))