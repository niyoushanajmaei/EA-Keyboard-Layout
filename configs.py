import numpy    as np

from string import ascii_lowercase as letters


from permutationsga.problem import IdenticalDecoder, InvPermDecoder, RandomKeysDecoder
from permutationsga.qap     import QAP, read_qaplib
from permutationsga.ga      import (
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



# The bur* instances are those that we will be using - note that we are only using 26 keys in this case.
problem = QAP(*read_qaplib("./instances/qap/bur26a.dat"))

def print_layout(l):
    l = [letters[i] for i in l]
    print(" ".join(l[:10]))
    print(" " + " ".join(l[10:19]))
    print("  " + " ".join(l[19:]))



# GA - Permutation
seed    = 42
rng     = np.random.default_rng(seed=seed + 1)
l       = problem.get_length()


class EA_Config:
    def __init__(
            self, pop_size=2**10,   decoder=None,   crossover_fn=crossover_pmx, indices_gen=None, 
            initialization=None,    selection=None, parent_selection=None,      n_stuck=None) -> None:
        
        self.population_size    = pop_size
        self.crossover_fn       = crossover_fn


        if decoder == None:
            decoder = IdenticalDecoder(problem)
        self.decoder = decoder

        if indices_gen == None:
            indices_gen = lambda: generate_sequential_indices(rng, l)
        self.indices_gen = indices_gen

        if initialization == None:
            initialization = RandomPermutationInitialization(l)
        self.initialization = initialization

        if selection == None:
            selection = TournamentSelection()
        self.selection = selection

        if parent_selection == None:
            parent_selection = SequentialSelector()
        self.parent_selection = parent_selection
    
        if n_stuck == None:
            n_stuck = 3
        self.n_stuck = n_stuck


def gen_ga(cfg: EA_Config):
    recombinator = FunctionBasedRecombinator(
        cfg.indices_gen,
        cfg.crossover_fn,
        cfg.parent_selection,
        cfg.population_size * 2, # Note: double as we are including the previous population
        include_what="population"
    )

    return ConfigurableGA(
        seed, cfg.population_size, cfg.decoder, cfg.initialization, recombinator, cfg.selection
    )


class Exp_Config:
    def __init__(self, exp_mame, max_gen=100, n_experiments=5, epsilon=10e-6) -> None:
        self.exp_name       = exp_mame
        self.max_gen        = max_gen
        self.n_experiments  = n_experiments
        self.epsilon        = epsilon