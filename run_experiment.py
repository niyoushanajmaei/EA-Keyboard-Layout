import json, os
import numpy    as np
from typing     import Dict
from time       import time
from os.path    import exists

from matplotlib import pyplot   as plt


from configs import EA_Config, Exp_Config, gen_ga



def tolerant_mean(arrs):
    """
    Gets column-wise mean, can deal with rows of different length
    """
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)

def round_to_multiple(n, m, func=round):
    return m * func(n / m)



def run_experiments(setups: Dict[str, EA_Config], exp_cfg: Exp_Config):
    all_fitnesses_overtime = []

    all_iterations      = []
    all_total_times     = []
    all_iter_times      = []


    try:

        for setup_name, cfg in setups.items():

            print(setup_name)

            fitnesses_overtime = [[] for _ in range(exp_cfg.n_experiments)]

            iter_times      = [[] for _ in range(exp_cfg.n_experiments)]
            total_times     = []
            iterations      = []

            for n_repeat in range(exp_cfg.n_experiments):
                ga = gen_ga(cfg)

                stuck_counter = 0

                exp_start_time = time()

                for i in range(exp_cfg.max_gen):
                    iter_start_time = time()

                    ga.generation()

                    fitnesses = [s.f for s in ga.population]

                    worst   = max(fitnesses)
                    best    = min(fitnesses)

                    fitnesses_overtime[n_repeat].append(fitnesses)


                    iter_times[n_repeat].append(time() - iter_start_time)

                    print(" " * 100, end="\r")
                    print(f"\t{n_repeat + 1:02}  -  {i + 1:03}  -  Best: {round(best)}  -  Worst: {round(worst)}", end="\r")


                    if abs(worst - best) < exp_cfg.epsilon:
                        stuck_counter += 1

                        if stuck_counter >= cfg.n_stuck:
                            break
                    else:
                        stuck_counter = 0
                
                iterations.append(i)
                total_times.append(time() - exp_start_time)

            
            avg_iter_times, error_iter  = tolerant_mean(iter_times)
            
            all_fitnesses_overtime.append(fitnesses_overtime)

            all_iter_times      .append(avg_iter_times.tolist())
            all_total_times     .append(np.mean(total_times))
            all_iterations      .append(np.mean(iterations))

            print(" " * 100, end="\r")

            # # Current best & worst
            # print("\tBest: ", avg_best[-1])
            # print("\tWorst:", avg_worst[-1])
            # print("\tDiff: ", avg_worst[-1] - avg_best[-1])

            print()

            if not exists("data/"):
                os.makedirs("data/")
            
            in_progress = f"data/{exp_cfg.exp_name}_in_progress.json"

            with open(in_progress, "wt") as f:
                json.dump([list(setups.keys()), all_fitnesses_overtime, all_iterations, all_total_times, all_iter_times], f, indent=4)
        

        with open(f"data/{exp_cfg.exp_name}.json", "wt") as f:
            json.dump([list(setups.keys()), all_fitnesses_overtime, all_iterations, all_total_times, all_iter_times], f, indent=4)
        
        os.remove(in_progress)
        
    except KeyboardInterrupt:
        pass
        
    return all_fitnesses_overtime, all_iterations, all_total_times, all_iter_times



def to_masked(x: list):
    """
    Converts a multidimensional list of varying sizes to a masked array. Expects a list of 4 dimensions:
    (number of setups/experiments/variations, number of repeat experiments, number of generations, population_size]
    """

    shape = [0, 0, 0, 0]

    shape[0] = max(shape[0], len(x))
    for setup in x:
        shape[1] = max(shape[1], len(setup))
        for exp in setup:
            shape[2] = max(shape[2], len(exp))
            for gen in exp:
                shape[3] = max(shape[3], len(gen))


    masked = np.ma.array(np.zeros(shape), mask=np.ones(shape))

    for setup_i, setup in enumerate(x):
        for exp_i, exp in enumerate(setup):
            for gen_i, gen in enumerate(exp):
                masked[setup_i][exp_i][gen_i][:len(gen)] = gen

    return masked


def plot_boxplots(fitnesses_avg, setup_names, optimal_sol=None, tick_size=5, lowest_val=5400000, highest_val=6300000):
    assert len(fitnesses_avg) == len(setup_names)

    max_gen_all = fitnesses_avg.shape[1]

    for i, setup_name in enumerate(setup_names):
        fig = plt.figure(figsize=(15, 7))

        data = fitnesses_avg[i].T
        
        n_gen = np.max(np.count_nonzero(~data.mask, axis=1))


        if optimal_sol != None:
            plt.axhline(optimal_sol, linestyle=":", label="Optimal Solution")
        plt.boxplot(data, showfliers=False)
        plt.xlim((0, n_gen + 1))
        plt.xticks(range(0, n_gen + tick_size, tick_size), labels=range(0, n_gen + tick_size, tick_size))
        plt.ylim((lowest_val, highest_val))

        plt.ylabel("Fitness")
        plt.xlabel("Generation")
        plt.title(f"Boxplot of different generations for \n{setup_name}")
        plt.legend()

        plt.show()


def plot_inverted_boxplots(fitnesses_avg, setup_names, optimal_sol=None, save_files=None, tick_size=5, lowest_val=5400000, highest_val=6300000):
    assert len(fitnesses_avg) == len(setup_names)
    if save_files != None:
        assert len(setup_names) == len(save_files)

    max_gen_all = fitnesses_avg.shape[1]

    for i, setup_name in enumerate(setup_names):
        fig = plt.figure(figsize=(7, 5))

        data            = fitnesses_avg[i].T
        max_gen         = max(np.count_nonzero(~data.mask, axis=1))

        if optimal_sol != None:
            plt.axvline(optimal_sol, linestyle=":", label="Optimal Solution")
        plt.boxplot(data[:,::-1], showfliers=False, vert=False)

        plt.ylim((max_gen_all - max_gen, max_gen_all + 1))
        plt.yticks(range(max_gen_all, max_gen_all - max_gen - tick_size, -tick_size), labels=range(0, max_gen + tick_size, tick_size))
        
        plt.xlim((lowest_val, highest_val))

        plt.ylabel("Generation")
        plt.xlabel("Fitness")
        plt.title(f"Boxplot of different generations for \n{setup_name}")
        plt.legend(loc='lower right')

        if save_files != None:
            save_file = save_files[i]

            plt.savefig(save_file)

        plt.show()

def plot_best(fitnesses_avg, setup_names, title, optimal_sol=None, legend_title=None):
    assert len(fitnesses_avg) == len(setup_names)

    for i, setup_name in enumerate(setup_names):
        best = np.ma.min(fitnesses_avg[i], axis=(1))
        
        plt.plot(best.tolist(np.NaN), label=setup_name)


    if optimal_sol != None:
        plt.axhline(optimal_sol, linestyle=":", label="Optimal Solution")
    plt.title(f"Best fitness per generation for {title}")
    plt.ylabel("Fitness")
    plt.xlabel("Generation")
    plt.legend(title=legend_title)
    plt.show()


def extract_data(data, setup_idxs, setup_names):
    idxs = [setup_idxs[name] for name in setup_names]
    return np.ma.array(data[idxs])


def gen_save_files(abs_folder, filenames, test_name, graph_type, file_type ="png"):
    save_files = []

    for filename in filenames:
        file_path = f"{abs_folder}/graphs/{test_name}/{graph_type}"
        file_name = filename.lower().replace(" ", "_")
        save_files.append(f"{file_path}/{file_name}.{file_type}")
    
    return save_files