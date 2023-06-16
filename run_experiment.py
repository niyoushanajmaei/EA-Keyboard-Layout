import json
import numpy    as np
from typing     import Dict
from time       import time


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

            with open(f"{exp_cfg.exp_name}.json", "wt") as f:
                json.dump([all_fitnesses_overtime, all_iterations, all_total_times, all_iter_times], f, indent=4)
        
    except KeyboardInterrupt:
        pass
        
    return all_fitnesses_overtime, all_iterations, all_total_times, all_iter_times