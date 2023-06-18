# Keyboard Optimization & Permutations

In our project, we focus on the keyboard optimization problem. 
The main goals of this project are to analyze the problem structures that can be exploited by current representation and crossover operators, identify opportunities for further improvement using problem-specific knowledge, and assess the performance of our proposed enhancements. 

Firstly, we develop a specialized crossover operator tailored specifically for the keyboard layout optimization problem. Additionally we propose two custom initialization algorithms
to optimize the performance and convergence.

To establish a reliable baseline, we carefully select a benchmark
promises great performance for QAP problems without bothering with domain specific knowledge. The chosen baseline should
demonstrate progress over time, avoid premature convergence, and
present a challenging target to surpass. This selection ensures a
fair evaluation of our proposed optimizations.


## Custom Initialization
[Frequency-based custom initialization](our_crossover_not_working/frequency_based_crossover.py) 

## Custom Crossovers
### 1. Linkage-based Crossover
[Linkage-based crossover](our_other_crossovers_working/linkage_based_crossover.py)

### 2. Region-based Crossover
[Region-based crossover](our_other_crossovers_working/region_based_crossover.py)

## Running the experiments
[File used for testing the initialization](initialization_test.ipynb)

[File used for testing the crossovers](crossover_test.ipynb)

[File used for running the final experiments](plot_run_results.ipynb)