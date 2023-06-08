Baseline: 
    - Random Initialization
    - Parent Selection: Random
    - Mutation: random swapping of two key assignments 
    - Crossover: Select the best fitness among the five configuration of OX, PMX and CX: PMX with sequential is doing better for our case, The paper says that PMX with uniform is better for QAP
        - We have a plot

Improvement: 
    - Custom Crossover: 
        - We tried implementing edge crossover but we had some problems
        - Six regions in the keyboard, copy 3 from each parent, then do repair operation
            - Repair operation: keep the repeated key from the parent that has it in the better position, replace the other one at random from the set of remaining characters
    - Custom Mutation: 
        - Keystrokes leading to the middle row have lower time: map higher frequency characters to the middle row
        - We have a code
        

- For next time: 
    - Tune the population size in the baseline, in recombination the population size is very important 
        - GRID SEARCH OR bisection
    - Tune the mutation probability
    - Mutation: 
        - do this for the initialization instead and have a less guided mutation
        - Dont hardcode the frequently used characters but do some analysis to be able to run this for more instances (bur***)
        - Proper initialization is important so maybe change to this to initialization to do this in a short time
        - Good initialization: higher chance that the good pieces are already in the population so the good parts will come together sooner
    - Do evaluation on both time on our machine and say which CPU, and also number of evaluations, normally limited budget
    - Crossover:
        - After picking a key, you might be interested in picking another key in a distance
        You could consider doing mutual information from the matrix.
        - Cluster the keyboard by mutual information
        - In gomea, the mutual information is used as an inverted distance matrix, we already have a distance matrix so we can use it for clustering
        - consider alternative clustering: Either the close ones on the keyboard, or the ones that have lower distance to eachother
        - Do the crossover like cycle crossover because the repair might ruin everything

Things that need to be done: 
- Changing mutation idea to initialization 
- Finding another mutation?
- Fix the crossover with this or another clusterings 
- tune the population size and other parameters for baseline 
- decide on the evaluation method and evaluate baseline
- evaluate the improvements as well 

Clemente: Writes the background section of the report and the skeleton of the report
Davis: Crossover
Nioosha: Change mutation to initialization
