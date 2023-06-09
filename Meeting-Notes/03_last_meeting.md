Writing part:
We are done with the introduction.

**We**: about Background, do we include information about related topics?

**Arthur**: Including everything is an overkill, but explaining why we picked a particular baseline, summarize most of the basic things of the literature.


**We**: We found that the initialization doesn't really matter

**Arthur**: Key point - depends on what the goal is. Intelligent initialization helps to 
    1. Converge faster
    2. Use smaller population sizes

**Arthur**: "In my experience without good initialization 5-10min, with good initialization I could bring that down to 5 seconds"
Bad initialization can hurt you, small diversoty and then the GA is stuck in a local optimum. Increase the speedup without computing too much.

**Arthur**: These problems are NP hard: so solving to optimality will take exponential time. But it doesn't necessarily mean that we can't find a good solution in a reasonable amount of time. Schema analysis could be done, however this is something iterative and operations in each step is polynomial. Could be the case that we need exponential number of iterations, but most likely not. 

**We**: Something on continous on random things. Should we add that in our background? 

**Arthur**: If you are not using random keys, don't include it. Just explain the contents of things that you will use your improvement on.

**We**: We got fairly close to the best solution.

**Arthur**: None of the groups have achieved it. Key point- fine-tuning, final push, problem specific knowledge should be used. 


**We**: What do you mean by evaluation of the baseline?
Arthur: I want to see how it converges over time, how it progresses over time. Repeat it a few times due to stochasticity. It reveals if it converges quicly or not.


**We** show the latex file. Explain graphs. Convergence plot, fixed budget, fixed target.

**Arthur**: Number of generations is not a good metric because it depends on the population size. (hints at using function call count)
What do we want to compare. Minimum and maximum fitness values are not good metrics - they are not representative of the population. 
The best fitness is good. What can we expect from a medium run? From a bad run? From a best run? Then plot quantiles - it would indicate how the algorithm is doing?
1 Find the best solution at every point of time.
2. Mean std, stability, quantiles.
The best solution is all we care about - how well does the ea solves my problem?

How many runs do we do?
5 is fine 10 is good 20 is plenty
Consider statistical significance.

**Nioosha**: Explain the results of different initializations. Surprisingly, we get the same results for the different initializations.

**Arthur**: I want to see what the initialization looks like, that sounds very interesting.

**Nioosha**: Shows the results by running script. Finds a bug

**Arthur**: [smiling] it happens. 

**Nioosha**: The results show that the initialization slightly improves the results.

**Arthur**: These results are actually quite impressive. Since your results are already quite good, the improvements will be quite small and hard to find. So it is great you found it. Also, great that you included this hyperparameter. However, I see a slight issue - the randomly initialized bad solutions will be immediately deleted. 

**Arthur**: Wants to express again that it is quite remarkable that the results we just found are closer to optimum than everything we've ever seen. Key point about the optimum - is my approach doing reasonable or gets stuck immediately?

**Nioosha**: Can we make a scale by ourselves? ...

**Arthur** : Of course, if it helps to tell the story, Just don't forget to include the convergence graphs.
Are there more things you want to show me?

**Davis** Crossover but no results.

**Arthur**: Schema theorem from lectures is quite good direction. 
Quick keys - should stay at the same speed category. This is because the parents are already of the good quality
High frequency keys should be taken from the same parent, the parents are highly optimized, (e.g. letter t and h should be taken from the same parent because ). In this way we are hoping for a linkage that is not broken.

If you want to be smart: detect when the particular key combination is a bad one and make sure to avoid it. 

If you're lucky, repair should be used  only a few times.

Deciding on a good crossover is hard but important.

The crossover doesn't have to be ideal, but better than random is good enough.

The entire deal with the GA, we need to look only at the frequencies of letters

Getting to the optimum:
1. Use two different encodings
2. Use a different representation - assign a keybord 

I have a GA I have a problem, I want to see how this works. Benchmark level will be compared ot other teams. Ideas of incoorporating instance-specific knowledge is good. At the end show the analysis of the results. 
Important: ARGUE WHY IMPROVEMENTS ARE GOOD. End result doesn't matter that much. Goal - without using instance-specific information get good resutlts.
If you use time, use the same computer for all the runs.

What are my improvements and how well do they perform?

Make sure to talk about the progress and improvement over time. 

**Parkhar** Can we reach you trhoughout the week?

**Arthur**: No because I am busy. If you reach me on Monday, then I might respond in time.
 
Maybe check the initialization quality specificly with a random initialization, it has not been . Show it in the report.

Good luck with the project and exams!




