# Aditi Prakash
Name: Aditi Prakash  
Email: aprakash86@gatech.edu, Cell Phone: 704-794-3924  
Interests: Machine Learning, Data Science, Software Development, Dance, Reading

# Week 1 : August 25th, 2021
## Overview
Discussed course format (10-week bootcamp followed by joining a sub-team), location of course GitHub and reference materials (https://github.gatech.edu/emade/emade/), expectations, Assignment 1, and notebooks. Attended lecture on Genetic Algorithms.

## Team Meeting Notes
### Lecture on Genetic Algorithms
Introduced concept of genetic algorithms that mimic evolutionary processes (mutation, selection, mating, fitness evaluation, reproduction, etc.) in order to maximize the fitness of individuals in a population of data. Identified steps of a genetic algorithm:
1. Random initialization of population.
2. Determining objective of population: how do we define performance of individuals?
3. Determining fitness of population: how does an individual's objective compare to that of others?
4. Subject individuals to selection methods (ex. fitness proportionate and tournament selection) so as to give preference to the fittest individuals in the population.  
5. Through an evolutionary loop, select parents from population, perform crossovers/mutations/selections on parents and save these modifications as offspring of the initial population, and determine the fitness of the population. Repeat until we maximize the fitness of the best individual in the population. 

Learned genetic algorithm solution to One Max Problem - a simple problem that presents the goal of maximizing the number of 1's that an individual contains (thereby maximizing the sum of the individual's values). 

## Lab 1 - Genetic Algorithms with DEAP
* Installed Conda, Python, and Jupyter Notebooks
* Cloned emade and reference-material repositories using Git 
### Lecture 1 - GA Walkthrough (introductory notebook for understanding of DEAP implementation of genetic algorithms)
* Installed DEAP using pip
* Imported base, creator, and tools libraries from DEAP
* Created FitnessMax Class to track objectives for individuals in One Max problem 
* Set weights attribute to have a value of 1.0 - our goal is to maximize this value for a given individual through the evolution process

Created
* Individual class which inherits from list and has fitness attribute
* Binary random choice generator attr_bool using the DEAP toolbox to randomly present either a 0 or 1 for each value in the list for an individual
* individual() method to create a list of 100 randomly generator 0's and 1's for each individual and registered with DEAP toolbox
* population() method to create a set of individuals

Defined evaluation function for fitness: a sum operation across all of an individual's values.

Performed
* in-place two-point crossover on individuals
* in-place mutation with a given probability of mutation on individuals

This notebook provided a solid introduction to the DEAP API and the representation of genetic algorithms in a high-level language like Python. While the lab itself presented a more in-depth example of the evolutionary process for more challenging optimization problems (like the n-queens problem), the information in this initial notebook will generalize well to future genetic algorithms problems.  

### Lab 1 - Genetic Algorithms with DEAP
This lab explored the One Max problem and the n-queens problem and defined genetic algorithms to solve both. 

One Max Problem:
For this problem, we followed many of the same steps that appeared in the Lecture 1 Notebook (see above). We define a main() function for the genetic algorithm, which evaluates the full population and initiates the evolutionary loop. Within the evolutionary loop, we select individuals for each successive generation, clone them, and perform mutations/crossovers on them. We then evaluate the fitness of these offspring and replace the existing population with the offspring. Finally, we return the fitnesses of the individuals (based on the predefined fitness operation - the sum of the individual's entries) and print statistics such as the mean fitness, squared sum of the fitnesses, and standard deviation of the fitnesses). We loop for some number of generations (40, in this case) and report the best individual that has resulted from this evolution process. Within the DEAP framework, we used libraries like creator (including the create() method), tools (including the selBest() method and the selTournament, mutFlipBit, and cxTwoPoint attributes), and base (including the Toolbox(), register(), select(),  mate(), and mutate() methods).

Findings: The global maximum (a best individual with a fitness equal to n, the number of entries in each individual) was reached within 40 generations about every 19 out of 20 times the algorithm was run; this indicates that our algorithm has an effectiveness of around 95%. Further improvements can be made by changing the bounds of the random number generation for crossover, mutation, and selection.  

![One Max Generations, Part 1](https://picc.io/pok5sgG.png)
![One Max Generations, Part 2](https://picc.io/ouFv77h.png)

N Queens Problem:
For this problem, we followed many of the same steps that appeared in the One Max Problem (see above). We define a size n = 25 for each individual and define a weight of -1.0 here, since we wish to minimize the number of conflicts between queens in our problem space. We then define a permutation function to populate the entries for each individual with numbers selected without replacement from range(n). We define our evaluation function as a measure of the number of conflicts along each diagonal of our board; with the creation process we defined for individuals, queens will not appear in the same row or column. [Describe evaluation function modification here w/ screenshots]. We then create the cxPartialyMatched() function for partially matched crossover, cxTwoPoint(), and mutShuffleIndexes() to shuffle values at different indexes within each individual (since we must remain within size n  = 25). We then modified the mutation function to be a uniform int mutation, wherein randomly selected entries for each individual are replaced with a randomly selected value between 0 and n. The improvements seen with this new mutation function are described in the Findings section below. Finally, we run a similar evolutionary loop as the one described for the One Max Problem (see above) for 100 generations, return the fitnesses of the individuals (based on the predefined fitness operation - the number of conflicts between queens) and print statistics. We loop for some number of generations (100, in this case) and report the best individual that has resulted from this evolution process. 

Findings:
![N Queens Generations, Part 1](https://picc.io/UzJTkn-.png)
![N Queens Generations, Part 2](https://picc.io/BAhG-pn.png)
[Tweak parameters and screenshots with improvement]

Visualizations:

With Shuffle Indexes Mutation:
![N Queens Visualization](https://picc.io/-qpvzmX.png)

With Uniform Int Mutation:
![N Queens Visualization with Uniform Int Mutation](https://picc.io/e1uHhHm.png)

We can see here that the average fitness value decreased much more quickly with the Uniform Int mutation than the Shuffle Indexes mutation. 

With 85 Generations and 0.1 Mutation Rate (Shuffle Index Mutation):
![N Queens Visualization with 85 Generations and 0.1 Mutation Rate](https://picc.io/MZtm5UD.png)

[Further improvements] 