# Aditi Prakash
Name: Aditi Prakash  
Email: aprakash86@gatech.edu, Cell Phone: 704-794-3924  
Interests: Machine Learning, Data Science, Software Development, Dance, Reading

# Week 1
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
* Created Individual class which inherits from list and has fitness attribute
* Created a binary random choice generator attr_bool using the DEAP toolbox to randomly present either a 0 or 1 for each value in the list for an individual
* Created individual() method to create a list of 100 randomly generator 0's and 1's for each individual and registered with DEAP toolbox
* Created population() method to create a set of individuals
* Defined evaluation function for fitness: a sum operation across all of an individual's values
* Performed in-place two-point crossover on individuals
* Performed in-place mutation with a given probability of mutation on individuals

This notebook provided a solid introduction to the DEAP API and the representation of genetic algorithms in a high-level language like Python. While the lab itself presented a more in-depth example of the evolutionary process for more challenging optimization problems (like the n-queens problem), the information in this initial notebook will generalize well to future genetic algorithms problems.  

### Lab 1 - Genetic Algorithms with DEAP
* Installed DEAP using pip
* Imported base, creator, and tools libraries from DEAP
* Created FitnessMax Class to track objectives for individuals in One Max problem 
* Set weights attribute to have a value of 1.0 - our goal is to maximize this value for a given individual through the evolution process
* Created Individual class which inherits from list and has fitness attribute
* Created a binary random choice generator attr_bool using the DEAP toolbox to randomly present either a 0 or 1 for each value in the list for an individual
* Created individual() method to create a list of 100 randomly generator 0's and 1's for each individual and registered with DEAP toolbox
* Created population() method to create a set of individuals
* Defined evaluation function for fitness: a sum operation across all of an individual's values
* Performed in-place two-point crossover on individuals
* Performed in-place mutation with a given probability of mutation on individuals

This notebook provided a solid introduction to the DEAP API and the representation of genetic algorithms in a high-level language like Python. While the lab itself presented a more in-depth example of the evolutionary process for more challenging optimization problems (like the n-queens problem), the information in this initial notebook will generalize well to future genetic algorithms problems.  