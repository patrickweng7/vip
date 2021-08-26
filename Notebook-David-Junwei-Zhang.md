# August 25, 2021

**Team Meeting Notes**

Lecture 1: Genetic Programming Introduction

Algorithm idea: each generation is created through mating/mutation of individuals in the previous population. Through numerous operations of this process, the fitness of the population will be optimized.

**Keywords and Concepts**

* Individual vs Population 

* Objective: performance measure of an individual (goal is to maximize objective) 

* Fitness: relative measure of performance in comparison to other individuals 

* Selection: represents “survival of the fittest” 

* Fitness Proportionate: the greater fitness value, the higher probability of being selected for mating 

* Tournament: several tournaments among individuals determine selection 

* Mate/Crossover: represents mating between individuals 

    * ie: Single point, double point crossover

* Mutation: introduces random modifications (to maintain diversity) 

    * ie: Bit flip 

 

**Genetic Algorithm Steps**

1. Randomly initialize population 

2. Determine fitness 

3. Repeat until fitness is good enough 

    a. Select parent from population 

    b. Perform mating/crossover 

    c. Mutation 

    d. Determine new fitness 

 

*Types of problem that are good for genetic programming: problems with highly discontinuous, large search spaces 

 