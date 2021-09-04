* Name: Rayan Khoury
* Email: rkhoury7@gatech.edu
* Mobile number: +1(678)789-7927

# **August 25th, 2021**
## **Meeting 1**
### Lecture 1
Generic Algorithm
With genetic algorithm, each new generation is created through mating/nutation of individuals in the previous population (then their fitness is evaluated). Through numerous operations of this process, it will eventually produce the best individual
1.	Individual: One specific candidate in the population(with properties such as DNA)
2.	Population: Group of individuals whose properties will be altered
3.	Objective: a value to characterize individuals that you are trying to maximize or minimize (usually the goal is to increase objective through the evolutionary algorithm
4.	Fitness: relative comparison to other individuals; how well does the individual accomplish a task relative to the rest of the population? 
5.	Selection: Represents ‘survival of the fittest’; gives preference to better individuals, therefore allowing them to pass on their genes
	* Fitness Proportionate: The greater the fitness, the greater the probability to be selected for next generation
	* Tournament: Several tournaments among individuals; winners are selected for mating
NB: (You can spin a roulette wheel and select a pool, highest wins)
6.	Mating/Crossover: Represents mating between individuals
7.	Mutate: Introduces random modifications; purpose is to maintain diversity
•	Algorithms: Various evolutionary algorithms to create a solution or best individual
1.	Randomly Initialize population
2.	Determine fitness of population
3.	Repeat:
         1. Select parents from population
         2. Perform crossover on parents creating population
         3. Perform mutation of population
         4. Determine fitness of population
         5. Continue until best individual is found

