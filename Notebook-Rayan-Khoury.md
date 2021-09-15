
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
	1. Fitness Proportionate: The greater the fitness, the greater the probability to be selected for next generation
	2. Tournament: Several tournaments among individuals; winners are selected for mating
NB: (You can spin a roulette wheel and select a pool, highest wins)
6.	Mating/Crossover: Represents mating between individuals
7.	Mutate: Introduces random modifications; purpose is to maintain diversity
Algorithms: Various evolutionary algorithms to create a solution or best individual
1.	Randomly Initialize population
2.	Determine fitness of population
3.	Repeat:
       1. Select parents from population
       2. Perform crossover on parents creating population
       3. Perform mutation of population
       4. Determine fitness of population
       5. Continue until best individual is found

## **Lab 1 – Genetic Algorithm with DEAP**

### a)	One Max Problem
1. 	The objective of this exercise was to find a bit string containing all 1s with a set length using the DEAP python library.
2.	After installing deap and importing this library, we defined the name of the class, the inherited class, and the objectives. We then created a class and defined a tuple that represents a single objective up fir maximization.
3.	The tournament selection of 3 individuals method was used to preserve more varied traits in this population.
4. Then, we evaluate our population according to each individual’s fitness
5. The algorithm is then set to run for 40 generations 
6.	After crossing over the generation and defining statistics for our population, we print out the result to check the progress over time. Max increased from 65.0 at generation 0 to 100.0 at generation 39. In addition, average increased from 53.82 in generation 0 to 97.81 at generation 39.
Thus, we conclude that after running the code multiple times, the generation did not always reach the optimal max and average due to the random nature of initialization, crossover, and mutation.

### b) The N Queens Problem
1.	This exercise revolves around the determination of a configuration of n queens on a nxn chessboard such that no queen can be taken by another.
2.	First, we create the fitness and individual classes (first we use n=20)
3.	Next, we define our crossover. This problem will consist of partially matched crossover mating. It represents swapping a pair of queens’ positions between 2 parent individuals which is more effective in this scenario.
5.	Then, we move into our mutation function
6.	Shuffling indexes in this exercise is crucial as this represents the position of the queens on the chessboard, but cannot mutate or duplicate as this may lead to an out of bounds result.
7.	After defining the loop and running it for 100 generations, we change from a max of 16.0 in generation 0 to 9.0 in generation 99. 
8.	Along with min, and avrg, , max has been gone down according to the following graphs
### Action Items
|Task Description|Current Status|Progress Notes|Date Assigned|Resolved Date|
|---|---|---|---|---|
|Lecture 1 Notes|Complete|NA|08/25/2021|08/29/2021|
|Lab 1 Notes|Complete|NA|08/25/2021|08/29/2021|
