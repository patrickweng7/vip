**Team Meeting Notes:**
* With genetic algorithms, each new generation is created through mating/mutation of individuals in the process.
Keywords:
		○ Individual: One specific candidate in the population
		○ Population: group of individuals whose properties will be altered.
		○ Objective: a value used to characterize individuals that you are trying to maximize or minimize
		○ Fitness:  relative comparison to other individuals.
		○ Evaluation: a function that computes the objective of an individual. 
		○ Selection: represents 'survival of the fittest'; gives preference to better individuals therefore allowing them to pass on their genes.
			§ Fitness Proportionate: Great fitness value means higher the probability of being selected for mating.
			§ Tournament: Several tournaments among individuals; winners are selected for mating.
		○ Mate/Crossover: represents mating between individuals
		○ Mutate: introduces random modifications; purpose is to maintain diversity
		○ Algorithms: various evolutionary algorithms to create a solution or best individual
			§ Randomly initialize population
			§ Determine fitness of population
			§ Repeat….
				□ Select parents from population
				□ Perform crossover on parents creating population
				□ Perform mutation of population
				□ Determine fitness of population
			§ Until best individual is good enough.


