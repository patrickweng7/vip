'''Name:''' Jessi Runjie Chen

'''Email:''' rchen417@gatech.edu

'''Cell Phone:''' 703-627-6826

'''Interests:''' Artificial intelligence, Machine Learning, Cybersecurity, Python, Java, Snowboarding, Anime 

= Fall 2021 =
== Week 1: August 25th - September 1st ==
![](https://github.gatech.edu/rchen417/Pictures/blob/master/Screen%20Shot%202021-09-08%20at%205.01.18%20PM.png)

'''August 25, 2021 Team Meeting Notes:'''
* Generic Algorithms Lecture:
** Goal: keep repeating the generation cycle overtime to try to find the best solution
** Advantages of Genetic Algorithms:
*** Good for when the search base is large, discontinuous, and largely non-linear
** Objective vs. fitness:
*** For example, the objective could be the score a student receives on an exam, and fitness is how that score fits on a curve, in comparison to all other students
** Selection:
*** Fitness proportionate: The greater the fitness value, the higher the probability of being selected for mating
*** The lowest fitness value has a probability to be selected
** Tournament: 
*** Several tournaments among individuals
*** Winners are selected for mating
*** The lowest fitness value will never be selected because it will just lose in the tournament
** Mate/Crossover: 
*** Mating between individuals
*** Single point: swapping DNA between the two individuals at a single point
*** Double point: swapping DNA between the two individuals at two points
** Mutate: introduces random modifications to maintain diversity
** Algorithms: various evolutionary algorithms to create a solution or best individual
*** Step 1: Randomly initialize population
*** Step 2: Determine fitness of the population
*** Step 3: Repeat the loop...
**** Select parents form population
**** Perform crossover on parents
**** Perform mutation
**** Determine fitness
*** End the loop until the best individual is good enough

* Python notebook setup
** Find labs on Github -> raw -> save with type .ipynb
** Open the saved lab file with Jupyter notebook


'''Action Items:'''
{| class="wikitable"
!Task
!Current Status
!Date Assigned
!Suspense Date
!Date Resolved
|-
|Install Anaconda and learn how to use Jupyter notebook
|Completed
|August 25, 2021
|September 1, 2021
|August 28, 2021
|-
|check out team GitHub and wiki page for note-taking examples, rubrics, and format
|Completed
|August 25, 2021
|September 1, 2021
|August 25, 2021
|-
|Finish GA Walkthrough and take notes on thoughts and results
|Completed
|August 25, 2021
|September 1, 2021
|August 30, 2021
|-
|Finish Lab 1 and take notes on results and thoughts
|Completed
|August 25, 2021
|September 1, 2021
|August 31, 2021
|}

'''GA Walkthrough'''
* Downloaded Anaconda and loaded the walkthrough python notebook using Jupyter
* This walkthrough utilizes the DEAP python package, DEAP stands for Distributed Evolutionary Algorithms in Python, it is useful for rapid prototyping and testing of ideas
* The FitnessMax class keeps track of the desired objective that we want to achieve, in this case, it is a fitness score of 1
* The individual class represents each individual in the population, it inherits a list and has fitness as an attribute
* Toolbox.attr_bool() generates the random 1s and 0s that make up the genome, running the function correctly yields either a 1 or a 0
* Calling toolbox.individual() will generate an individual instance that has a genome of 100 randomized 1s and 0s
* Calling toolbox.population(n) will generate n individuals with genomes of 100 randomized 1s and 0s
* evalOneMax(individual) evaluates the individual’s fitness level, in this case, it’s the sum of the 1s and 0s of its genome
** My individual 1 has a score of 46.0
** My individual 2 has a score of 49.0
* tools.cxTwoPoint(parent1, parent2) returns two children from parent1 and parant2
** My child 1 has a score of 44.0
*** Child 1 has a worse score than both parents
*** Child 1 deviates from its first parent at 14 different bits
** My child 2 has a score of 51.0
*** Child 2 has a better score than both parents
* tools.mutFlipBit(child, indpb=probablility) flips bits in the child’s genome sequence with a certain probability, the purpose of this is to add mutation
** After mutation, my mutated child differs from the unmutated child at 4 different bits

'''Lab 1 - Genetic Algorithms with DEAP'''
* One Max Problem
** Purpose
*** This lab solves the One Max Problem, which is a simple genetic algorithm problem with the goal of finding an individual with the best fitness
** Background/context
*** A population of individuals will be generated
*** Each individual will have a genome sequence with a length of 100 and made up of arbitrary 1s and 0s
*** The individual whose sequence contains all 1s will be considered to have the best fitness and finding such an individual is the goal of this lab
*** The fitness score of each individual is calculated by summing its genome sequence, therefore, individuals with more 1s will have a higher fitness score. The maximum score is 100
*** “Mate”, “mutate”, and “select” are methods used to evolve and produce individuals with better fitness scores
**** Mate is a 2-point crossover function, that takes in parent individuals to produce children
**** Mutate is flipping bits arbitrarily with a probability of 5%
**** Select is a tournament of 3 individuals who will be competing against each other to find the individual that has the best fitness score out of the 3
** Experiment
*** A population of 300 individuals is created
*** Each of the 300 individuals is evaluated to obtain their fitness score
*** Then, the evolution process of 40 generations is initiated
*** During the evolution process, offsprings are first selected by tournament, and then mated with a 50% probability and mutated with a 20% probability
*** Then, the population is replaced by the modified offspring, whose fitness scores are re-evaluated
*** Finally, statistics are printed to find out who the best individual is
** Results
*** The objective was reached in Generation 31
*** The average fitness value of the population got higher and higher as the number of generations increase
*** After running the code several times, there was an instance where the objective of a max fitness score of 100 wasn’t achieved in 40 generations, the max score was 99. This makes sense because due to the varied factors that contribute to the selecting, mating, and mutating process, the result has certain unpredictability and will not always achieve the desired value

* The N Queens Problem
** Purpose
*** Place n queens on an nxn chessboard in a way such that no queen can be taken by one another.
** Background/context
*** Default n = 20
*** The weight of the fitness objective is -1.0, because we are minimizing an objective instead of maximizing an objective as seen on the One Max problem. This is because we want to minimize the conflict between two queens on the chessboard
*** Individuals are defined by a list of integers, with each integer defining the column number of each queen, and the row number of each queen is defined by the indices of the corresponding integers. 
** Experiment
*** A population of 300 individuals is generated (n = 300)
*** 100 generations are initiated
*** Generations are evolved by selecting, mating, and mutating
*** statistics are printed
** Result
*** After 100 generations, a global minimum of 0 is still not reached (the best fitness score is 1 for most of the time
*** I implemented a new mutation function, that once called upon, will make at least one swap and at most all indices will be swapped. The number of swaps is stored in a variable “swapNum”. The two indices that will be swapped per “swapNum” are determined by two other random function calls. After utilizing this mutation function, the best fitness score was able to consistently achieve 0.
*** I think that my mutation function worked better than the one provided by the lab because it adds more randomization. in the mutate method provided by the lab, each index is looped in increasing order to obtain a chance of it being swapped by another random index, there could be a possibility where no index will be swapped even when the mutate function is called. However, my mutate function will guarantee that at least one swap will be made, and the number of swaps made and the indices swapped are all randomized.

== Week 2: September 1st - September 7th ==

'''September 1, 2021 Team Meeting Notes:'''
* Genetic Programming (GP) Lecture
** Instead of taking an individual and having a function evaluator to obtain objective scores, the individual is the function itself, so that the function gets improved over time
** To represent this, we can use a tree structure
** Nodes are called primitives and represent functions
** Leaves are terminals and represent parameters
** Inputs are at the leaves of the tree
** Output is at the root of the tree
** Crossover in GP
*** Pick random nodes/leafs in the trees participating in the crossover, swap the subtrees of the nodes/leafs
** Mutation in GP
*** Inserting a node or subtree
*** Deleting a node or subtree
*** Changing a node
** Evaluating a tree
*** We can feed a number of input points to the functions to get outputs
*** We can measure the difference between the truth and what was outputted
** Primitives that can make the evolution tree easier
*** power()
*** factorial()
*** sin()
*** cos()
*** tan()
* This is the goal of EMADE, to evolve algorithms and functions so that the best and most efficient choice is selected

'''Action Items:'''
{| class="wikitable"
!Task
!Current Status
!Date Assigned
!Suspense Date
!Date Resolved
|-
|Migrate notes from google docs to GitHub notebooks
|Completed
|September 1, 2021
|September 8, 2021
|September 4, 2021
|-
|Finish Lecture 2 notes
|Completed
|September 1, 2021
|September 8, 2021
|September 4, 2021
|-
|Finish Lab 2 - Genetic Programming (part 1) and take notes
|Completed
|September 1, 2021
|September 8, 2021
|September 8, 2021
|-
|Understand the graphs returned by Lab 2 by asking questions about them during the next team meeting
|In Progress
|September 8, 2021
|September 8, 2021
|
|-
|Find out how to insert pictures in my notebook and if pictures are required by asking questions about them during the next team meeting
|In Progress
|September 8, 2021
|September 8, 2021
|
|}

'''Lab 2 - Genetic Programming (part 1)'''
*Purpose
**This lab focuses on genetic programming which is the tool that will be used for automated algorithm design
***The problem that Genetic programming is trying to solve is to find the best combination of the given primitives in order to get as close as possible to the ideal function and to reach the objectives
*Background/context
**The fitness is created with a weight of -1.0, the individuals are created and represented as tree structures
**A primitive set is initialized and the primitives "add", "subtract", "multiply", "negative" are added to the tree
*** I checked out NumPy documentation and added the primitives "numpy.power" and "numpy.mod" to the tree. They both take two required arguments, so I specified "arity=2" for both
**The toolbox, individual, population, and compiler are defined
**The evaluation function is defined to find the mean squared error between the function of the primitive tree and the function that we are trying to generate
***The goal is to optimize the primitive tree by minimizing this mean squared error
***We already know what the ideal function is and the primitives that it utilizes
***Our primitive tree contains all the primitives that the ideal function utilizes
***we need to find the best combination of these primitives that achieve the ideal function
**The genetic operators are registered
***"evaluate", "select", "mate", "expr_mut" (which is a function that will be passed in as a parameter for the gp.mutUniform method), and "mutate"
***I checked out the DEAP source code and added a new mutation method, gp.mutNodeReplacement, as "mutateNodeReplace"
***gp.mutNodeReplacement replaces a randomly chosen primitive from an individual by a randomly chosen primitive with the same number of arguments from the attributes of the individual
** Experiment
*** A population of 300 individuals is generated (n = 300)
*** 40 generations are initiated
*** During each generation, individuals are evolved by selecting, mating, and mutating in order to create the next generation
*** statistics are printed
** Results
*** After running the experiment with the mutation method provided by the lab default, "mutate", the best individual is: multiply(add(power(x, subtract(x, add(remainder(x, subtract(x, x)), x))), add(power(remainder(subtract(x, x), remainder(x, x)), subtract(power(x, x), x)), x)), add(x, x)), (nan,)
*** After running the experiment with the mutation method I created, "mutateNodeReplace", the best individual is add(add(x, multiply(multiply(x, add(x, multiply(x, x))), x)), multiply(remainder(add(x, multiply(x, power(x, x))), add(x, x)), x)), (nan,)

== Week 3: September 8th - September 14th ==

'''September 8, 2021 Team Meeting Notes:'''

'''Action Items:'''
{| class="wikitable"
!Task
!Current Status
!Date Assigned
!Suspense Date
!Date Resolved
|-
|Insert lab result diagrams for the completed labs in my notebook
|In Progress
|September 8, 2021
|September 15, 2021
|
|-
|Add explanations, justifications, and reflections to Lab 2 results
|In Progress
|September 8, 2021
|September 15, 2021
|
|}
 
