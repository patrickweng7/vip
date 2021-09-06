•	Introduction to VIP
•	Syllabus 
•	Allowing others to reproduce the work done (documentation)
                                                      
Automated Algorithm Design

GitHub: github.gatech.edu/emade/emade
                                                      
Topic 1: Genetic Algorithms

Each new generation is created through the manipulation/mutation of individuals. Their fitness is then evaluated.

Individuals: One specific candidate in the population (with properties such as DNA)

Population: Group of individuals whose properties will be altered

Objective: a value to characterize individuals that you are trying to maximize or minimize (usually the goal is to increase objective through the evolutionary algorithm

Fitness: relative comparison to other individuals; how well does the individual accomplish a task relative to the rest of the population? 

Selection: Represents ‘survival of the fittest’; gies preference to better individuals, therefore allowing them to pass on their genes
	Fitness Proportionate: The greater the fitness, the greater the probability to be selected for next gen
	Tournament: Several tournaments among individuals; winners are selected for mating

Fitness proportionate: When fitness value goes up, the higher the probability that he gets picked.

Tournament: Several tournaments – winners are selected for mating.

Mating/ Crossover: Taking 2 or more individuals and exchanging the DNA between them.

Mutate: Random modifications (The goal is to maintain diversity)

•	Algorithms: Various evolutionary algorithms to create a solution or best individual
1.	Randomly Initialize population
2.	Determine fitness of population
3.	Repeat
	Select parents from population
	Perform crossover on parents creating population
	Perform mutation of population
	Determine fitness of population
	Continue until best individual is found


                                              One Max Problem-Example Output

Results: Overtime, through the evolution, we get to the point where the vectors are full of 1s

                                  
Lab 1: Genetic Algorithm with DEAP

 One Max Problem 

The objective of this exercise was to find a bit string containing all 1s with a set length using the DEAP python library. I installed DEAP and imported this library, I had to define the name of the normal and inherited classes. I created my own class. The tournament selection of 3 individuals’ method is important because it let us make sure that more varied traits in this population are present. 

 

After this, we rank each individual of our population according to their fitness. The algorithm is then set to run for 40 generations.

def main():
    pop = toolbox.population(n=300)
    
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

After the crossover on the entire population, we print out the result to check the progress over time:
o	Max increased from 65.0 at generation 0 to 100.0 at generation 39
o	Average increased from 53.82 in generation 0 to 97.81 at generation 39

We can deduce that after running the code many times, one can notice that the optimal maximum expected wasn’t always reached and that should be due to the random nature of initialization, crossover, and mutation.

 The N Queens Problem 

The N Queens is the problem of putting N chess queens on an NxN chessboard such that no two queens attack each other. We use n=20 to create the fitness and individual classes. After that, we define our evaluation function like as below:

def evalNQueens(individual):
    size = len(individual)
    #Count the number of conflicts with other queens.
    #The conflicts can only be diagonal, count on each diagonal line
    left_diagonal = [0] * (2*size-1)
    right_diagonal = [0] * (2*size-1)
    
    #Sum the number of queens on each diagonal:
    for i in range(size):
        left_diagonal[i+individual[i]] += 1
        right_diagonal[size-1-i+individual[i]] += 1
    
    #Count the number of conflicts on each diagonal
    sum_ = 0
    for i in range(2*size-1):
        if left_diagonal[i] > 1:
            sum_ += left_diagonal[i] - 1
        if right_diagonal[i] > 1:
            sum_ += right_diagonal[i] - 1
    return sum_,

In the next step, we define our crossover. We will be facing a partially matched crossover mating. It shows swapping a pair of queens’ positions between 2 parent individuals which is more effective in this scenario. We finally use our mutation function, shown below:

a)	def mutShuffleIndexes(individual, indpb):
b)	    """Shuffle the attributes of the input individual and return the mutant.
c)	    The *individual* is expected to be a :term:`sequence`. The *indpb* argument is the
d)	    probability of each attribute to be moved. Usually this mutation is applied on 
e)	    vector of indices.
f)	    
g)	    :param individual: Individual to be mutated.
h)	    :param indpb: Independent probability for each attribute to be exchanged to
i)	                  another position.
j)	    :returns: A tuple of one individual.
k)	    
l)	    This function uses the :func:`~random.random` and :func:`~random.randint`
m)	    functions from the python base :mod:`random` module.
n)	    """
o)	    size = len(individual)
p)	    for i in range(size):
q)	        if random.random() < indpb:
r)	            swap_indx = random.randint(0, size - 2)
s)	            if swap_indx >= i:
t)	                swap_indx += 1
u)	            individual[i], individual[swap_indx] = \
v)	                individual[swap_indx], individual[i]
w)	    
x)	    return individual,

In this exercise, it is a must to Shuffle indexes because it represents the position of the queens on the chessboard. At the same time, we cannot mutate or duplicate indexes as this might cause a result to be out of bounds. At the end, I defined the loop and ran it for the 100 generations, one can see that we change from a max of 16.0 in generation 0 to 9.0 in generation 99. Min, average, and max significantly decreased when measuring the fitness throughout the generations.

Topic 2: Genetic Programming

Tree Representation is very used:

•	We can represent a program as a tree structure.
o	Nodes are called primitives and represent functions
o	Leaves are called terminals and represent parameters

How is the Tree Stored?

•	The tree is converted to a lisp preordered parse tree

o	Operator followed by inputs.

More examples:

What’s the function?

F(x) = 2 – (0+1) (Note: It’s a constant)

Crossover in GP:

o	Crossover in tree-based GP is simply exchanging subtrees
o	Start by randomly picking a point in each tree
o	The subtrees are exchanged to produce children

Mutation in GP:

o	Mutations can involve: Inserting a node or subtree, deleting a node or subtree, changing a node.

Example: Symbolic Regression

o	Using simple primitives, use genetic programming to evolve a solution to y = sin(x)
o	Primitives include: +, *, -, /
o	Terminals include integers and x
o	How did Calculus 1 solve this?  Taylor Series for sin(x)!

Evaluating a tree:

o	We can feed a number of input points into the function to get outputs 
o	Run f(x)
o	We can measure error between outputs and truth

What Primitives could I use to make this evolution easier?

o	Power()
o	Factorial()
o	Sin()
o	Cos()
o	Tan()

This is the idea behind EMADE.






