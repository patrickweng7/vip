
***
* Introduction to VIP
* Syllabus 
* Allowing others to reproduce the work done (documentation)

* Phone number : 4049368747
* Personal e-mail : rayan.dabbagh@gmail.com
                                                      
# **Automated Algorithm Design**

_GitHub: github.gatech.edu/emade/emade_

### Objective of lecture 3:

* Recognize the power of multiple objective optimization in supplying a population of solutions not just a single objective.

_Go ahead and rate your skills in ML and Python (used for group forming)_

**Questions of the day:** What are you looking for in a data/mate? What is an algorithm looking for in a mate?

* _Gene Pool:_ Set of genome to be evaluated during current generation
    * _Genome_

        * Genetic description of an individual
        * DNA
        * GA = set of values
        * GP = tree structure, string

    * _Search Space_

        * Set of all possible genomes
        * For automated algorithm design -> set of all possible algorithms

* The evaluation of genome associates a genome with a set of scores

    * _Scores:_

        * True positive or TP: How often is the desired objective identified
        * False positive or FP: How often is something else than the desired object identified

    * _Objectives_:

        * Set of measurements each genome is scored against
        * Phenotype 

* _Objective Space:_ set of objectives

* _Evaluation:_ Maps an in individual or genome.

    * From a location in search space: Genotypic description
    * To a location in objective space: Phenotype description
* _Classification Measures:_

    * A data set made of positive and negative samples is inserted in a classifier which gives out one of the following results:
        
1- _Actual Positive:_

    * True Positive (TP) with the TP rate or TPR = TP/P = TP/(TP+FN)
    * False Negative (FN) with a FN rate or FNR = FN/P = FN/(TP+FN)
        
2- _Actual Negative:_

    * False Positive (FP) with a FP rate or FPR = FP/N = FP/(FP+TN)
    * True Negative (TN) with a TN rate or TNR = TN/N = TN/(TN+FP)

* _Other measures include:_

    * Precision or Positive Predictive Value (PPV): PPV = TP/(TP+FP) -> Bigger is better
    * False Discovery Rate (FDR): FDR = FP(TP+FP) = 1 - PPV -> Smaller is better
    * Negative Predictive Value (NPV): NPV = TN/(TN+FN) -> Bigger is better
    * Accuracy (ACC): ACC = (TP+TN)/(P+N) = (TP+TN)/(TP+FP+FN+TN) -> Bigger is better

* _Pareto Optimality:_

    * An individual is Pareto optimal if there is no individual in the population that outperforms this individual on all objectives
    * The set of all Pareto individuals is known as the Pareto Frontier
    * These individuals represent unique contributions
    * We want to drive selection by favoring Pareto individuals but maintain diversity by giving all individuals some probability of mating.

* _Strength Pareto Evolutionary Algorithm:_

    * Each individual is given a strength S (S is how many others in the population it dominates)
    * Each individual receives a rank R (R is the sum of S's of the individuals that dominate it)
    * A distance to the kth nearest neighbour is calculated and a fitness of R+1/(𝛔k + 2) is obtained

**Notebook self evaluation:**

The scores are the following:

* Notebook Maintenance: 25/25
* Meeting notes: 15/15
* Personal Work & accomplishments: 35/35
* Useful resource: 24/25

**The overall grade is a 99/100**

## Topic 2: Genetic Programming

**Tree Representation is very used:**

    * We can represent a program as a tree structure.
    * Nodes are called primitives and represent functions
    * Leaves are called terminals and represent parameters.

**How is the Tree Stored?**

    * The tree is converted to a lisp preordered parse tree
    * Operator followed by inputs.

**More examples:**

**What’s the function?**

    * F(x) = 2 – (0+1) (Note: It’s a constant)

**Crossover in GP:**

    * Crossover in tree-based GP is simply exchanging subtrees
    * Start by randomly picking a point in each tree
    * The subtrees are exchanged to produce children

**Mutation in GP:**

    * Mutations can involve: Inserting a node or subtree, deleting a node or subtree, changing a node.

**Example: Symbolic Regression**

    * Using simple primitives, use genetic programming to evolve a solution to y = sin(x)
    * Primitives include: +, *, -, /
    * Terminals include integers and x
    * How did Calculus 1 solve this?  Taylor Series for sin(x)!

**Evaluating a tree:**

    * We can feed a number of input points into the function to get outputs 
    * Run f(x)
    * We can measure error between outputs and truth

**What Primitives could I use to make this evolution easier?**

    * Power()
    * Factorial()
    * Sin()
    * Cos()
    * Tan()

**This is the idea behind EMADE.**

### Lab 2:

                                                      
## **Topic 1: Genetic Algorithms**

### Each new generation is created through the manipulation/mutation of individuals. Their fitness is then evaluated.

**Individuals:** One specific candidate in the population (with properties such as DNA)

**Population:** Group of individuals whose properties will be altered

**Objective:** A value to characterize individuals that you are trying to maximize or minimize (usually the goal is to increase objective through the evolutionary algorithm)

**Fitness:** Relative comparison to other individuals; how well does the individual accomplish a task relative to the rest of the population? 

**Selection:** Represents ‘survival of the fittest’; gives preference to better individuals, therefore allowing them to pass on their genes

   _1.    Fitness Proportionate:_ The greater the fitness, the greater the probability to be selected for next generation

   _2.    Tournament:_ Several tournaments among individuals; winners are selected for mating NB: (You can spin a roulette wheel and select a pool, highest wins)

**Mating/Crossover:** Taking 2 or more individuals and exchanging the DNA between them.

**Mutate:** Random modifications (The goal is to maintain diversity)

**Algorithms:** Various evolutionary algorithms to create a solution or best individual
1. Randomly Initialize population
2. Determine fitness of population
3. Repeat

    * Select parents from population
    * Perform crossover on parents creating population
    * Perform mutation of population
    * Determine fitness of population
    * Continue until best individual is found

### One Max Problem-Example Output                                       

**Results:** Overtime, through the evolution, we get to the point where the vectors are full of 1s

                                  
### Lab 1: Genetic Algorithm with DEAP

                                                       One Max Problem

The objective of this exercise was to find a bit string containing all 1s with a set length using the DEAP python library. I installed DEAP and imported this library, I had to define the name of the normal and inherited classes. I created my own class. The tournament selection of 3 individuals’ method is important because it let us make sure that more varied traits in this population are present. 

 
After this, we rank each individual of our population according to their fitness. The algorithm is then set to run for 40 generations.

`def main():`
    `pop = toolbox.population(n=300) `
    `# Evaluate the entire population`
    `fitnesses = list(map(toolbox.evaluate, pop))`
    `for ind, fit in zip(pop, fitnesses):`
        `ind.fitness.values = fit`

After the crossover on the entire population, we print out the result to check the progress over time:
   * Max increased from 65.0 at generation 0 to 100.0 at generation 39
   * Average increased from 53.82 in generation 0 to 97.81 at generation 39

We can deduce that after running the code many times, one can notice that the optimal maximum expected wasn’t always reached and that should be due to the random nature of initialization, crossover, and mutation.

                                                       The N Queens Problem

The N Queens is the problem of putting N chess queens on an NxN chessboard such that no two queens attack each other. We use n=20 to create the fitness and individual classes. After that, we define our evaluation function like as below:

`def evalNQueens(individual):`
    `size = len(individual)`
    `#Count the number of conflicts with other queens.`
    `#The conflicts can only be diagonal, count on each diagonal line`
    `left_diagonal = [0] * (2*size-1)`
    `right_diagonal = [0] * (2*size-1)`
    `#Sum the number of queens on each diagonal:`
    `for i in range(size):`
        `left_diagonal[i+individual[i]] += 1`
        `right_diagonal[size-1-i+individual[i]] += 1`
    `#Count the number of conflicts on each diagonal`
    `sum_ = 0`
    `for i in range(2*size-1):`
        `if left_diagonal[i] > 1:`
            `sum_ += left_diagonal[i] - 1`
        `if right_diagonal[i] > 1:`
            `sum_ += right_diagonal[i] - 1`
    `return sum_,`


In the next step, we define our crossover. We will be facing a partially matched crossover mating. It shows swapping a pair of queens’ positions between 2 parent individuals which is more effective in this scenario. We finally use our mutation function, shown below:

`def mutShuffleIndexes(individual, indpb):`
    `size = len(individual)`
    `for i in range(size):`
        `if random.random() < indpb:`
            `swap_indx = random.randint(0, size - 2)`
            `if swap_indx >= i:`
                `swap_indx += 1`
            `individual[i], individual[swap_indx] = \`
                `individual[swap_indx], individual[i]`
    `return individual,`


In this exercise, it is a must to Shuffle indexes because it represents the position of the queens on the chessboard. At the same time, we cannot mutate or duplicate indexes as this might cause a result to be out of bounds. At the end, I defined the loop and ran it for the 100 generations, one can see that we change from a max of 16.0 in generation 0 to 9.0 in generation 99. Min, average, and max significantly decreased when measuring the fitness throughout the generations.

