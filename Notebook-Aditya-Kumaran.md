== Team Member ==
[[files/GeorgiaTechBuzz.jpg|thumb|123x123px]]

Team Member: Aditya Kumaran

Email: akumaran6@gatech.edu
Cell Phone: +65 86455011

Interests: Writing fiction, Music, Reading, Sports

== September 1, 2021 (Week 2) == 

=== Lecture Notes: ===
* Learned how to add images to notebook via 'git clone'
* Introduced to genetic algorithms:
    * Individual is now the function itself
    * Practices running data through the individual instead of evaluating the individual AS the data.
        *  Ex.- 0,1,2,3,4,5 -> Individual -> 0,1,4,9,16,15 -> Evaluate
 	*  Individual function here is squaring
 	*  Evaluator would match output data to truth data for accuracy
    * Tree representation 
        *  Represents a program.
	*  Made up of nodes (primitives, functions) and leaves (terminals, end of the tree. Parameters or input data)
	*  Read bottom to top node
	*  Stored as a ‘lisp treeordered parse tree’
	*  [+,*,3,4,1]
            *  First is root
	    *  Next two are '*' and 1
	    *  '*' has two inputs, and they come before 1
    * Crossover
        *  Single-point crossover is just exchanging subtrees
            *  Starts by randomly selecting a point in the tree
            *  Subtrees are swapped to produce children
    * Mutation
        *  Inserting a node or subtree
        *  Deleting a node or subtree
        *  Changing a node


=== Individual Notes: ===
* Imported libraries from deap required for genetic programming (algorithms, base, creator, tools, gp)
* Created fitness and individual classes, which will be represented as a tree structure made of primitives. Evaluation compiles the primitive tree from leaves to root node.
* Initialized primitive set and added primitives like mathematical operators (add, subtract, multiply, negative). Added custom primitives np.deg2rad(arity=1) and np.ceil(arity=1).
* Registered four tool functions for expr (returns a tree based on a primitive set and maximum and minimum depth), individual, population, and compile (makes the tree into a function).
* Defined evaluation function, comparing the compiled function with the function we're trying to generate by minimizing mean squared error.
* Registered genetic operators for evaluate, select (tournament select, 3 per pod), mate (one point crossover), expr_mut, mutate. Added alternate mutation method, gp.mutInsert (inserts branch at a random position in individual).
* Performed the customary evolutionary loop and outputted the same statistics.
    * Achieved 1.16e-16 minimum fitness in best individual: add(x, multiply(x, add(multiply(x, x), add(x, multiply(x, multiply(x, x))))))
* Added images from labs (last week and this week).

=== Action Items: ===
{| class="wikitable"
!Task
!Current Status
!Date Assigned
!Suspense Date
!Date Resolved
|-
|Update Weekly Notebook
|Completed
|September 1, 2021
|September 8, 2021
|September 2, 2021
|-
|Begin "Lab 2 - Genetic Programming and Multi-Objective Optimization.ipynb" with JupyterLab
|
|September 1, 2021
|September 8, 2021
|September 1, 2021
|}


== August 25, 2021 (Week 1) == 

=== Lecture Notes: ===
* Learned about project requirements: Anaconda, JupyterLab, Python deap
* Introduced to EMade's GitHub, Wiki, Personal Progress Notebooks, Python Jupyter Notebooks, Slack
* Introduced to Genetic Algorithms:
    *  Allegories for how DNA works.
    *  Population-based solution.
    *  Many solutions to your problem, each being a “genome” and can mutate either individually or in groups 
* "Individual"
    *  Specific candidate in the population
    *  Like one person's DNA
* "Population"
    *  Group of individuals that will be altered
* "Objective"
    *  Performance measure of individual
    *  Is an objective measurement, not relative
* "Fitness"
    *  Relative measure of performance
* "Evaluation"
    *  Computes the objective of an individual
* "Selection"
    *  Gives preference to better individuals to pass on their genes
    *  Can be:
        *  Fitness Proportionate = fitness value is proportional to the probability of being selected for mating
        *  Tournament = from a group of individuals, the winner (best individual) is selected for mating. More random than fitness proportionate.
* "Mating"/"Crossover"
    *  Represents mating between individuals, in that DNA is taken from both places
    *  Can be a single point splice, double point splice, etc.
* "Mutation"
    *  Making a small change to an individual (changing a person's DNA)
    *  Purpose is to maintain diversity
* Algorithms
    *  Initialize population
    *  Evaluate population to get objective, fitness
    *  Loop through:
        1. Select parents
        2. Mating actions
        3. Mutations
        4. Determine fitness
        5. (until the best individual is acceptable)

=== Individual Notes: ===
* Downloaded and installed Anaconda Individual Version for Windows 10
* Using Anaconda Navigator, I launched JupyterLab
* Retrieved the DEAP Problem from the Calendar, under the Assignments column for the first week. Saved the file as .ipynb
* Imported the .ipynb into JupyterLab via 'New -> Text File'
* Opened a new Terminal window in JupyterLab, and used 'pip install deap' to install deap

''' OneMax Problem '''
* Using toolbox.register and tools.initRepeat, we'll create an individual with a list of 100 booleans (either 0 or 1).
* Writing the evalOneMax() function to evaluate the total fitness of an individual, we sum all of the 100 bits an individual carries.
* Defined four tool functions for evaluation (evalOneMax()), mating (2 point crossover), mutation (independent probability of bit flipping = 5%), and selection (tournament style, 3 per pod).
* Initialized population of 300, mapped the evaluation function to the population using: map(toolbox.evaluate, pop). Assigned individuals their fitness values as properties.
* Defined an evolutionary loop (40 generations), and performed tournament selection on the population, cloning the selected offspring to create separate instances from the previous iteration.
* Matched the even terms with their adjacent odd terms and called toolbox.mate() with 50% probability. Deleted the mated offspring's fitness values.
* Mutated individuals with 20% probabilities, deleted the mutated offspring's fitness values. 
* Re-evaluated the modified offspring and assigned their newly evaluated fitness values. Replaced old population with offspring.
* Calculated max, min, mean, and standard deviation statistics for new population.
* Ran main():
    * Achieved 100% maximum fitness in 31 generations.
    * Achieved 100% maximum fitness in 39 generations.
    * Achieved 100% maximum fitness in over 40 generations (99% maximum in 40 generations).
    * Achieved 100% maximum fitness in 34 generations.
    * Achieved 100% maximum fitness in 39 generations.

''' N Queens Problem '''
* Created fitness and individual classes for an nxn chessboard (sample is 20x20). Fitness is weighted negatively, since the goal is to minimize conflicts between the queens on the board.
* Created individuals using toolbox_q.permutation (returns randomized list of numbers less than n, representing the queens' columns), since there is only one queen per column.
* Count the number of queens on each diagonal for evalNQueens(individual), and sum the total number of conflicts on both left and right diagonals.
* Writing the partially matched crossover function for two individuals. Chose two random crossover points, and swapped the individuals' bits between those indices.
* Wrote the mutation function for individuals with a given probability of each attribute to be swapped with another random index (indpb).
* Implemented custom mutation function, swapping an additional term that's halfway between the index term and the randomly selected term.
    * def mutationCustom(individual, indpb):
    *     size = len(individual)
    *     for i in range(size):
    *         if random.random() < indpb:
    *             far_index = random.randint(0, size - 2)
    *             if far_index >= i:
    *                 far_index += 1
    *             middle_index = (i + far_index) / 2
    *             individual[i], individual[middle_index], individual[far_index] = \
    *                 individual[middle_index], individual[far_index], individual[i]
    * 
    *     return (individual, )
* Registered four tool functions for evaluation (evalNQueens()), mating (partially matched), mutation (independent probability of bit flipping = 2/n), and selection (tournament style, 3 per pod).
* Performed the same evolutionary loop as in OneMax and outputted the same statistics.
* Ran main() with mutShuffleIndexes():
    * Achieved 0 minimum fitness in 32 generations.
    * Achieved 0 minimum fitness in more than 100 generations (1.0 minimum in 100 generations).
    * Achieved 0 minimum fitness in more than 100 generations (1.0 minimum in 100 generations).
    * Achieved 0 minimum fitness in more than 100 generations (1.0 minimum in 100 generations).
    * Achieved 0 minimum fitness in 81 generations.
* Ran main() with mutationCustom():
    * Achieved 0 minimum fitness in more than 100 generations (1.0 minimum in 100 generations).
    * Achieved 0 minimum fitness in more than 100 generations (1.0 minimum in 100 generations).
    * Achieved 0 minimum fitness in 15 generations.
    * Achieved 0 minimum fitness in more than 100 generations (1.0 minimum in 100 generations).
    * Achieved 0 minimum fitness in 74 generations.
* Learned that my custom mutation function often gets to a minimum of 1.0 quickly, but routinely fails to reach a minimum of 0.


=== Action Items: ===
{| class="wikitable"
!Task
!Current Status
!Date Assigned
!Suspense Date
!Date Resolved
|-
|Set up personal notebook page
|Completed
|August 25, 2021
|September 1, 2021
|August 25, 2021
|-
|Join Slack
|Completed
|August 25, 2021
|September 1, 2021
|August 25, 2021
|-
|Complete "Lab 1 - Genetic Algorithms with DEAP.ipynb" with JupyterLab
|Completed
|August 25, 2021
|September 1, 2021
|August 29, 2021
|}
