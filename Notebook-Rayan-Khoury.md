
* Name: Rayan Khoury
* Email: rkhoury7@gatech.edu
* Mobile number: +1(678)789-7927

# **October 6th, 2021**
## **Meeting 7**
### Class Notes:
* Discussed and learned about Emade
     * EMADE is Evolutionary Multi-Objective Algorithm Design Engine
     * Combines a multi-objective evolutionary search with high-level primitves to automate process of designing ML algorithms
* Launched a project with a presentation on the 25th of October
    * Setup Emade 
    * Successfully setup and run MySQL and allow users to have access to the server
    * EMADE runs across multiple datasets - preprocessed into gzipped csv files
    * Each train and test file create a DataPair object in EMADE
    * Can decompress them or use an editor such as gvim to view while zipped
    * Each row corresponds to an instance, each column is a feature, final column is truth

|Task Description|Current Status|Due Date|Date Assigned|Resolved Date|
|---|---|---|---|---|
|Notebook Update|Complete|10/13/2021|10/06/2021|10/08/2021|
|Setup EMADE|Complete|10/13/2021|10/06/2021|10/10/2021|
|Complete MySQL setup|Complete|10/13/2021|10/06/2021|10/10/2021|

# **September 29th, 2021**
## **Meeting 6**
### Class Notes:
* Presented our powerpoint presentation regarding the titatnic project.
* Compared MOGP and ML for the same project and made various conclusions 
* Saw all other groups present and asked questions
* We were informed to try a crossover on the hof (hall of fame) individuals.

|Task Description|Current Status|Due Date|Date Assigned|Resolved Date|
|---|---|---|---|---|
|Notebook Update|Complete|09/29/2021|10/06/2021|09/04/2021|

# **September 22nd, 2021**
## **Meeting 5**
### Class Notes:
* Discussed projects that were done the previous week. We discovered that the pareto optimal solutions are often lost due to codominance.
* We were instructed to use Multiple Objective genetic programming to find a set of pareto optimal solutions for the same problem (titatnic).
* Trees take the same input as the ML model
* Strongly or loosely typed genetic programming is allowed
* We are only allowed to use selection, mutation, crossover, however we were not allowed to use any other algorithms. 
* Write our own genetic loop.
* Compare pareto fronts of both ML and MOGP
* Submit our submissions as a group on canvas
* Create a powerpoint including our findings

### Individual Notes, Group and Individual findings
* Created Google Colab notebook with same preprocessing as Titanic ML assignment
* Created an outline for implementation
    * Selected primitive datasets
    * Defined evaluation function
    * Wrote evolutionary loop
* Worked with strongly typed GP
* Worked with the NSGA II as selection method
* We used uniform mutation and and single-objective but we realised that it would not give us the best result. Thus, we referred to cxOnePointLeafBiased and mutNodeReplacement which improved our AUC greatly.
* Created Hall of Fame (hof) using the best individuals in all generations
* Predicted survived feature for test.csv
* Best Learner: FPR = 0, FNR = 0.9122807017543859
* Findings: 
    * MOGP was much better than Ml in terms of AUC
    * MOGP recognized individuals with high FPR and FNR rates, while the learners in the ML Pareto frontier tended to favour higher FNRs and lower FPRs.

![ML vs MOGP comparision](https://picc.io/ITF22eB.png)

|Task Description|Current Status|Due Date|Date Assigned|Resolved Date|
|---|---|---|---|---|
|Notebook Update|Complete|09/22/2021|09/15/2021|09/22/2021|
|Attend group meetings|Complete|09/22/2021|09/15/2021|09/21/2021|
|Complete model with MOGP|Complete|09/22/2021|09/15/2021|09/20/2021|
|Complete ppt presentation|Complete|09/22/2021|09/15/2021|09/21/2021|

# **September 15th, 2021**
## **Meeting 4**
### Class Notes:
* Presentation Guidelines and skills:
    * Make sure title slide has the following:
        * Clear and appropriate title
        * List of contributors
        * Date of presentation
    * If slides include graphs:
        * Have a clear title
        * Label axis and include a readable font
        * Make sure the Pareto Front lines go the appropriate direction for minimization versus maximization
    * Include page numbers as you will be able to go back to any given slide at anytime. 

* Introduction to Machine Learning:
    * Introduced to Kaggle and the titanic project
    * Use scikit learn for predictors. 
    * Use files train.csv, test.csv, predictions.csv, and the python notebook to structure project.
    * Pandas is the python equivalent of google sheets, it is used to read train.csv and test.csv
    * Use isna to find N/A values and replace them with averages of their columns (Use this to clean your dataset).
    * Divide training and testing data according to the following:
        * x_train: top x rows of train.csv
        * x_test: bottom(n-x) rows of train.csv
        * y_train: survived x rows of train.csv
        * y_test: survived(n-x) rows of train.csv
* Group project
    * We were divided into 4 groups were we are supposed to train and test data from the titanic example using different learners and find co-dominant solutions.
    * Use scikit documentation to learn ML models and get predictions

|Task Description|Current Status|Due Date|Date Assigned|Resolved Date|
|---|---|---|---|---|
|Notebook Update|Complete|09/15/2021|09/08/2021|09/14/2021|
|Join the Group slack|Complete|09/15/2021|09/08/2021|09/10/2021|
|Attend group meetings|Complete|09/15/2021|09/09/2021|09/12/2021|
|Complete model with learner|Complete|09/15/2021|09/08/2021|09/14/2021|


# **September 8th, 2021**
## **Meeting 3**
### Lecture 3 : Multiple Objectives
* Gene Pool: Set of genome to be evaluated during current generation
    * Genome
        * Genetic description of an individual
        * DNA
        * GA = set of values
        * GP = tree structure, string
    * Search Space
        * Set of all possible genomes
        * For automated algorithm design -> set of all possible algorithms
* The evaluation of genome associates a genome with a set of scores
    * Scores: 
        * True positive or TP: How often is the desired objective identified
        * False positive or FP: How often is something else than the desired object identified
    * Objectives
        * Set of measurements each genome is scored against
        * Phenotype 
* Objective Space: set of objectives
* Evaluation - Maps an in individual or genome.
    * From a location in search space: Genotypic description
    * To a location in objective space: Phenotype description
* Classification Measures
    * A data set made of positive and negative samples is inserted in a classifier which gives out one of the following results:
        * Actual Positive:
            * True Positive (TP) with the TP rate or TPR = TP/P = TP/(TP+FN)
            * False Negative (FN) with a FN rate or FNR = FN/P = FN/(TP+FN)
        * Actual Negative:
            * False Positive (FP) with a FP rate or FPR = FP/N = FP/(FP+TN)
            * True Negative (TN) with a TN rate or TNR = TN/N = TN/(TN+FP)
* Other measures include:
    * Precision or Positive Predictive Value (PPV): PPV = TP/(TP+FP) 
    * False Discovery Rate (FDR): FDR = FP(TP+FP) = 1 - PPV 
    * Negative Predictive Value (NPV): NPV = TN/(TN+FN)
    * Accuracy (ACC): ACC = (TP+TN)/(P+N) = (TP+TN)/(TP+FP+FN+TN) 

![Maximization](https://picc.io/_AX345W.png)

![Minimization](https://picc.io/sEK6d2x.png)

* Pareto Optimality:
    * An individual is Pareto optimal if there is no individual in the population that outperforms this individual on all objectives
    * The set of all Pareto individuals is known as the Pareto Frontier
    * These individuals represent unique contributions
    * We want to drive selection by favoring Pareto individuals but maintain diversity by giving all individuals some probability of mating.
* Strength Pareto Evolutionary Algorithm
    * Each individual is given a strength S (S is how many others in the population it dominates)
    * Each individual receives a rank R (R is the sum of S's of the individuals that dominate it)
    * A distance to the kth nearest neighbour is calculated and a fitness of R+1/(𝛔k + 2) is obtained
## VIP Notebook Grading
| Category | Criteria | Poor | Intermediate | Exemplary |
| --- | ----------- | --- | ----------- |----------- |
| Notebook Maintenance | Name & contact info |  |  | 5 |
| " " | Teammate names and contact info easy to find |  |  | 5 |
| " " | Organization |  |  | 5 |
| " " | Updated at least weekly |  |  | 10 |
| Meeting notes | Main meeting notes |  |  | 5 |
| " " | Sub-teams' efforts |  |  | 10 |
| Personal work and accomplishments | To-do items: clarity, easy to find |  |  | 5 |
| " " | To-do list consistency checked and dated |  |  | 10 |
| " " | To-dos and cancellations checked and dated |  |  | 5 |
| " " | Level of detail: personal work and accomplishments |  |  | 13 |
| Useful resource | References (internal, external) |  |  | 8 |
| " " | Useful resource for the team |  |  | 14 |
| Total |  |  |  | 95 |
 
### Action items
|Task Description|Current Status|Due Date|Date Assigned|Resolved Date|
|---|---|---|---|---|
|Lecture 3 Notes|Complete|09/15/2021|09/08/2021|09/13/2021|
|Self-Evaluation|Complete|09/15/2021|09/08/2021|09/13/2021|
|Lab 2(part II) Notes|Complete|09/15/2021|09/08/2021|09/14/2021|

# **September 1st, 2021**
## **Meeting 2**
### Lecture 2
1. Reviewed notes of previous week (genetic algorithms)
2. Introduction to genetic programming
3. Genetic algorithms have evaluator functions that obtain an individual's objective score. On the other hand, the individual is the function in genetic programming.
4. The tree representation was introduced:
    * Nodes, also known as primitives, represent functions.
    * Leaves, also known as terminals, represent parameters.
         * The output is produced at the root of the tree, whereas the input is at a terminal (usually in the beginning).
         * An example of a function could be f(x) = 3*4+1. When placed in a tree, the function could be read and executed as follows [+,*,3,4,1]
    * The tree is converted to a "lisp preordered parse tree". This generated by starting from the root and then expanding.

![Tree](https://picc.io/jFjPnv1.png)

5. Crossover in Genetic Programming:
    * Crossover is tree-based through the exchange of subtrees.
    * Is is initiated by picking a random position on any tree. This leads to the exchange of the points and everything under them, creating a subtree.
    * The subtrees are exchanged to produce children.

![Crossover in GP](https://picc.io/p0bLTHh.png)

6. Mutation in Genetic Programming:
    * Mutation occurs when inserting, modifying, or deleting a node or subtree
*  An Example: Symbolic Regression
    * Using simple primitives, use genetic programming to evolve a solution to y = sin(x). (primitives include: +,*,-,/)
    * This solution is evolved using Taylor series of sin(x)
7. Evaluating a Tree:
    * Feed a number of inputs into a function to get outputs
    * Run the function
    * Measure the error between outputs and truth
8. Primitives that make this evolution easier:
    * Power()
    * Factorial()
    * sin()
    * cos()
    * tan()
* **This is the main idea behind EMADE**
## **Lab 2: Genetic Programming and Multi-Objective Optimization**
* First, we import the libraries.
* We create out fitness and individual classes.
* In this lab, our individual class inherits from the DEAP library and not from a list as our individual will be represented in a tree. Trees are the most common data structures in genetic programming as they are made of functions and variables called primitives. When evaluating an individual, we compile the tree from its leaves to its roots.\
* We then initialize a primitive set and add the ones that can be used.
* After defining out toolbox, individual, population, and compiler, we define our evaluation function.
* We run our evolutionary algorithm for 40 generations. Our best individual results as follows:
   *  `Best individual is add(multiply(x, add(multiply(add(multiply(x, x), x), x), x)), x), (1.2992604035586602e-16,)`
### Multi-Objective Genetic Programming with Symbolic Regression
* Multi-objective optimization is a result of modifications in the previous part of the code, especially in mean squared error and the size of our tree.
* We add three new primitives, then reinitialize our primitive set and set a seed for randomization.
* Then, we define our pareto dominance, which returns true if the first individual dominates the second individual. Thus, we end up wuth our new and modified best individual.
### Action items
|Task Description|Current Status|Due Date|Date Assigned|Resolved Date|
|---|---|---|---|---|
|Lecture 2 Notes|Complete|09/08/2021|09/01/2021|09/03/2021|
|Lab 2 Notes|Complete|09/08/2021|09/01/2021|09/05/2021|

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
|Task Description|Current Status|Due Date|Date Assigned|Resolved Date|
|---|---|---|---|---|
|Lecture 1 Notes|Complete|09/01/2021|08/25/2021|08/29/2021|
|Lab 1 Notes|Complete|09/01/2021|08/25/2021|08/29/2021|
