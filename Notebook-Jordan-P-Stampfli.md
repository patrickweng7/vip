## Table of Contents
- [Team Member](#team-member)
- [September 11, 2021](#september-11-2021)
  * [Lecture 3](#lecture-3)
  * [Action Items](#action-items)
- [September 2, 2021](#september-2-2021)
  * [Lab 2](#lab-2)
  * [Action Items](#action-items-1)
- [September 1, 2021](#september-1-2021)
  * [Lecture 2](#lecture-2)
  * [Action Items](#action-items-2)
- [August 28, 2021](#august-28-2021)
  * [Lab 1 Notes](#lab-1-notes)
  * [Action Items](#action-items-3)
- [August 25, 2021](#august-25-2021)
  * [Lecture 1](#lecture-1)
  * [Action Items](#action-items-4)


## Team Member 

Team Member: Jordan Stampfli

Email: jstampfli3@gatech.edu

Cell Phone: 914-874-3666

## September 11, 2021

### Lecture 3
* objectives
  * recognize power of multi objectives
  * understand pareto dominance
  * understand classification terms
  * use multi objectives to form teams
* what does algorithm look for in mate
  * speed, accuracy, memory usage
  * scalability, reliability, adaptability, consistency
* gene pool is the set of genomes to be evaluated
* search space
  * all possible genomes (algorithms)
* evaluation
  * true positive: identify the desired object
  * false positive: identify something else as the desired object
  * true negative: identify something else as not the desired object
  * false negative: identify the object as something else
  * objective space: set of objectives
  * maps from the search space to the objective space
* classification measures
  * confusion matrix: true positive, false negative, false positive, true negative
    * maximization measures
      * true positive rate (TPR): TP/(TP+FN)
      * specificity (true negative rate):TN/(TN+FP)
    * minimization measures
      * false negative rate (FNR): 1-TPR=FN/(TP+FN)
      * false positive rate (FPR): 1-TNR=FP/(TN+FP)
    * other measures
      * precision (PPV): TP/(TP+FP), want to maximize
      * false discovery rate (FDR): FP/(TP+FP), want to minimize
      * negative predictive (NPV): TN/(TN+FN), want to maximize
      * accuracy (ACC):(TP+TN)/(AP+AN), want to maximize
* objective space
  * individuals evaluated using objective functions
  * scores give individuals a point in space
  * individuals phenotype
  * extendable to N objectives
* pareto optimality
  * individual is pareto optimal if no other individual outperforms in all objectives
  * set of all pareto optimal individuals is pareto frontier
  * each pareto optimal individual represents unique contributions
  * drive progress by favoring pareto optimal individuals for reproduction
    * maintain diversity by giving all individuals a chance to reproduce
* Nondominated Sorting Genetic Algorithm 2, want to minimize
  * population separated into nondomination ranks
    * all possible frontiers created by removing the better frontiers progressively
  * individuals selected by binary tournament
    * lower rank beats higher rank
    * ties broken by crowding distance
      * points with greater separation that are more alone win
* Strength pareto Evolutionary Algorithm 2, want to minimize
  * each individual given strength S
    * S=how many points the individual dominates
  * each individual gets a rank R
    * sum of S for all individuals that dominate the individual 
    * pareto individuals have a rank R=0, since they aren't dominated by any other points
  * fitness is R + 1/(distance_k + 2)
    * lower rank will always beat higher rank regardless of distance
    * distance_k: the distance to the Kth nearest neighbor
    * favors further distances, points that are more alone

### Action Items
|Task|Due Date|Progress Notes|Current Status|
|---|---|---|---|
|Lab 2 Part 2|9/15|NA|Incomplete|
|Self grading rubric|9/15|NA|Incomplete|

## September 2, 2021

### Lab 2
* added exponents and logs with arities 2 and 1
  * difficulty getting np.float_power to work
    * tried changing input values
    * tried to define using data types instead of arity
  * similar issues with logs
* successfully added sin and cos primitives which weren't used in optimal answers
* added the primitive swap mutation
* results with four basic primitives: +,-,/,*
  * best individual: (X+(X\*X)*(X+X\*X)+X\*X)=(X+X<sup>3</sup>+X<sup>4</sup>+X<sup>2</sup>)
* results with four basic primitives and power
  * best individual: 2X*(X<sup>X</sup>+X<sup>2</sup>)
  * on different runs got very complex trees with 20+ primitives

### Action Items
|Task|Due Date|Progress Notes|Current Status|
|---|---|---|---|
|Lab 2|9/8|Walked through with multiple attempted primitives, mutations, and parameters|Complete|
|Join Slack|9/4|Used school email|Complete|


## September 1, 2021

### Lecture 2
* genetic algorithm vs genetic programming
  * genetic programming: individuals are functions
    * converts input data to output data
    * tree structure
      * nodes: primitives (functions)
      * leaves: terminals (parameters/input data)
      * output comes from root
      * stored as a "listp preorered parse tree"
      * crossover is exchanging subtrees
      * mutations: insert, remove, change
* symbolic regression example
  * evolve y=sin(x)
  * primitives: +,-,/,*
  * terminals: integers, X
  * evaluation: sum squared error

### Action Items

|Task|Due Date|Progress Notes|Current Status|
|---|---|---|---|
|Lab 2|9/8|NA|Incomplete|
|Join Slack|9/4|NA|Incomplete| 

## August 28, 2021

### Lab 1 Notes
* installed deap in anaconda
* linked deap tutorial on creator and toolbox
  * we defined fitness objective and individual object
  * "attr-bool"
    * generate boolean values
  * "individual"
    * create 100 individuals randomly using "attr_bool"
  * "register"
    * output a list of 100 individuals using "individual"
* did research to understand stride, splicing, and map()
* n-queens problem
  * two queens are on the same diagonal if the sum of the row and column are equal
  * right diagonals travel from the upper right to the lower left
  * left diagonals travel from the upper left to the lower right
  * partially matched crossover
    * within the crossover area, each parent switches their value at the current index with the value in their list equal to the value at the other 
      parent's index
    * was hard to understand from the code in the lab, much easier to grasp when looking at an example
  * wrote mutate method to randomly shuffle the values in an area between two random indices
    * achieved average of all averages of 0.54, alternative mutate achieved 1
  * had to add the line "%matplotlib inline" to get the plot to show

**Swapping index 2 using partially matched crossover**

Parent 1
|0|3|2|1|
|---|---|---|---|

Parent 2
|1|2|0|3|
|---|---|---|---|

**After the swap**

Parent 1
|0|2|3|1|
|---|---|---|---|

Parent 2
|1|3|0|2|
|---|---|---|---|

### Action Items

| Task | Due Date | Progress Notes | Current Status|
|---|---|---|---|
|Install Anaconda| 8/28 | Completed for Laptop|Complete|
|Lab 1| 9/01 | Improved in Jupyter Notebook and general python as well as deap|Complete|

## August 25, 2021 
* Dr. Zutty went over the syllabus and presentation dates
* Dr. Zutty gave lecture 1

### Lecture 1
* genetic algorithms
  * evolve generations through selective breeding to improve fitness and optimize solutions
  * individual: one person in population
  * population: group of all individuals
  * objective: performance metric (raw score)
  * fitness: relative performance (curved score)
  * evaluation: computes objective given an individual
  * selection: survival of the fittest
    * fitness proportionate: increase fitness, increase chance of selection
    * tournament: individuals pair off into groups and compare fitness to see who is selected
  * mate/crossover: splice genes
  * mutation: random modifications used to maintain diversity
* genetic algorithm formula
  * initialize population
  * evaluate population
  * loop
    * select parents
    * mate and mutate
    * evaluate new generation

### Action Items

| Task | Due Date | Progress Notes | Current Status|
|---|---|---|---|
|Install Anaconda| 8/28 | Completed for Desktop|Incomplete|
|Lab 1| 9/01 | Comfortable with Jupyter Notebook|Incomplete|