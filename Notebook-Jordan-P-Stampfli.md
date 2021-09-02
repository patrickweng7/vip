## Table of Contents
- [Team Member](#team-member)
- [September 2, 2021](#september-2-2021)
  * [Lab 2](#lab-2)
  * [Action Items](#action-items)
- [September 1, 2021](#september-1-2021)
  * [Lecture 2](#lecture-2)
  * [Action Items](#action-items-1)
- [August 28, 2021](#august-28-2021)
  * [Lab 1 Notes](#lab-1-notes)
  * [Action Items](#action-items-2)
- [August 25, 2021](#august-25-2021)
  * [Lecture 1](#lecture-1)
  * [Action Items](#action-items-3)


## Team Member 

Team Member: Jordan Stampfli

Email: jstampfli3@gatech.edu

Cell Phone: 914-874-3666

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