## Table of Contents
- [Team Member](#team-member)
- [September 18, 2021](#september-18-2021)
  * [Subteam Meeting](#subteam-meeting)
  * [Action Items](#action-items)
- [September 17, 2021](#september-17-2021)
  * [Titanic Data Set](#titanic-data-set)
  * [Action Items](#action-items)
- [September 15, 2021](#september-15-2021)
  * [Action Items](#action-items-2)
- [September 11, 2021](#september-11-2021)
  * [Lecture 3](#lecture-3)
  * [Lab 2 Multiple Objectives](#lab-2-multiple-objectives)
  * [Self Grading Rubric](#self-grading-rubric)
  * [Action Items](#action-items-3)
- [September 2, 2021](#september-2-2021)
  * [Lab 2](#lab-2)
  * [Action Items](#action-items-4)
- [September 1, 2021](#september-1-2021)
  * [Lecture 2](#lecture-2)
  * [Action Items](#action-items-5)
- [August 28, 2021](#august-28-2021)
  * [Lab 1 Notes](#lab-1-notes)
  * [Action Items](#action-items-6)
- [August 25, 2021](#august-25-2021)
  * [Lecture 1](#lecture-1)
  * [Action Items](#action-items-7)


## Team Member 

Team Member: Jordan Stampfli

Email: jstampfli3@gatech.edu

Cell Phone: 914-874-3666

## September 18, 2021

### Subteam Meeting
* discussed my progress on the problem so far
  * shared code using google colab
  * decided not to work more on processing data
* wrote code to check codominance of the models I had worked with yesterday
  * used same split as jupyter notebook example, 33% in testing
  * found four to be codominant: perceptron, random forest, logistic regression, gaussian naive bayes
  * had to change the na filling for age
    * used simple mean
* found predictions for each of the four models on the test data
  * saved the 4 csv files 

|Model|False Negative|False Positive|
|---|---|---|
|Perceptron|30|27|
|Random Forest|30|30|
|Logistic Regression|31|25|
|Gaussian|40|21|

### Action Items
|Task Description|Current Status|Progress Notes|Date Assigned|Suspense Date|Resolved Date|
|---|---|---|---|---|---|
|Kaggle Titanic Data set|Complete|Struggled to find ways to add accuracy|9/15|9/22|9/18|
|Submit Titanic Test Predictions|Incomplete|Have the csv, unclear where to submit the assignment|9/15|9/22|NA|

## September 17, 2021

### Titanic Data Set
* began work on the kaggle titanic data set
* went through the jupyter notebook tutorial
* decided to use groupings for both age and fare because the data range was much greater than the other features
  * used roughly uniform groupings after filling the na with the means 
  * could possibly be improved by taking groupings more significant to the survival rate based on age
    * similar idea could be used with fare which is also uniformly grouped currently
* used name to derive titles rather than just deleting them
  * titles grouped into Mr, Ms, Mrs, Master, and other
  * the title feature would be set to 0 for all those that didn't list a title with their name 
* created several other new features relatives, gender_embarked, and age_class 
  * relatives is siblings plus parents
  * gender_embarked relates the gender to where they departed 
    * males who left from C had a high survival rate, whereas women who left from Q or S had a high survival rate
  * age_class is the product of a person's age grouping and their pclass
    * someone young and wealthy is very likely to survive
    * someone who is old and poor is very unlikely to survive
* next I ran cross evaluation scoring for accuracy and the confusion matrix on the following models
  * perceptron, random forest, stochastic gradient descent, logistic regression, k nearest neighbor, gaussian naive bayes, support vector machine, 
    decision tree, and neural network 
  * the pareto optimal models were random forest, logistic regression, knn, gaussian, svc, decision tree, and neural network
  * the highest mean accuracy of any model was 0.82 (random forest), the lowest false negative rate was 0.1 (decision tree), and the lowest false 
    positive rate was 0.21 (gaussian)
* further steps to improve
  * fine tune each of the pareto optimal models further looking at each of the hyperparameters 
  * look for more ways to identify positive information in the data
    * possibly do something more complex than just the mean for the missing ages and fares
    * possibly look for some way to include the cabin or ticket columns

### Action Items
|Task Description|Current Status|Progress Notes|Date Assigned|Suspense Date|Resolved Date|
|---|---|---|---|---|---|
|Kaggle Titanic Data set|Incomplete|Much progress made, large number of pareto optimal models, mediocre accuracy|9/15|9/22|NA|
|Submit Titanic Test Predictions|Incomplete|Needs to be submitted as csv in order starting with index 892|9/15|9/22|NA|

## September 15, 2021
* teams set for the remainder of the bootcamp
* Professor Zutty went through jupyter notebook for kaggle titanic data set
  * showed different resources for scikit, numpy, and pandas
  * allowed resources for titanic data set are scikit, numpy, and pandas

### Action Items
|Task Description|Current Status|Progress Notes|Date Assigned|Suspense Date|Resolved Date|
|---|---|---|---|---|---|
|Kaggle Titanic Data set|Incomplete|Allowed to use pandas, numpy, and scikit|9/15|9/22|NA|
|Submit Titanic Test Predictions|Incomplete|Needs to be submitted as csv in order starting with index 892|9/15|9/22|NA|

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

### Lab 2 Multiple Objectives
* general results
  * \\\three local images
* can greatly alter the pareto curve by essentially destroying the problem and having a constant population with no crossover or mutation
  * allows for massive minimization of the tree size
  * tree size much more impactful for area under the curve metric
  * \\\two local images
* similar idea with worse results is to cap height growth from mutation and crossover much more strictly
  * better function emulation not relevant enough for area under the curve metric
  * \\\two local images
* should consider weighting the error function much more heavily 
  * current mean squared error function makes the function performance almost irrelevant in comparison to the scaled height of the tree
* decrease the relevance of tree height
  * height currently being measured by the sum of all primitives and terminals
  * instead could be measured by the depth of the tree: roughly log(sum of all primitives and terminals)

### Self Grading Rubric
|Section|Specific|Score|
|---|---|---|
|Notebook Maintenance|Name & Contact Info|5|
||Teammate names and contact info easy to find|5|
||Organization|5|
||Updated at least weekly|10|
|Meeting Notes|Main meeting notes|5|
||Sub-teams' efforts|10|
|Personal Work & Accomplishments|To-do items:clarity, easy to find|5|
||To-do list consistency (weekly or more)|10|
||To-dos & cancellations checked & dated|5|
||Level of detail: personal work & accomplishments|15|
|Useful Resource|References (internal, external)|9|
||Useful Resource for the team|15|
|Total|Total Score Out of 100|99|

### Action Items
|Task Description|Current Status|Progress Notes|Date Assigned|Suspense Date|Resolved Date|
|---|---|---|---|---|---|
|Lab 2 Part 2|Completed|Most interesting to see the more unique and customizable primitives at the end|9/8|9/15|9/11|
|Self grading rubric|Complete|NA|9/8|9/15|9/11|

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
|Task Description|Current Status|Progress Notes|Date Assigned|Suspense Date|Resolved Date|
|---|---|---|---|---|---|
|Lab 2|Completed|Walked through with multiple attempted primitives, mutations, and parameters|9/1|9/8|9/2|
|Join Slack|Complete|Used school email|9/1|9/5|9/2|


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
|Task Description|Current Status|Progress Notes|Date Assigned|Suspense Date|Resolved Date|
|---|---|---|---|---|---|
|Lab 2|Incomplete|NA|9/1|9/8|NA|
|Join Slack|Incomplete|NA|9/1|9/5|NA|

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
|Task Description|Current Status|Progress Notes|Date Assigned|Suspense Date|Resolved Date|
|---|---|---|---|---|---|
|Install Anaconda|Complete|Completed for laptop|8/25|8/28|8/28|
|Lab 1|Complete|Improved in Jupyter Notebook and general python as well as deap|8/25|9/1|8/28|

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
|Task Description|Current Status|Progress Notes|Date Assigned|Suspense Date|Resolved Date|
|---|---|---|---|---|---|
|Install Anaconda|Incomplete|Completed for desktop|8/25|8/28|NA|
|Lab 1|Incomplete|Comfortable with Jupyter Notebook|8/25|9/1|NA|