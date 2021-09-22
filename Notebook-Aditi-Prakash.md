# Aditi Prakash
Name: Aditi Prakash  
Email: aprakash86@gatech.edu, Cell Phone: 704-794-3924  
Interests: Machine Learning, Data Science, Software Development, Dance, Reading

# Week 1: August 25th, 2021
## Overview
Discussed course format (10-week bootcamp followed by joining a sub-team), location of course GitHub and reference materials (https://github.gatech.edu/emade/emade/), expectations, Assignment 1, and notebooks. Attended lecture on Genetic Algorithms.

## Team Meeting Notes
### Lecture on Genetic Algorithms
Introduced concept of genetic algorithms that mimic evolutionary processes (mutation, selection, mating, fitness evaluation, reproduction, etc.) in order to maximize the fitness of individuals in a population of data. Identified steps of a genetic algorithm:
1. Random initialization of population.
2. Determining objective of population: how do we define performance of individuals?
3. Determining fitness of population: how does an individual's objective compare to that of others?
4. Subject individuals to selection methods (ex. fitness proportionate and tournament selection) so as to give preference to the fittest individuals in the population.  
5. Through an evolutionary loop, select parents from population, perform crossovers/mutations/selections on parents and save these modifications as offspring of the initial population, and determine the fitness of the population. Repeat until we maximize the fitness of the best individual in the population. 

Learned genetic algorithm solution to One Max Problem - a simple problem that presents the goal of maximizing the number of 1's that an individual contains (thereby maximizing the sum of the individual's values). 

## Lab 1 - Genetic Algorithms with DEAP
* Installed Conda, Python, and Jupyter Notebooks
* Cloned emade and reference-material repositories using Git 
### Lecture 1 - GA Walkthrough (introductory notebook for understanding of DEAP implementation of genetic algorithms)
* Installed DEAP using pip
* Imported base, creator, and tools libraries from DEAP
* Created FitnessMax Class to track objectives for individuals in One Max problem 
* Set weights attribute to have a value of 1.0 - our goal is to maximize this value for a given individual through the evolution process

Created:
* Individual class which inherits from list and has fitness attribute
* Binary random choice generator attr_bool using the DEAP toolbox to randomly present either a 0 or 1 for each value in the list for an individual
* individual() method to create a list of 100 randomly generator 0's and 1's for each individual and registered with DEAP toolbox
* population() method to create a set of individuals

Defined evaluation function for fitness: a sum operation across all of an individual's values.

Performed:
* in-place two-point crossover on individuals
* in-place mutation with a given probability of mutation on individuals

This notebook provided a solid introduction to the DEAP API and the representation of genetic algorithms in a high-level language like Python. While the lab itself presented a more in-depth example of the evolutionary process for more challenging optimization problems (like the n-queens problem), the information in this initial notebook will generalize well to future genetic algorithms problems.  

### Lab 1 - Genetic Algorithms with DEAP
This lab explored the One Max problem and the n-queens problem and defined genetic algorithms to solve both. 

**One Max Problem:**
For this problem, we followed many of the same steps that appeared in the Lecture 1 Notebook (see above). We define a main() function for the genetic algorithm, which evaluates the full population and initiates the evolutionary loop. Within the evolutionary loop, we select individuals for each successive generation, clone them, and perform mutations/crossovers on them. We then evaluate the fitness of these offspring and replace the existing population with the offspring. Finally, we return the fitnesses of the individuals (based on the predefined fitness operation - the sum of the individual's entries) and print statistics such as the mean fitness, squared sum of the fitnesses, and standard deviation of the fitnesses). We loop for some number of generations (40, in this case) and report the best individual that has resulted from this evolution process. Within the DEAP framework, we used libraries like creator (including the create() method), tools (including the selBest() method and the selTournament, mutFlipBit, and cxTwoPoint attributes), and base (including the Toolbox(), register(), select(),  mate(), and mutate() methods).

Findings: The global maximum (a best individual with a fitness equal to n, the number of entries in each individual) was reached within 40 generations about every 19 out of 20 times the algorithm was run; this indicates that our algorithm has an effectiveness of around 95%. Further improvements can be made by changing the bounds of the random number generation for crossover, mutation, and selection.  

![One Max Generations, Part 1](https://picc.io/pok5sgG.png)
![One Max Generations, Part 2](https://picc.io/ouFv77h.png)

**N Queens Problem:**
For this problem, we followed many of the same steps that appeared in the One Max Problem (see above). We define a size n = 25 for each individual and define a weight of -1.0 here, since we wish to minimize the number of conflicts between queens in our problem space. We then create a permutation function to populate the entries for each individual with numbers selected without replacement from range(n). We define our evaluation function as a measure of the number of conflicts along each diagonal of our board; with the creation process we defined for individuals, queens will not appear in the same row or column. [Describe evaluation function modification here w/ screenshots]. We then write the cxPartialyMatched() function for partially matched crossover, cxTwoPoint(), and mutShuffleIndexes() to shuffle values at different indexes within each individual (since we must remain within size n  = 25). We modified the mutation function to be a uniform int mutation, wherein randomly selected entries for each individual are replaced with a randomly selected value between 0 and n. The improvements seen with this new mutation function are described in the Findings section below. Finally, we run a similar evolutionary loop as the one described for the One Max Problem (see above) for 100 generations, return the fitnesses of the individuals (based on the predefined fitness operation - the number of conflicts between queens) and print statistics. We loop for some number of generations (100, in this case) and report the best individual that has resulted from this evolution process. 

Findings:
![N Queens Generations, Part 1](https://picc.io/UzJTkn-.png)
![N Queens Generations, Part 2](https://picc.io/BAhG-pn.png)

Visualizations:

1. With Shuffle Indexes Mutation:
![N Queens Visualization](https://picc.io/-qpvzmX.png)

2. With Uniform Int Mutation:
![N Queens Visualization with Uniform Int Mutation](https://picc.io/e1uHhHm.png)

* We can see here that the maximum fitness value decreased much more quickly with the Uniform Int mutation than the Shuffle Indexes mutation. We also see that the average and minimum fitness values tended towards 0 more closely than they did with the Shuffle Index mutation. 

3. With 85 Generations and 10% Mutation Rate (Shuffle Index Mutation):
![N Queens Visualization with 85 Generations and 10%  Mutation Rate](https://picc.io/MZtm5UD.png)

* We can see here that with a 10% mutation rate as opposed to the initial 20% mutation rate and with 85 generations as opposed to 100, we obtain a best individual with a fitness of 0 more consistently than we did previously. The maximum fitness also trends towards our best fitness more quickly than before. This also points to the fact that Shuffle Index Mutation may not be the best mutation for this particular problem, since a lower percentage of that mutation led to more consistent results in fewer generations. 

Additional improvements can be made to the current n-queens algorithm such that we obtain an individual with the optimal fitness in a minimum number of generations. We can continue to tweak the probabilities of mutation and mating for offspring, change the tournament size, change our methods of mating, mutation, selection, etc., change the parameters of our mating and mutation (ex. points of mating, values that the data in our individuals can be mutated to), and change our evaluation function.

**Action Items:**
| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| --- | ----------- | --- | ----------- |----------- |
| Install DEAP and set up JupyterLab for Lab 1 | Completed | 8/25/2021 | 9/1/2021 | 8/30/2021 |
| Complete Lecture 1: GA Walkthrough | Completed | 8/25/2021 | 9/1/2021 | 8/30/2021 |
| Complete Lab 1 | Completed | 8/25/2021 | 9/1/2021 | 8/30/2021 |
| Set Up Notebook | Completed | 8/25/2021 | 9/1/2021 | 8/30/2021 |
| Review Genetic Algorithms | Completed | 8/25/2021 | 9/1/2021 | 8/30/2021 |

# Week 2: September 1st, 2021
## Overview
Attended lecture on genetic programming and completed Lab 2 on the same topic. Continued to discuss course expectations and direction after 10-week bootcamp. 

## Team Meeting Notes
### Lecture on Genetic Algorithms
Introduced concept of genetic programming with the goal of optimizing a function (represented as a tree structure) to achieve a particular target output.  
1. Nodes: primitives, represent functions 
2. Leaves: terminals, represent parameters 
3. Explored examples of lisp preordered parse trees that represent functions
4. Crossover in GP: exchanging subtrees
5. Mutation in GP: Inserting/deleting nodes and subtrees
5. Measuring error (ex. Mean Squared Error)
7. Identifying primitives that can make modeling a function easier 

### Lab 2 - Genetic Programming
This lab explored the problem of optimizing a set of primitives to achieve a target function model. This exercise is in contrast to typical machine learning or data modeling, wherein we attempt to fit a function to data. Here, we use the mean squared error to obtain the fitness of each individual in the population; that is, we determine the MAE between our primitives-based function and the target function.   

We first create our fitness and individual classes, where individuals are of the PrimitiveTree type. We then initialize the set of primitives our trees can draw from and register our objects with the DEAP toolbox. We also define our evaluation function (which uses the MAE between the modeled function and the actual function) and register the evaluation, selection, mating, and mutation operators with the DEAP toolbox. We then programmed the same evolutionary algorithm that was used in Lab 1 for the n-queens problem and obtained the best individual after 40 generations. We also graphed the results and printed our statistics. 

Findings: The global maximum (a best individual with a fitness or MAE of 0) was almost reached. The best maximum individual reached a minimum fitness value of around 1.5. The average and minimum fitnesses approached a fitness of 0 closely (0 was an asymptote for these values). Further improvements can be made by changing the bounds of the random number generation for crossover, mutation, and selection.
  
The best individual was determined to be the following: Best individual is add(add(multiply(x, x), multiply(add(multiply(x, multiply(x, x)), multiply(x, x)), x)), x), (8.620776339403237e-17,). 

Visualization:
![Genetic Programming Visualization](https://picc.io/x91IjkA.png)

* We can see here that the maximum fitness value seems to oscillate around a fitness of about 2.0 and does not continue decreasing after about the 10th generation. 

Additional improvements can be made to the current genetic programming algorithm such that we obtain an individual with the optimal fitness in a minimum number of generations. We can continue to tweak the probabilities of mutation and mating for offspring, change the tournament size, change our methods of mating, mutation, selection, etc., change the parameters of our mating and mutation (ex. points of mating, values that the data in our individuals can be mutated to), and change our evaluation function.

**Action Items:**
| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| --- | ----------- | --- | ----------- |----------- |
| Continue to install DEAP and supporting libraries | Completed | 9/1/2021 | 9/8/2021 | 9/6/2021 |
| Complete Lab 2: Genetic Programming | Completed | 9/1/2021 | 9/8/2021 | 9/6/2021 |
| Review Genetic Programming Notes | Completed | 9/1/2021 | 9/8/2021 | 9/6/2021 |
| Update Notebook | Completed | 9/1/2021 | 9/8/2021 | 9/6/2021 |

# Week 3: September 8th, 2021

## Overview
Attended lecture on multi-objective optimization and completed Lab 2's Multi-Objective programming exercise. Filled out survey sheet with ML and Python self-ratings.  

## Team Meeting Notes
### Lecture on Multi-Objective Optimizations:
Accuracy, speed, memory helps define metrics that an algorithm might look for in a mate
Scalability, reliability, adaptability, consistency (tradeoff between precision and accuracy)

Search space - all of the things that make up an individual (one max - 1,0 for each element in the list for how many elements there are)
Limitations in memory limit how deep our algorithms can be. Full search space is the full set of possible algorithms. 
How can we come up with good fitnesses to help us search these worlds?
Objectives are our phenotype, evaluation function allows us to go from the genotype to the phenotype. 
Binary classification vs. multiclass classification - 1 or 0 versus in a set of things

Algorithm is for red objects, we are trying to find apples.

Precision or positive predictive value - overlooked but very important
Assessing algorithm’s consistency of performance with itself, regardless of truth 
Accuracy - bigger is better
Blue and green form a tradeoff space between each other against Objective 1 and Objective 2

Dominated solution - there is an individual that would live in the space to the left and under a given point 
Non dominated - there is no such individual 
Co-dominant, none are dominated, form Pareto frontier 
Want to keep diversity of genotypes, want all tree structures to stay in population, algorithms will continue to be diverse (their representations are diverse), want to reward this, stops algorithms from converging 
Nondominated solution is called Pareto optimal in this class
Would much rather have spread out points on Pareto frontier than clumped up individuals on either end in Pareto frontier
Reward places that are off by themselves so we can keep that diversity 
Higher crowding distance wins

SPEA2: How many points does it dominate (look up and to the right)
S is how many others in the population it dominates
Rank is the sum of S’s of the individuals that dominate it 

Tiebreakers:
Fractional so serves as tiebreaker, one with higher distance is going to have a smaller effect on rank, if crowding distance is smaller, you’ll be closer to 1, almost at the next range, favor larger distance because it will get inverted 
Niching - trying to spread diversity 
Both algorithms favor nondomination of something more highly than how different it is from everything else. 
Kth nearest neighbor - look at Euclidean distance in a space for all points to a kth neighbor 
Larger the distance, the better, minimizes the 1/sigma, which minimizes the rank + 1/sigma 

### Lab 2 - Multi-Objective Genetic Programming
This lab explored the problem of optimizing a set of primitives based on more than one objective to achieve a target function model. Here, we minimize the mean squared error and the size of the tree. We also add the sin, cos, and tan functions to our set of primitives and reinitialize the toolbox. We then define a function to evaluate our symbolic regression and note that this new problem, with an evaluation function that takes the sin, cos, and tangent of the points into consideration when evaluating the individuals for fitness, cannot be solved within 100 generations like the ones we worked on previously. 

We then define the pareto dominance function, which compares two individuals and returns the individual which dominates the other in the objective space. We initialize 300 individuals and leave one individual as the comparison individual. We then sort the population we created by each individual's Pareto dominance as compared to the "spare" individual. Plotting the objective space, we are able to visualize the individuals that minimize both objectives and exist along the Pareto front. Running the evolutionary algorithm, we identify the Best Individual: negative(cos(multiply(add(cos(sin(cos(sin(cos(tan(x)))))), cos(x)), tan(x))))
with fitness: (0.2786133308027132, 15.0). 

DEAP's Mu plus Lambda algorithm, which takes in a mu and lambda value (number of individuals to select for each successive generation, and the number of children to produce at each generation). This allows us to control the size of the population as well as the selection process between individuals. We identify that the size of our trees grows over generations, but the MAE quickly drops to a sub-1 value over generations. Visualizing our pareto front, we see that the Area Under Curve: 2.3841416372199005 indicates the amount of objective space that exists below our current Pareto front. 

Visualization:
[Screenshots](https://docs.google.com/document/d/1isLlHDQdceJ9ZrUbcG3oIeYEq1j0an6Oi0SyJKpFyvM/edit?usp=sharing)

* Improvements:
Modifying the following hyperparameters reduced the AUC of the Pareto front to 0.97. 
NGEN = 50
MU = 60
LAMBDA = 75
CXPB = 0.4
MUTPB = 0

Additional improvements can be made to the current genetic programming algorithm such that we obtain an individual with the optimal fitness in a minimum number of generations. We can continue to tweak the probabilities of mutation and mating for offspring, change the tournament size, change our methods of mating, mutation, selection, etc., change the parameters of our mating and mutation (ex. points of mating, values that the data in our individuals can be mutated to), and change our evaluation function.

**Action Items:**
| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| --- | ----------- | --- | ----------- |----------- |
| Complete Lab 2: Multi-Objective Genetic Programming | Completed | 9/8/2021 | 9/15/2021 | 9/12/2021 |
| Review Multi-Objective Programming Notes | Completed | 9/8/2021 | 9/15/2021 | 9/12/2021 |
| Complete Self-Grading Rubric | Completed | 9/8/2021  | 9/15/2021 | 9/12/2021 |
| Update Notebook | Completed | 9/8/2021 | 9/15/2021 | 9/12/2021  |

## Self-Grading Rubric
[Self-Grading Rubric Linked Here](https://drive.google.com/file/d/113sDYMD9rzibZrI8jPXoMj3mhbJmRQFn/view?usp=sharing)

Markdown version of self-grading rubric here:
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
| " " | Level of detail: personal work and accomplishments |  |  | 14 |
| Useful resource | References (internal, external) |  |  | 10 |
| " " | Useful resource for the team |  |  | 14 |
Comments: I keep my notebook as detailed as possible and ensure that when I look back at my documentation for each week, I am able to recall all of the information I need in a timely and efficient manner. I also make sure my writing and documentation are are easily understandable as possible so that other people can navigate my work efficiently as well. 
| Column Totals |  |  |  | 98 |

# Week 4: September 15th, 2021
## Overview
Received bootcamp subteam assignments (I am in Bootcamp Subteam 4) and explored Kaggle Titanic dataset. Discussed Titanic ML assignment wherein each member of our subteam is to select a learner, use it to predict the 'Survived' feature in the Titanic dataset, and determine the FNR and FPR of that learner. All of our learners must be codominant, meaning that no learner should outperform any other learner on both minimization objectives (FNR and FPR). Exchanged contact information with team and decided to meet throughout the week and create Slack channel for communication. Discussed preliminary ideas for data preprocessing and hyperparameter tuning.

## Team Meeting Notes
### Notes on Titanic ML Assignment 
- nans, strings, balance data, fold data, make sure everyone is using same X_train, y_train, X_test, y_test
- Post csv representing predictions of your model that was co-dominant with rest of group. 
- Sci-kit learn - classification (ex. Support Vector machine)
- Do Pareto graphing for minimization objectives
- Pandas documentation
- Why did the decision classifier perform so well when we didn’t do that much?
- Make sure submission samples are in the same order for everyone 
- Pandas, sci-kit learn - dig deep 
- Use n folds
- Look at cabin values and encode Embarked 
- Do k fold splits for all learners
- Cross val score - average of false negatives and false positive 
- Look at average for nan values across samples with similar features versus all samples
- Create csv files with data that we’re using for preprocessing 
- Create a jupyter notebook to graph pareto frontier - everyone inputs their values
- Don’t mix up the rows
- Undersampling/oversampling 

## Titanic ML Problem 
### Data Preprocessing
* Created Google Colab notebook for group preprocessing
* Created ParetoFront.ipynb for group to input objective values for individual learner and confirm co-dominance
* Imported pandas, numpy, and sklearn methods 
* Mounted Drive to Colab and read in train and test sets as dataframes
* Dropped Name feature (irrelevance) and Cabin feature (too sparse to work with)
* Set PassengerID as index
* Replaced null values of Embarked feature with mode of Embarked column and null values of Ticket feature with '100'. Held off on replacing Age and Fare null values here and replaced them later with median value of each respective feature for a given Pclass. This is so that the null values in the Age and Fare columns are not replaced with values that are not representative of the central value of those features for all samples of a particular type (in this case, a particular Pclass). 
* One hot encoded Embarked feature values so as to not incorrectly assign a magnitude of value to each Embarked class (ie. 'Embarked': {'C': 0, 'Q': 1, 'S': 2} might cause our learner to assume a relationship between Survived and Embarked for rows with an Embarked class of 'S' and no relationship between Survived and Embarked for rows with an Embarked class of 'C'). Created three columns, 0, 1, 2, each of which is assigned either the value 0 or 1 for each sample based on the Embarked class for that sample. 
* Replaced Sex feature categories with 1 for male and 0 for female
* Extracted numerical part of Ticket feature and re-assigned Ticket column values to numerical portion (type=integer). This is so as to consider the relationship between ticket assignments and survival empirically (for instance, those with lower ticket numbers may have purchased their tickets earlier than those with higher ticket numbers, which could indicate residence in a particular location of the ship (ex. the upper or lower deck) at the time of the crash, impacting survival). This feature engineering had little to no impact on the FNR and FPR of the model. 
* Replaced null Age and Fare values with median values based on Pclass of passenger (see above). 
* Split training data into training and testing sets (test_size=0.33, random_state=10)
* Selected XGBoost learner due to its speed and ability to handle null data
* Initially ran XGBoost predictions with default hyperparameters 
* Obtain confusion matrix for predictions 
* Modified XGBoost hyperparameters
Final Learner: XGBoostClassifier(objective="multi:softprob", num_class=2,  eta=0.005, max_depth=10, subsample=0.98, colsample_bytree=0.9, eval_metric="auc", n_estimators=10000, scale_pos_weight=0.2). This learner had 31 False Positives and 26 False Negatives. 
Interestingly, using booster="gblinear" as opposed to the default booster="gbtree" dramatically decreased the FPR and increased the FNR. This indicates that the boosting technique is really the strength of XGBoost, as a linear booster did not distribute its false predictions evenly between the FNR and FPR. 

Findings:
Charlie's multi-layer perceptron classifier and my XGBoost learner had vastly different FNR and FPR values, given the same preprocessed data. Charlie's performed much better in the FPR objective and mine performed much better in the FNR objective. This indicates that neural networks, specifically MLP classifiers, tends to favor false positive prediction at the risk of accuracy while XGBoost favors even distribution of the FNR and FPR as well as high accuracy.  Additional improvements can be made to our learners by continuing to tweak the hyperparameters to achieve a particular FNR, FPR, and accuracy, as well as more advanced preprocessing techniques (normalization, removing noise, principal component analysis, etc.). 

**Action Items:**
| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| --- | ----------- | --- | ----------- |----------- |
| Review Titanic Dataset and Preprocessing/Hyperparameter Tuning Techniques | Completed | 9/15/2021 | 9/22/2021 | 9/16/2021 |
| Titanic ML Learner Predictions| Completed | 9/15/2021 | 9/22/2021 | 9/17/2021 |
| Create Subteam Slack | Completed | 9/15/2021 | 9/18/2021 | 9/15/2021 |
| Meet to Discuss Individual Learners' Performance | Completed | 9/15/2021 | 9/18/2021 | 9/18/2021 |