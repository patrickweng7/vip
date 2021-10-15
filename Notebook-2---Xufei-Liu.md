This is the start of the second notebook. First notebook found [here](https://github.gatech.edu/emade/emade/wiki/Notebook-Xufei-Liu).


= Fall 2020 =
Subgroup 2 (Bootcamp): [[Notebook Aryaan Anuj Mehra|Aryaan Mehara]], [[Notebook Bernadette Gabrielle Santiago Bal|Bernadette Bal]], [[Notebook Jon Greene|Jon Greene]], [[Notebook Han Mai Nguyen|Hannah Nguyen]]

Subgroup 2 Github Page: https://github.gatech.edu/amehra37/Titanic_ML_Group2

== August 19, 2020 ==

====== '''Lecture Notes:''' ======
* Genetic algorithms depend on previous generations/populations and will eventually create the best individual who's fitness is optimized.
** Initialize the population, then evaluate and select the best ones. Continue until you find the best one.

* Key words:
** Individual - one solution to the problem
** Population - set of individuals who we are altering properties of
** Objective - the problem that you are trying to solve through minimizing/maximizing
** Fitness - relative; how well is this individual compared to others?
** Evaluation - function that "computes" objective of individual
** Selection - survival of the fittest, prefers better individuals
*** Fitness proportionate - greatest fitness = higher chance of selection
*** Tournament - randomly pair individuals and the winners will reproduce

* Single Mate Crossover vs Double Point crossover to exchange mating between individuals
* Mutate: Random modifications to individuals to maintain diversity
* '''Actions:''' Randomly initialize, determine fitness, select parents, perform crossover, perform mutation, determine fitness, etc...
* '''One Max Problem Example''' - Either 0 or 1 with 100 values. We want an individual of all one's.
** Fitness = sum of all numbers to get objective value, we want to get to 100 

====== '''Lab 1 Notes:''' ======
* '''One Max Problem'''
** 1) Create FitnessMax using base.Fitness class and Individual. Base fitness can either be (1.0,) for maximizing or (-1.0,) for minimizing.
** 2) We create the random generator for the numbers in the population and make our individuals use that random generator 100 times. For the population, we create it to call the individual function multiple times.
** 3) We create the evaluation function. In this case it's just a sum.
** 4) We have 4 functions defined in the toolbox for evaluating, mating, mutating, and tournament selection.
** 5) For the genetic algorithm, we start by creating a population and use a loop for every generation
** 6) We use a selection process to choose the best individuals for mating
** 7) We mate the individuals and create children, then mutate the children slightly if necessary to keep diversity.
** 8) Now we take over the original population and replace it with all of the offspring.[[files/N Queens Graph.png|thumb|286x286px|A graph of the N queens problem from lab 1.]]
'''N Queens Problem vs One Max Problem'''
* For the N queens problem, we want to minimize conflicts between queens, so for base.Fitness we use (-1.0,).
* Our evaluate function counts queens on each diagonal
* We define our own cross-over function to swap pairs of queens.
* The idea is the same as the One Max Problem, but we tweek the functions for mating, selection, evaluation, etc.
* For mutation function I swapped the queens in a different way in the list.

====== Action Items ======
{| class="wikitable"
!Task
!Status
!Assigned Date
!Due Date
!Date Completed
|-
|Create Wiki
|Complete
|8/19/2020
|8/26/2020
|8/19/2020
|-
|Join Slack
|Complete
|8/19/2020
|8/26/2020
|8/19/2020
|-
|Complete Lab 1
|Complete
|8/19/2020
|8/26/2020
|8/24/2020
|}

== August 26, 2020 ==

====== Lecture Notes ======
'''Genetic Programming'''
* Instead of putting individual through an evaluator to get a score, the individual consumes data and returns the output
** The output will be what is evaluated
** In example, the function is Y=X^2, and we evaluate the final output.
* Tree representation
** We use tree structure with nodes as primitives (functions) and leaves as terminals (parameters)
** The input is a terminal and the output is the final function value?
** Lisp Pre-ordered Parse Tree
*** Operator followed by inputs (such as [+,*,3,4,1] is 3*4+1 = 13)
*** We start with the top and work our way out
*** Crossovers in tree-based GP is exchanging subtrees
** Crossovers work by exchanging subtrees
** Mutation - insert notes/subtree, delete, or change node/subtree
* Example: Symbolic Regression
** Evolve solution to y=sin(x) using primitives +,*,-,/
** Terminals include integers and x (a variable)
** Evaluating the tree:
*** Feed input points into function to get outputs and then run f(X)
*** Measure the error through MSE or SSE to see how close it is
** Which primitives would make it easier?
*** Power, Factorial, Sin, etc.
'''Lab 2 Part 1'''
* Added primitive np.maximum and np.square
* Tried out three mutate functions (mutShrink, mutNodeReplacement, mutEphemeral)
[[files/MutUniform.png|none|thumb|Original mutUniform function]]
[[files/MutShrink.png|none|thumb|Tested results with mutShrink function.]]
[[files/MutNodeReplacement1.png|none|thumb|Visual for mutNodeReplacement graph.]]
[[files/MutEphemeral.png|none|thumb|Graph for mutEphemeral.]]

====== Action Items ======
{| class="wikitable"
!Task
!Statuss
!Assigned Date
!Due Date
!Date Completed
|-
|First Half Lab 2
|Complete
|8/26/2020
|9/2/2020
|8/31/202
|}

== September 2, 2020 ==

====== Lecture Notes ======
'''Multiple Objectives/MO in MOGA and MOGP'''
* Supplying a population of solutions and not just single objectives!
* Translation of vector of scores from evaluation to a fitness value
* Gene pool
** Set of genome that are evaluated
* Evaluation/Scores
** True positive: How often we identify the desired objective
** False positive: How often we mistakenly identify something else as the desired objective.
** Objective: Set of measurements to score the genome/individual
*** Objective space: Set of objectives
* [https://en.wikipedia.org/wiki/Confusion_matrix Classification Measures] (Wiki Page linked)
** Data set has positive and negative samples that go through a classifier
** Through that you get Confusion Matrix with actual positives and negatives, with type 1 and type 2 errors. 
** Type 2: False negative, False positive: Type 1 error
** Visit the wiki [https://en.wikipedia.org/wiki/Confusion_matrix confusion matrix] page to refresh notes.
** '''Sensitivity/True Positive Rate/Hit Rate/Recall''': Number of TP/P = TP/(TP+FN), bigger is better
** '''Specificity (SPC) or True Negative Rate (TNR)''': TN/N = TN/(TN+FP), bigger is better
** '''False Negative Rate (FNR)''': FN/P = FN/(TP+FN), smaller is better
** '''Fallout/False Positive Rates''': FP/N = TN/(FP+TN) = 1-TNR = 1 - SPC, smaller is better
** '''Other measures'''
*** Precision/Positive Predictive Value: TP/(TP+FP), bigger is better
*** False Discovery Rate: FDR = FP/(TP+FP) = 1-PPV, smaller is better
*** Negative predictive value (NPV): TN/(TN+FN), bigger is better
*** Accuracy: ACC = (TP+TN)/(P+N), bigger is better
* '''Fitness Computation'''
** In the objective space with two objectives, we can have a graph. You can use objective functions like MSE, cost, complexity, TPR, etc.
** We give each individual a point in the objective space, called the '''phenotype''' (expression of genes) vs genotype
** '''Pareto Optimality:''' Individual is Pareto if no other individual outperforms the individual on all objectives. Set of all Pareto individuals is the Pareto frontier, and all represent unique contributions. We want to favor these individuals
*** We want to discover the true Pareto frontier.
'''NSGA II''': Nondominated sorting Genetic Algorithm II
* We separate population into nondomination ranks
* Individuals chosen using binary tournament, with lower rank beating higher rank. If same rank, we break tie with crowding distance (sum of distances of other points to this one) with higher distances winning, since it's further away.
* Strength S: How many in the population it dominates
* Rank R: Sum of S's of individuals that dominate it, nondominated if R = 0.
* We calculated the distance to find o^k and R+1/(o^k + 2) for fitness.
'''Lab 2'''
* Changed the mutation function to MutShrink and increased generation number.[[files/Original Pareto Front 1.png|none|thumb|Original pareto front with mutUniform.]][[files/Graph with mutShrink.png|none|thumb|This is the graph with the mutShrink function before we create the pareto front.]][[files/Imagejfdk.png|none|thumb|New pareto front when changed to mutShrink for mutation function.]]

====== Action Items ======
{| class="wikitable"
!Task
!Status
!Assigned Date
!Due Date
!Date Completed
|-
|Second half of Lab 2
|Complete
|9/2/2020
|9/9/2020
|8/30/2020
|-
|Email Jason with ratings
|Complete
|9/2/2020
|9/9/2020
|9/2/2020
|}

== September 9, 2020 ==

====== Class Notes ======
* Talked about titanic example, trying to create a better machine learning algorithms
** Need same pre-processed feature data, multiple objectives have to be co-dominant results
* Don't want data that has missing data, so we need to clean the dataset.

====== Group Meeting: ======
Meeting on 9/12/2020
* Aryaan Mehra helped with two versions of feature engineering to try and clean the dataset.
* We agreed to explore on our own to see if we can create a better dataset for ML
Meeting on 9/14/2020
* Decided to use the dataset that I cleaned for us to use machine learning on.
* I dropped all columns except for survived, age, sex, and Pclass for the titanic data.
* We decided to try different ML models on the dataset that I cleaned.

====== Personal Exploration ======
'''Contributions to the group:'''
* Cleaned dataset by dropping fare, embarked, sibsp, parch, cabin, name, and ticket.
* With the training and smaller test data, I got 0.844 in the test using the decision tree classifier.
* Jon decided to use an MLP model while Bernadette used Gradient boost model.
* Jon helped create a graph with our points as a makeshift Pareto front.
[[files/Confusion matrix 1.png|thumb|Confusion matrix from a Gaussian NB model using my cleaned data.]]
'''Modeling and test runs:'''
* Unfortunately, the models aren't as accurate when submitted to kaggle, with only about a 0.74 test rate.
* Link to personal exploration file: https://github.gatech.edu/amehra37/Titanic_ML_Group2/blob/master/Personal%20Exploration%20Xufei.ipynb
* Tried GaussianNB, decision tree classifier, random forest classifier, svm.SVC, NN (neural network), KNeighbors classifier and decided to stick with decision tree classifier.
*[[files/Titanic Pareto v2 jg.png|thumb|Pareto front for our group with models from cleaned data.]]Decided to use Gaussian NB model to get about 0.75 on Kaggle.
'''Self Graded Rubric:'''
* Name and contact: 5
* Teammate names: 5 (linked above)
* Neat, legible: 5
* Organization: 5
* Updated weekly: 5
* Group topics: 5
* Other individuals: 4
* To do items: 5
* To do consistency: 5
* To do cancellation: 4
* Level of details: 5
* References: 2
* Useful resource: 4

====== '''Action Items''' ======
{| class="wikitable"
!Task
!Status
!Assigned Date
!Due Date
!Date Completed
|-
|Self Grading Rubric
|Complete
|9/9/2020
|9/16/2020
|9/13/2020
|-
|Email and meet team
|Complete
|9/9/2020
|9/16/2020
|9/12/2020
|-
|Kaggle machine learning titanic
|Complete
|9/9/2020
|9/16/2020
|9/15/2020
|}

== September 16, 2020 ==

====== Class notes: ======
* Presentation Notes: Have a title slide
* Graphs:
** Have a title, labels, clearly readable
** Pareto front lines go to appropriate direction for min vs max problems
** Make sure you include page numbers during the presentation.
* You can have text on slides since presentations must be "stand-alone".
* However, don't present too long. It'll also go on our wiki page.

* On the bootcamp subteam page, you can look at the wiki and see past results from other members in past teams.
* No restrictions on primitives and operators! Can set it up as primitive tree, regression problem, etc.
'''Subteam Meeting 9/16/2020:'''
* Decide to create outline of presentation before the meeting
* Next meeting to be held on Saturday to determine how to progress
* Third meeting on Monday to finalize PowerPoint and presentation
* For now, we'll individually look at potential primitives for our project until the next meeting.
'''Subteam Meeting 9/20/2020'''
* Divided up slides to work on for presentation: I will work on ML slides and confusion matrices
* Aryaan had created template for GP modeling
** I will be exploring/adding and taking away primitives and mutation functions
* Aryaan has created the pareto front for the graph which I will be editing and changing
'''Subteam Meeting 9/23/2020'''
* Went over powerpoint which is linked [https://docs.google.com/presentation/d/1PjjLMWpuzzlQbO89IK7YysEGPhbUQ1qgWy0L_YwcQcU/edit?usp=sharing here]
* We still need to create a csv file with our data points to upload, however, the pareto front for GP is complete.

====== '''Personal Exploration:''' ======
* Looked at past presentations for ideas on how to modify titanic data
** For our GP model, here are possible evaluation functions:
*** Summing up the FP/FN and trying to minimize that
*** Do we want to prefer one accuracy over the other? Just look at FP or FN?
*** Our mutation function: mutNodeReplacement or mutUniform perhaps?
*[[files/XufeiParetoFront.png|thumb|1x1px]]Machine learning models
** Added slides with machine learning confusion matrices and names of matrices.
** ML Models used with resulting accuracy score[[files/ParetoFront5.png|thumb|Pareto front that I designed in jupyter notebook using the 8 ML models.]]
*** Decision Tree: 0.8508474576271187
*** K Neighbors Classifier: 0.7796610169491526
*** Support Vector Machines: 0.7932203389830509
*** Neural Network: 0.8338983050847457
*** Random Forest Classifier: 0.8440677966101695
*** Gaussian NB: 0.7694915254237288
*** Stochastic Gradient Descent: 0.8338983050847457
*** Passive Aggressive Classifier: 0.7288135593220338
** Tweaked and adjusted Pareto front to make it look nicer. Result is what I've created on the side.
GP Models:
* Aryaan created a template for GP modeling which Jon and I have been working on tweaking by adding more primitives. Unfortunately, it seems that the more we add, the lower the AUC gets. We have linked to it here: https://github.gatech.edu/amehra37/Titanic_ML_Group2/blob/master/Titanic_GP.ipynb

====== Action Items: ======
{| class="wikitable"
!Task
!Status
!Assigned Date
!Due Date
!Date Completed
|-
|Meet with team to work on titanic model
|Complete
|9/16/2020
|9/23/2020
|9/20/2020
|-
|View past presentations for bootcamp
|Complete
|9/16/2020
|9/23/2020
|9/18/2020
|-
|Explore new primitives and ideas for model
|Complete
|9/16/2020
|9/19/2020
|9/21/2020
|-
|Work on new evolutionary model for titanic
|Complete
|9/16/2020
|9/23/2020
|9/20/2020
|-
|Create presentation for subteam presentations
|Complete
|9/16/2020
|9/23/2020
|9/22/2020
|}

== September 23, 2020 ==

====== Class Presentations/Notes: ======
* '''Subgroup 2: (Us)''' 
** Presented first
** Should adjust for tournament selection since it is mostly for one objective instead of multi-objective.
* '''Subgroup 1:'''
** Used genHalfAndHalf with mutUniform for their loosely typed tree that returns a float. One point crossover mating.
** ML algorithms was not as good at GP accuracy, but ML was better at balancing false positives and negatives.
* '''Subgroup 3:'''
** Strongly types primitives with override for 0's with NSGA II selection with single point crossover and mutUniform. The AUC under the MOGP Pareto front is lower and more varied when compared with the ML AUC.
* '''Subgroup 4:'''
** Used NSGA2 and one point crossover, along with mutUniform. Had 200 generations.
** Genetic programming had lower AUC as well.

====== Action Items: ======
{| class="wikitable"
!Task
!Status
!Assigned Date
!Due Date
!Date Completed
|-
|Clone emade repository
|Complete
|9/23/2020
|9/30/2020
|9/28/2020
|}

== September 30, 2020 ==

====== Class ======
* '''EMADE: Evolutionary Multi-objective Algorithm Design Engine'''
** Combines multi-objective evolutionary search with high level primitives to automate process of designing ML algorithms
** To run emade...
*** Start at top level directory to run
*** Input file: xml document to configure moving parts in EMADE
*** Configure a MySQL connection, running locally server can be localhost.
*** Objectives: Names used as columns in database with the weight specifying if it should be minimized or maximized
**** Evaluation function specifies the  name of method in src/GPFramework/evalFunctions.py
**** Achievable/goal is used to steer the optimization, lower and upper bounds it.
*** We can make the memory limit low (2-3) per worker as EMADE is resource intensive.
*** Evolution parameters 
**** Controls "magic constants" or hyperparameters
***** Combines the elite pool with the offspring
**** Adding -w makes you a worker instead of starting a master process,
*** We need about 20 minutes to an hour, etc to get successful completion and see evolution.
* Run EMADE as a group with 1 person setting up the sql server as the master process
** Run for many generations
** Play with SQL, play with database
** Plot non-dominated frontier at end of run to compare with ML and MOGP assignments
** Make other plots and figures to analyze EMADE running to find successful trees
** Presentation on Monday the 21st

====== Action Items: ======
{| class="wikitable"
!Task
!Status
!Assigned Date
!Due Date
!Date Completed
|-
|Clean Notebook/Revise
|Complete
|9/30/2020
|10/7/2020
|10/3/2020
|-
|Configure mysql serve (5,6,7)
|Complete
|9/30/2020
|10/7/2020
|10/5/2020
|-
|Download/install git-lfs
|Complete
|9/30/2020
|10/7/2020
|10/5/2020
|}

== October 7, 2020 ==
'''Class'''

Went over basic download instructions for getting emade and mysql up and running on computer. Class was just to pop in and pop out for any questions that any of us might end up having.

'''Subteam Meeting 10/12/2020'''
* Tried to connect to Aryaan's mysql server to work
* Examined EMADE file
* Reviewed last recording of bluejeans to see which items to change
* Aryaan said he'd talk to Jason to fix the issue

'''Subteam Meeting 10/13/2020'''
* Issue was that we were not connected to campus vpn
* We all managed to connect to the server and ran EMADE for 30 generations
* Bernadette created the final powerpoint: https://docs.google.com/presentation/d/1tLoWIUuHeTLkjpWgDoTORUlnknGVHr7sNlDJSx1zsXI/edit?usp=sharing
'''Action Items:'''
{| class="wikitable"
!Task
!Status
!Assigned Date
!Due Date
!Date Completed
|-
|Meet with Subteam to Discuss emade preparations
|Complete
|10/7/2020
|10/14/2020
|10/12/2020
|-
|Connect to Aryaan's sql server
|Complete
|10/12/2020
|10/14/2020
|10/13/2020
|}

== October 14, 2020 ==

====== '''Class Meeting''' ======
* Office hours to go over some issues we had
* We needed to install a different version of mysql and finish up the powerpoint
* Might change some of the parameters of the emade folder.
[[files/Image177.png|thumb|443x443px|Headless chicken run with improvements and changing the code for the probabilities.]]

====== '''Subteam Meeting 10/15/2020''' ======
* Analyzed data from the first EMADE run
* Ended up with a messed up Pareto front but we fixed it later
Decided to change some parameters for another run
* Increased probabilities of headless chicken crossovers and decreased probabilities of some mutations

====== '''Personal Contribution:''' ======
* Added to the powerpoint to create the conclusion and comparison slides
* Final presentation: https://docs.google.com/presentation/d/1tLoWIUuHeTLkjpWgDoTORUlnknGVHr7sNlDJSx1zsXI/edit?usp=sharing
* Practiced presentation by myself and timed myself to see how long the presentation lasts.

====== '''Action Items:''' ======
{| class="wikitable"
!Task
!Status
!Assigned Date
!Due Date
!Date Completed
|-
|Modify Emade File
|Complete
|10/14/2020
|10/19/2020
|10/17/2020
|-
|Finish Final Powerpoint
|Complete
|10/14/2020
|10/19/2020
|10/18/2020
|-
|Rehearse Presentation
|Complete
|10/18/2020
|10/19/2020
|10/19/2020
|}

== October 19, 2020 ==

====== Class Presentations ======
* Stocks subteam
** Created CEFLANN Architecture and looked at implementing Keras NN in EMADE
** Looking at building portfolios using GP and EMADE
** Used technical indicators such as traders heuristics, price, volume, open interest, etc
** Did research on multiple potential indicators
** CEFLANN works quicker than other ones
*** Preprocessed data same way the paper did, with dat afrom Yahoo Finance for S&P 500 data
*** Ran regression on EMADE from technical indicators as features and let it go for 85 generations
*** Potential other EMADE primitives with ML models and look at other time windows
*** First semester members:
**** Create own technical indicator primitives for EMADE for the SPY ETF stocks dataset
**** Get idea from papers and implement them to EMADE adapting
** Bootcamp subteam 3
*** Dropped name and ticket for data, and created the "Deck category"
*** Applying machine learning models
*** AUC for ML models: 0.2541
*** Strongly typed GP with logical, relational, algebraic primitives with csOnePoint function for mating
**** mutUniform function for mutating
**** Minimized FNR and FPR
*** ML EMADE model
**** Created 3D pareto front graph
*** Challenges:
**** Used NSGA II
**** One-hot encoding
**** Bias towards dominant individuals
** Modularity
*** Try to abstract ML models even more
**** Automatically defined functions
**** ARL (Adaptive representataion through learning) introduces modularity into EMADE
***** Once they find good secion of tree, they make it a new primitive that can be used across the population
*** Future work
**** Want to see how ARLs perform on non-trivial dataset with more signal and spatial primitives 
**** New heuristics to add new ARLs with Differential Fitness, more intelligent ARLs, etc.
**** Implement more intelligent ARLs that can search through population for individuals
** Bootcamp Subteam 1
*** Dropped Name and Ticket and accounted for missing values
**** Split data into training and testing sets wiht 70% and 30% split
**** For ML they printed out all the confusion matrices
**** The MOGP had adding, subtraction, multiplication, negation, max, and min with 1 point crossover and mutUniform
**** For EMADE, they ran 17 generations with 311 final valid individuals, with 235 individuals on the pareto front
***** Used 5-fold cross validation on titanic training data
***** Pareto front is really, really small...?
** NLP NN
*** "Evolutionary Neural AutoML for Deep Learning"
*** They use tree based networks instead of graph based structures
*** Add in terminals and terminal mutations which will take a longer runtime
*** Allow one layer to branch into multiple layers in a tree structure
**** Example learner ran for 182 generations
*** No difference between using single point crossover vs two point crossover
*** Adaptive mutation function: Reduce mutations of good individuals!
*** Worked on adding primitives for the Chest X-Ray stuff
*** Worked on multiple different datasets to test their NN models
*** Potential BERT embeddings in the future
** Bootcamp subteam 2
*** Us! 
*** Fix EMADE data set to be like our other tested data
** ezCGP
*** Uses a block structure
*** They use Gaussian Noise, various types of pooling layers, and convolutions with refactored transfer learning pipeline
*** New structure using CIFAR10 to train all individuals in full (which takes a really long time)
*** Working to reduce computation time
**** Super-convergence runtime reduction
**** Future: better mating methods and better network architecture seeding
** Bootcamp subteam 4
*** Used NSGA II with mutUniform and cxOnePoint
*** 200 generations with 50 individuals elected each time, and AUC of 0.16.
*** 28 generations with 0.0277 area under curve which is really, really low!
*** EMADE < GP < ML for AUC

====== '''Action Items:''' ======
{| class="wikitable"
!Task
!Status
!Assigned Date
!Due Date
!Date Completed
|-
|Make Rank Lists of Subteams
|Complete
|10/19/2020
|10/26/2020
|10/23/2020
|}

== October 26, 2020 ==

====== Class Meeting ======
* Part of ezCGP subteam
* Joined their call after to figure out what we need to do
** Plan to meet once on Saturday and once on Monday to get crash course into ezCGP information
** Need to clone git repository and do initial set up with git accounts before Saturday meeting
[[files/EzCGP nodes.png|thumb|Basic overview of an example tree for ezCGP cartesian tree.]]
====== Subteam Meeting 10/31/2020 ======
* From 10:00-11:00
* Intro to CGP (Cartesian genetic programming)
** A tree on it's side, DAG (Directed acyclic graph) with not too many inputs, only left to right
** If just DAG structure and not DAG cartesian structure, you can have multiple inputs and rows linking to each other, but only in one direction, sometimes can be squished into one row instead of multiple rows.
*** We only use one row
** Features: reusable nodes, fixed lengths, inactive notes
*** To figure out inactive nodes, work backwards from the output since it only goes in one direction. 
*** Gets rid of bloat. 
** Implementation (Accepted ones)
*** 1 row with main columns for nodes 
*** "1+4" evolutionary strategy
**** No mating (too destructive), but get for mutant offspring from 1 parent, only mutate UNTIL we get to an active node 
* Intro to ezCGP
** Custom DAG, custom primitives/data types
*** Have not gotten to custom features yet 
** [https://ieeexplore.ieee.org/abstract/document/6815728 Original Paper] 
** We have a main output node and we count it by indices 
** Rather than random numbers floating around, we have a table of random values for the blocks to choose, grabs hyperparameters from list of arguments
*** Now we only have to pass in data, since primitives already in this bank
**** True mutate to false, 2 to another integer, etc with each having own way to mutate 
**** Arguments might not be used, might be used multiple times 
** Main node is a dictionary for strongly typed primitives  
** Blocks
*** We have blocks of code where data preprocessing happens in first chunk and data classification in second chunk 
*** Thus there are sub genome structures with own rules, parameters, etc. for more customizability 
*** We define the order but the computer decides most of the custom components 

====== Personal Exploration: ======
* Set up SSH Keys for windows which was the hardest part
** Had to look up windows guide here: https://phoenixnap.com/kb/generate-ssh-key-windows-10
** Downloaded git bash in order to install SSH
** Changed python version to an earlier version that works with open ssh

====== Action Items: ======
{| class="wikitable"
!Task
!Status
!Assigned Date
!Due Date
!Date Completed
|-
|Set up public github account
|Complete
|10/26/2020
|10/31/2020
|10/28/2020
|-
|Set up dependencies/download packages
|Complete
|10/26/2020
|10/31/2020
|10/28/2020
|-
|Learn about intro to ezCGP
|Complete
|10/26/2020
|10/31/2020
|10/31/2020
|}

== November 2, 2020 ==

====== Class Meeting: ======
* Stocks team
** Caught up first semester students
** Added primitives to emade
** Working on editing data to find sources of error
* ezCGP (our subteam)
** Code review and and questions section after main meeting for new students
** Past meeting last week for new students on info
* NLP
** Working on coding and caught up new students
* Modularity
** Got statistical significance showing improvement for first ten generations for changes to tournament selection
** Looking into more literature to modify their ARL vs alternative selection
** First semester students got a lecture on modularity

====== Subteam Meeting: ======
* Need to scope out what the subteam has done in the past.
* First semester students notes:
** We have observations as rows and columns as various features, and the final y is the classification
** X => data with observations and features, eg 500 x 10
** Y => classification matrix, eg 500 x 1
*** We want to solve for XA + B = Y so that a0x0 + a1x1 _ + ...b = y + error where A is size 10 x 1 and is a linear transformation
** Neural network: A set of linear transformations and nonlinear functions
*** We have layers where each layer is a linear transformation and nonlinear functions
*** Data -> NN with 3 layers
*** X_input*A_0 + B_0 = X_0
*** X_0*A_1 + B_1 = X_1
*** X_1*A_2+B_2 = X_2
*** Each transformation can whittle down the matrix to lower dimensions
*** Non linear transformations with some linear components! Maps n dimensions to 1 dimension
** Images
*** The matrix are pixels corresponding to picture dimension and each value is between 0 and 255 
*** Next we do a computation with a kernel and image chunk to get a number
**** In turn, this replaces to get a "future space", but mapping image to a same size or smaller size
*** We apply more kernels to get another future map from this one, etc
*** Eventually, we hope the final classification is enough to get us a classification
* Logistics
** Make sure tensor flow is 2.x version
*** import tensorflow as tf
*** tf.keras.....
** Built model/graph first (DAG)
*** input = tf.keras.Input_Layer(image-size(500,600,5)) #defines shape
*** method = tf.keras.Conv2d(kernel=(3,3))
*** x = method(input) OR x = tf.keras.Conv2d(kernel=(3,3))(x)
*** This can be followed by multiple other laters
*** kernel is activation which is nonlinear function
*** For the last layer we have y = tf.keras.Conv2d(...)(x)
*** tf.keras.Model(input,y) to show where to start and where to stop

====== Subteam Meeting 11/5/2020 ======
* Returning students - set up a time to meet outside of class
* New students (me) - three experiements, try implementing the titanic data for ezcgp and for old students to use as benchmark.
** Compare to titanic emade results vs ezcgp
* Steps for getting started
** Save dataset into dataset folder and use datatools
** Next week we meet with Jason to figure stuff out

====== Personal Exploration: ======
* Read through old entries of ezCGP's meeting to see what they've accomplished.
** Link to subteam is [[Fall 2020 Sub-team Weekly Reports#EZCGP|here]]
** Originally in 2 small groups, one for research to look for papers to read for ideas and implementation. Another is for code maintenance to work on current code.
** Began runs on multi-gaussian test problems to make sure primitives work
** Migrated old primitives to new code development branch and created [https://github.com/ezCGP/ezCGP/wiki/Github-Flow:-Committing-Code guidelines]
** Learned new framework with Tensorflow2 and tried to seed high performing individuals into new framework
** Set up PACE-ICE to start trying to run new issues

====== Action Items: ======
{| class="wikitable"
!Task
!Status
!Assigned Date
!Due Date
!Date Completed
|-
|Review previous week's work
|Complete
|11/2/2020
|11/9/2020
|11/8/2020
|-
|Look into implementing titanic data into ezCGP
| --
|11/5/2020
|11/9/2020
|Moved to next week
|}

== November 9, 2020 ==

====== Class Meeting ======
* Stocks subteam
** Worked on sanity checks to compare results to paper
*** Paper explained functions used to explain trend score, but the table provided was not consistent
*** Focused on genetic algorithms with the truth data that they already had
*** Results are not currently lining up paper results
* EZCGP (Our subteam)
** Set up base and connection issues on Saturday for returning sub-members
** Try to have baseline scores done by next meeting
** Fixed some problematic methods since last time
*** No more tensor flow errors
* NN/NLP
** Adding to the code base with new targeting done on data sets so they'll have a few weeks for tests
** The bounding boxes researchers found that 8 out of 15 classes worked on xray dataset so it may not generalize well.
** Have been looking at Amazon product reviews on Kaggle.
* Modularity 
** First semesters are going to be assigned runs and experiments so they can begin analysis
** Continue doing runs from GTRI inspiration
** Selection method uses binary tournament
** Should also do a sanity check and need to compare to the benchmark

====== Subteam Meeting: ======
* Make sure PACE documentation sheet is updated
* Very lowkey, once PACE is working we'll be able to help as well
* Grant username access as well
* In future, make sure you branch off of 2020 repo to show your own work

====== Personal Exploration ======
* Working on installing SSH for windows
** Used gitbash and solved the issue
** Also updated Windows 10 so that I could install the SSH client
** Got PACE-ICE working through the documentation that was given out
* Downloaded a special software called WinSCP to make using PACE-ICE easier
** Link and information found here: https://docs.pace.gatech.edu/storage/winscp_guide/
** Notes for this:
*** You have to be on Tech's vpn on or campus to connect
*** For host name leave it as pace-ice.pace.gatech.edu
*** Leave the password section blank - a gui will pop up and ask you to input the password after it connects
** WinSCP works as a GUI method of navigating through PACE-ICE server so you won't have to do it on command line if you have a windows computer
** Makes it easier to download multiple files at once

====== Action Items: ======
{| class="wikitable"
!Task
!Status
!Assigned Date
!Due Date
!Date Completed
|-
|Review PACE Documentation
|Complete
|11/9/2020
|11/16/2020
|11/12/2020
|-
|Set up PACE on computer
|Complete
|11/9/2020
|11/13/2020
|11/12/2020
|-
|Begin researching ways to implement titanic data into ezCGP
| --
|11/9/2020
|NA
|Cancelled task to focus on MINIST Data Upload
|}

== November 16, 2020 ==

====== Class Notes ======
* Stocks subteam
** Calculated given trendscore with the stats in the paper but still not reproducible
** Going to email the people who wrote the article to see if they can reproduce the results
* ezCGP
** Started doing baseline runs, working on making a script that resubmits themselves
** Restricted to 10 hour run time
* NLP
** Only blocker is GPU system
** Uses notion for keeping track of work
* Modularity
** Going slower because they're limited by the individuals 

====== Subteam Meeting ======
* Assignment: Load in MNIST data set into numpy arrays and play around with it.
** See what you can decipher from MNIST data set
* Plans to continue running ezCGP trials and see if we can sort out bugs in the PACE-ICE system to get it to run

====== Subteam Meeting 11/19/2020 ======
* Finally got PACE-ICE working, and Rodd said he'd take care of training to rerun the simulations
** Plan to rerun with different seeds
** I'm also going to skim through and see if I can help with running a few simulations through PACE ICE
* Overview of final presentation
** Intro to ezCGP and code development progress
** More information on runs, experimental setup, and extra work done with seeding.
** Work to replicate emade on the ezCGP data to see if we get the same result
** Current benchmarks - maybe create visuals for them?
*** Also create visuals for genomes
* Need to get MNIST data in to the easy data sets for processing => talk about process for loading them in
** Creating an easy data set without downloading it through tensor flow
** Focus on block structure optimization
** Assigned to work on creating the Pareto plot for generational fitnesses to visualize how well it's doing
[[files/Overview of Process.png|thumb|An overview of the process we go through to classify handwritten digits taken from the site linked in the notebook.]]

====== Personal Exploration: ======
* Looked at information from MNIST dataset to prepare myself for downloading scripts.
** Information on loading in sourced from [http://rasbt.github.io/mlxtend/user_guide/data/loadlocal_mnist/ here]
** Overview
*** MINIST constructed from two datasets, consisting of a training set that has handwritten digits from 250 people
*** Half are high school students, half are from Census Bureau
*** Each feature vector has 784 pixels with 50000 images
*** Target variable is the respective handwritten digit from 0 to 9
* MNIST Information from Handwritten Digits classification information can be found [https://www.learnopencv.com/handwritten-digits-classification-an-opencv-c-python-tutorial/ here]
** Working on building a handwritten digit classifier using OpenCV
** We convert an input image into its features, then through a learning algorithm to a label assignment as seen in picture to the side
** Step 1: Deskewing or Preprocessing
*** We align the images to a reference image, normally through a facial feature detector. 
*** Can apply a similarity transformation to an image such as fixing the slant of written digits so that it's no longer skewed
*** Done through using image moments that OpenCV provides
** Step 2: Histogram of Oriented Gradients (HOG) descriptor
*** We convert he image to a feature vector using HOG descriptor that you can read about [https://www.learnopencv.com/histogram-of-oriented-gradients/ here]
*** Tweak parameters and test to see which gives you the best results
** Step 3: Model training and Learning a Classifer 
*** For images, we can use SVM as classification algorithm which you can read about [https://www.learnopencv.com/image-recognition-and-object-detection-part1/ here]
* Pareto front code block
** Found past code done by [[Notebook Aryaan Anuj Mehra|Aryaan]] when we worked together from the bootcamp which I've added into my code block
** Link to code block here: https://codeshare.io/5eAJbJ
** Edited the Minimization section to turn it to maximization by changing the <c to >c on line 9 of the code since I will be maximizing the recall and precision of the ezCGP data
** Code takes in numpy array or data frame, and returns the information for us to plot the front.
** Steps for plotting the front also done by [[Notebook Aryaan Anuj Mehra|Aryaan]] and is linked here: https://codeshare.io/G8mgxZ
** In the future I will need to preprocess and clean the data so that I can use the Pareto front function to plot fronts for the generational data

====== Action Items: ======
{| class="wikitable"
!Task
!Status
!Assigned Date
!Due Date
!Date Completed
|-
|Look up information on MINIST
|Completed
|11/16/2020
|11/23/2020
|11/21/2020
|-
|Examine past Pareto front code from bootcamp
|Completed
|11/19/2020
|11/23/2020
|11/22/2020
|}

== November 23, 2020 ==

====== Class Meeting ======
* Stocks subteam
** Evaluated a couple of individuals using training from price data and technical indicators 
** Plan to finish technical indicator primitives in EMADE soon
** Contacted researchers of paper but did not get response
** Inverse shares correlate with price data
* ezCGP
** Working on doing their baseline runs to get data
** Gotten some individuals after 1 test run with 8 hours and visualized the individuals
** Almost done with the basic outline
* NLP
** Baseline runs for each data set with 2 runs for four different scenarios to accurately get targeting
** Worked on runs with original primatives
** Created documentations and new primatives for computer vision
*** Looked into future extraction methods and not sure how they fit in
*** How to combine primatives in three different files with any restrictions?
*** Spatial methods revolve around csv instead of numpy
* Modularity
** Planning to get 10 different runs before the final presentation, need to take in the data pair and return the data pair.
** Need to do more analysis to notice the changes in certain generations where irregularities were hpapening.
** Multiple ARLs happening when data pairs are merged
** Also working on MNIST (which I'm currently doing)
*** Might reach out to them for help
* Note: 20-25 minute presentations with 5-10 minutes of presentations

====== Subteam Meeting ======
* Daniel led the meeting since Rodd was absent
** Summary of results from last week and reiterated who is working on which slides
* Working on finishing our current slides
* Moved next meeting to Saturday at 2PM
[[files/Pareto Front for Generation 1.png|thumb|171x171px|Pareto front for the first generation with associated AUC with points]]

====== Subteam Meeting 11/28/2020 ======
* Met to update each other on final progress and finish up the final presentation
** In depth steps to PACE-ICE and setup is complete on the notion page [https://www.notion.so/PACE-ICE-GPU-for-Ez-CGP-8be7a2e57c6649229f36505d093952dd here] done by Hoa Luu.
*** Decided on PACE-ICE since it was fast and had better memory usage
** Need to finalize results from the runs so that we can analyze the data for Daniel to look at
** Need to pass in metrics for tensorflow to record as we're training the model through attributes that help the fit method.
* Our final presentation is linked [https://docs.google.com/presentation/d/1cbx_daOsFvMZIgBQvVnmiJBmmm-Lvha61Mfe68Ej7c8/edit?usp=sharing here]
** Future goal - to mix ezCGP with emade since emade uses DEAP, to see the comparisons in the results
[[files/Pareto front for generation 1 without points.png|thumb|167x167px|The pareto front for the first generation with AUC and without the points]]
====== Personal Exploration ======
* Completed the script for the MNIST data loader
** Followed the instructions from  [http://rasbt.github.io/mlxtend/user_guide/data/loadlocal_mnist/ here]
** We had to install mlxtend so that we could run the code, using "pip install mlxtend" in the command line
** Other than that we were able to directly copy and paste the code in the command line
** Worked with Angela Young to finish this script together
* Completed successful first pareto plot for generational fitness
** Complete script for generating the pareto front is seen [https://codeshare.io/50KjEN here]
** Pareto front for the first generation is seen to the side
** Notice the first AUC is 0.92 for the first generation
** To preprocess the data, first we multiply by -1 to get positive values
*** This is because ezCGP works to minimize values, so we multiplied our values by -1 so that we could minimize instead of maximize
** Need to work to just plot the line and get rid of the points on the graph and then graph multiple lines on the same plot.
*** Issue: A lot of our final AUCs and Pareto fronts past the second generation are so similar that there is no visual difference
*** Solution: I will only graph the first generation and ninth (final generation) on the plot in the final PowerPoint which is below
** Also want a chart for just the AUC or note the final AUC
*** Issue: The AUCs are similar and we cannot see a difference
*** Solution: I deleted the first generation and showed the change in AUC excluding that. Unfortunately, the AUCs for the second and third generation are the same, and the AUCs for generations 5-9 are also the same.
Completed PowerPoint slide with the three main charts is shown below
[[files/PPT slide I worked on.png|center|600x600px|Contains the three graphs that I did in matplotlib so that you can see how the AUC changes with the generations]]
* Personal Analysis of my Designed Graphs:
** Notice that there is a significant change between generations 1 and 9
** With generation 1 included, we see a large jump in AUC from 1 to 2 relative to other jumps
*** Typically, this would be a small jump, but our original model did so well it was hard to improve
*** The good performance is due to the effect of transfer learning blocks that were developed in the past
** Generations 5-9 were the same pareto front and AUC, this may be because there was nothing else to improve and it was already very close
** Without generation 1, we can see some improvement from generations 2 to 5. At the very least, our learning model ended up progressing and doing better until it was too difficult to have it do any better
* Final AUCs of all of the curves are listed below:
** Generation 1: 0.920881502437016
** Generation 2: 0.981823413391421
** Generation 3: 0.981823413391421
** Generation 4: 0.981861232839681
** Generation 5: 0.9819120285577867
** Generation 6: 0.9819120285577867
** Generation 7: 0.9819120285577867
** Generation 8: 0.9819120285577867
** Generation 9: 0.9819120285577867
* You can find all the code I used and files of generational data which I pulled from PACE-ICE here: https://github.com/ezCGP/ezCGP/tree/master/post_process/ezCGP%20pareto%20front
** I committed this to the ezCGP repository under post_process
* Presentation Notes I created for the slide:
**Results:
*** Left graph, we have the pareto front of the first generation and ninth generation for comparison, with their associated AUC’s underneath the graph. As you can see, there was an improvement
*** We’re trying to maximize both recall and precision so we want higher AUCs
*** Issue: The change in the AUCs and fronts don’t have a visual difference past the first generation, as we can see in the graphs on the right
*** First generation jumps up very high with AUC very quickly
**** Included second plot that doesn’t have the first generation so you can see the small increases in AUC from generations 2 to 5
**** From generations 5 to 9, however, there is no increase in the AUC
** Analysis
*** Graph 1:
**** We evolved with a 3rd objective score (accuracy) which was poorly calculated so we removed it from our presentation.
**** This is why it Generation 9 isn’t strictly dominating Generation 1.
*** Small change in AUC/almost no change?
**** A small population size of 20 made it hard to push the pareto front after it reached competitive scores, less diversity
**** We just started with really good performance...transfer learning block had a lot to do with that
**** The reason the auc didn’t improve much after the second generation was because our original model already performed really well
**** By the fifth generation the model couldn’t find a way to improve anymore

====== Action Items: ======
{| class="wikitable"
!Task
!Status
!Assigned Date
!Due Date
!Date Completed
|-
|MNIST data loader
|Complete
|11/23/2020
|11/30/2020
|11/27/2020
|-
|Plot first pareto plot for generationial fitness
|Complete
|11/23/2020
|11/30/2020
|11/26/2020
|-
|Create plot for multiple Pareto fronts
|Complete
|11/28/2020
|12/1/2020
|11/28/2020
|-
|Create plot for change in AUCs
|Complete
|11/28/2020
|12/1/2020
|11/28/2020
|-
|Run over analysis of fronts and data with Rodd
|Complete
|11/28/2020
|12/2/2020
|11/28/2020
|-
|Finish and polish notebook
|Complete
|11/23/2020
|12/2/2020
|11/30/2020
|}

== December 2, 2020 ==

====== Final Presentations ======
* '''Stocks Presentation'''
** Research paper inconsistencies
*** Trading signal calculated did not align with rules researchers specified
** Genetic labeling
*** Developed new training method to deal with inconsistencies
*** Developed new fitness function to treat time point independently to optimize profit per transactions
*** Compared two evolutions and looked at overlap for differences in results
** Technical Indicator Primatives
*** Tried simple moving average (used by paper)
*** Exponential moving average prefers recent days
*** Moving average convergence/divergence shows when and how often the direction signal line gets crossed
*** Stochastic Oscillator and relative strength is also used
*** Used EMADE features to receive more complex inputs than just prices and return an OBV value for time range
** Developing more technical indicators
*** Bollinger bands (moving average and upper/lower band)
*** CCI (momentum based and measure price relative to average time)
*** MFI (momentum and volume based to measure flow of money)
** Feature Data EMADE run
*** Truth values from paper don't align with current results, results may be a fluke
*** Second run used stream to features and worked on 10 days at a time, had issues with normalization
** Future work
*** Run Stream_to_Features on normalized data with different time ranges
**** Would smaller or larger time ranges be better?
*** Test granularity 
*** Explore with genetic labeling
*'''Neural Networks Presentation'''
**Uses an evolutionary approach to neural architecture search using EMADE
**Compared to Evolutionary Neural AutoML for Deep Learning
***Uses toxicity dataset and Chest X-ray
***They used AUROC but subteam used accuracy
***Used a combination of EMADE and NNLearner with tree based networks rather than graphs
****This increased the search space!
**Changes to Architecture
***BERT is pre-trained language model, and they used distilled version for less memory/faster processing
***In the future, want to do more with BERT
**New matting/mutations
***Single point and two point
***Adaptive mutation function
****Mutated bad individuals, decrease mutation chance for good individuals (which decreases randomness)
****In the future, use a logistic function to determine mutation probabilities depending on fitness in generations
***Work with PACE cluster and also worked with my subteam to look at EMADE on pace
****Some issues such as running out of space and user doesn't have permissions
***CV Primitives added 
****Adaptive mean threshholding
****Adaptive gaussian thresholding
****Otsu's Binarization (2D) which minimizes weighted within-class variance 
***Also began primitives documentation on Notion with 40 primitives and 17 variants
**Results and Goals
***Wikidetox dataset looked at 160k comments with classification of whether it is toxic or not
****Unbalanced since 9.5% of data was toxic
****Ran 24 generations
***Chest X-ray had 14 possible diseases looking at over 100k x-rays
****Trained with 70% of data, tested with 20%, validated with 10%
****Bounded box/YOLO architecture only looks at images once for improved speed
*****How much accuracy is sacrificed for this?
*****Unfortunately not all X-rays had this and some diseases are not localized to a single region
****Evolution didn't get to better models since they were simple. EMADE made tiny networks relative to others since it wants few parameters
*****In the future can add more mutations or modify the objective. Also possible to seed in larger models
***Amazon product reviews dataset
****Looking to determine tone from the text
****However, some of the results are in other languages
*****In future, possible to maybe build in a translator?
**Future Works
***Looking at Multi Task Learning
***Trying to make BERT layer beter
***Working on more datasets
***Trying coevolution
***Looking at more complex adaptive mutations
*'''ezCGP Presentation (Our Subteam)'''
**Notice the lack of diversity in our individuals
***Previous individuals could change brightness slightly but ezCGP liked the pass-through and transfer learning creates the best individuals
***Replacing the dense NN with the green blocks
*'''Modularity Presentation'''
**Working to abstract individuals to create blocks that can help the genetic process
***Goal is to have reusable blocks of code 
***Hopefully lead to novel solutions
***Based on Adaptive Representation through Learning (ARLs) 
**Currently:
***Search population for combination of parent and children nodes and choose combination based on cdf from frequency/fitness
***Next, abstract the combinations into single node
***This way, ARLs can wrap around each other and grow
**Experimentation on titanic dataset
***Has 40 generations with 10 trials with seeded runs (5 individuals each)
***Uses default parameters
***Differential Fitness: Different in individual and the most fit parent as a herustic
***Only find individuals with positive differential function values, modified cdf to value individuals with higher differences
**Results
***Big significance in generations 16-19 that converges with basline in future generations
***Pareto individuals used ARLs but they're mostly pass primitives
***Alternative selection method: Wants to increase probability of getting ARLs
****Individuals more likely to be picked depending on number of ARLs that they have in linear fashion 
****However, some individuals have bloat with over 40 ARLs
***Next, they restricted data pairs to only allow subtrees that return EmadeDataPair object such as filters and learners
****Unfortunately, no statistical significance
****Does not converge in the future but could be limited by titanic dataset
***Last data pair restrictions were merged with alternative selection to reduce bloat
****Most individuals only have 0,1,or 2 ARLs
****Performs worse overall but there is a small sample size
**Overall Results/Future
***Many generations have results from generations 10-20
***Overall, ARLs seem to be beneficial
***They may lower the std.dev in later generations
***In the future
****MNIST dataset
*****Looks like a lot of what we've done in our subteam
****Measuring diversity
****Heuristics, integrating ARLs and ADFs
*****Maybe compare individual's fitness with the population
*****ADFs (automatically designed functions) are more individualistic but also mroe evolable
****Modifying creation of ARLs
*****Increased targeting and isolation with abstract larger trees
****Increase genetic material