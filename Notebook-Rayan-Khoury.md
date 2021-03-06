
* Name: Rayan Khoury
* Email: rkhoury7@gatech.edu
* Mobile number: +1(678)789-7927

# **December 6th, 2021**
## **Meeting 16**
### Class Notes:
* Preparations for final presentations
* Had a small revision throughout the week to make sure of the content presented

### Presentation Notes:
* NLP: 
    * Word embeddings include Word2Vec, Glove, and Fasttext
    * Objectives are number of parameters and mean squared error of F1 score
    * NNLearner2 has two arguments
* Modularity:
    * Changed tuples into classes throughout infrastructure.
    * Merge stocks changes into ARL_Update branch
* Stocks:
    * Aiming to outperform a previous paper's work
    * They improved in all stocks significantly expect for JNJ

* Personal contributions:
I created and presented the slides titled "NN-VIP Setup Locally" 

|Task Description|Current Status|Due Date|Date Assigned|Resolved Date|
|---|---|---|---|---|
|Final Presentation|Complete|12/10/2021|12/06/2021|12/10/2021|

# **November 29th, 2021**
## **Meeting 15**
### Class Notes:
* Worked on my the final presentation with all the members
* Discussed with Pranav the literature review

### Personal Notes:
* The final presentation can be found here: https://docs.google.com/presentation/d/1kEOKk6Esu_CEE2FzyLRO6HDHYNDNhkNw4JH3fD7fRV0/edit?usp=sharing
* I completed slide 51
* Summary of Literature:
    * EMADE is based on DEAP genetic programming:
        * In DEAP, algorithms are expressed as trees and functions are expressed as nodes
        * N trees are selected to create new trees in each generation
    * Emade Neural Networks changes:
        * Network layer types are added from Keras as EMADE primitives
        * A layer tree is a pre-ordered representation of NNs
        * Neural networks can e created from LayerTree specifications which are also NNLearners

|Task Description|Current Status|Due Date|Date Assigned|Resolved Date|
|---|---|---|---|---|
|Practice for final presentation with the team|Complete|12/06/2021|12/03/2021|11/29/2021|
|Update Notebook|Complete|12/06/2021|12/05/2021|11/29/2021|
|Review the presentations given to us by the Camerons|Complete|12/01/2021|12/03/2021|11/29/2021|

# **November 22th, 2021**
## **Meeting 14**
### Class Notes:
* Was not able to attend class because I travelled for Thanksgiving

### Personal Notes:
* I continued reading the articles given to us. This week I focused on "Evolving Deep Neural Networks" (https://arxiv.org/pdf/1703.00548.pdf)
* Here are some notes I have taken from this paper:
    * NEAT, Neuroevolution of augmenting topologies, works on the evolution of neurons and it is low-dimensional as it adds neurons and does not result in errors
    * HyperNEAT is a technique for evolving large-scale neural networks using the geometric regularities of the task domain.
    * Changing primitives and tunning hyperparemters evolve blueprints and modules
    * CIFAR10 is initializing image embedding through pre-trained ImageNet, GoogLeNet and ResNet

|Task Description|Current Status|Due Date|Date Assigned|Resolved Date|
|---|---|---|---|---|
|Complete my final presentation slide|Complete|11/29/2021|11/26/2021|11/22/2021|
|Update Notebook|Complete|11/29/2021|11/23/2021|11/22/2021|

# **November 15th, 2021**
## **Meeting 13**
### Class Notes:
* Went through SCRUMS of each group where weekly updates were shared

### Personal Notes:
* I have read "Neural Networks to get a general understanding"
* Notes Summary:
    * LSTM Layer: Long-short Term Memory Neural Network Layer
    * 1D Convolutional Layer: Convolutional kernel convolved with layer input over single spatial dimension
    * 1D MaxPooling Layer: Maximum value of size for 1D data
    * GRU Layer: Gated recurrent unit
    * Dense Layer: Fully connected NN layer
    
|Task Description|Current Status|Due Date|Date Assigned|Resolved Date|
|---|---|---|---|---|
|Read "Concurrent Neural Tree and Data Preprocessing AutoML for Image Classification"|Complete|11/22/2021|11/19/2021|11/15/2021|
|Work on Notebook|Complete|11/22/2021|11/16/2021|11/15/2021|
|Learn more about neural networks|Complete|11/22/2021|11/22/2021|11/15/2021|

# **November 8th, 2021**
## **Meeting 12**
### Class Notes:
* We went through group scrums where each team talked about their weekly updates.
* Our team is working to get some infrastructure in place that allows us to do effective runs of EMADE for our final presentation.

### Group Notes:
* We were assigned the corresponding slides to complete for the final presentation. 

### Bootcamp members meeting with returning members:
* Bootcamp members met with Cameron W and Cameron B to get to know more about EMADE's NN-VIP version, along with explaining machine learning concepts.
* EMADE:
    * There are 2 primitive set types:
        * MAIN
        * ADF
    * EMADE object fit information into XML files 
    * handleWorker, setObjectives, myStr are helper functions used in EMADE
    * swap_layer, concat_healer, cx_ephemerals are mating?mutation functions.
* Neural Networks
    * Uses a set of layers where each layer has a set of nodes
    * Activation Functions Used:
        * Sigmoid
        * tanh
        * ReLu
        * Leaky
        * Maxout
        * ELU
    * Models that map inputs to output efficiently
    * Each layer in the network contains operation and activation segments
    * Types of layers: 
        * Dense
        * Convolutional
        * LSTM
        * Recurrent
* NN-VIP EMADE
    * Start by running the command launchEMADE.py input.xml
    * Initiate database, track population status, and handle queues of individuals using master algorithm
    * Connect to database and retrieve individuals that need to be evaluated using the worker algorithm

### Personal Notes:
* I read the following paper: https://dl.acm.org/doi/pdf/10.1145/3321707.3321721
* This paper focused on deep neural networks, their architecture, and experimental results
* I also read this article regarding neural networks: https://towardsdatascience.com/understanding-neural-networks-19020b758230?gi=1341d1e6f5c3

|Task Description|Current Status|Due Date|Date Assigned|Resolved Date|
|---|---|---|---|---|
|Read "Neural Networks to get a general understanding"|Complete|11/15/2021|11/08/2021|11/12/2021|
|Complete EMADE Setup|Complete|11/15/2021|11/08/2021|11/14/2021|
|Read the midterm presentation and prepare for final|Complete|11/15/2021|11/08/2021|11/10/2021|


# **November 1st, 2021**
## **Meeting 11**
### Class Notes:
* I was assigned to the Neural Architecture Search (NAS) sub-team.
* We went through group scrums where each team talked about their weekly updates.

### Group Notes:
* We were introduced to NAS by Cameron W and Cameron B and got to learn about what the team does.
* We were informed about the team's division of tasks and our team's preparation for the final presentation.

### Personal Notes:
* I setup EMADE locally and on PACE following the instructions found on the group's wiki page.
* Installed numpy, keras, and tensorflow as the standard conda version did not include them
* Was tasked to read the literature present on the trello board to understand the basics of what was happening in our team. These literatures include "Evolutionary Neural AutoML for Deep Learning", "Evolving Deep Neural Networks", "Concurrent Neural Tree and Data Preprocessing AutoML for
Image Classification", and "NEUROEVOLUTION OF NEURAL NETWORK ARCHITECTURES USING CODEEPNEAT AND KERAS".

|Task Description|Current Status|Due Date|Date Assigned|Resolved Date|
|---|---|---|---|---|
|Update Notebook|Complete|11/08/2021|11/01/2021|11/06/2021|
|Join the Trello and Slack channel|Complete|11/08/2021|11/01/2021|11/01/2021|
|Read the assigned literature and prepare notes and questions|Complete|11/08/2021|11/01/2021|11/01/2021|

# **October 25th, 2021**
## **Meeting 10**
### Class Notes:
* Final Presentation
* Presented our final research work
* Answered questions regarding our research findings
* Talked about graphs we used comparing our results
* Learned about the subteams, such as NAS, NLP, and stocks, among other teams, and their scope of work.

* Presentation: https://docs.google.com/presentation/d/1Rgt1bLAuUg87MrD0WF8Mro7PKvBSm4EcFExPrzp5_wU/edit#slide=id.p

|Task Description|Current Status|Due Date|Date Assigned|Resolved Date|
|---|---|---|---|---|
|Update Notebook|Complete|11/01/2021|10/25/2021|10/28/2021|
|Completed the preference assessment on canvas|Complete|11/01/2021|10/25/2021|11/01/2021|


# **October 20th, 2021**
## **Meeting 9**
### Class Notes:

* Workday with VIP alumni and returning students
* Worked out the master and worker processes with the team and debug remote connection errors
* Prepared the midterm presentation

|Task Description|Current Status|Due Date|Date Assigned|Resolved Date|
|---|---|---|---|---|
|Work on EMADE Presentation|Complete|10/25/2021|10/20/2021|10/22/2021|
|Update Notebook|Complete|10/25/2021|10/20/2021|10/21/2021|
|Completed remote connection|Complete|10/25/2021|10/20/2021|10/20/2021|
|Completed midterm presentation|Complete|10/25/2021|10/20/2021|10/24/2021|

# **October 13th, 2021**
## **Meeting 8**
### Class Notes:
* Emade Installation Session
    * Cloned Emade using Git Bash
    * Added Conda 
    * Created Conda virtual environment named "EnvironmentEmade"
    * Installed Emade dependencies using conda and git
    * Installed SQL
    * SQL connection to Aditi's server was successful 

|Task Description|Current Status|Due Date|Date Assigned|Resolved Date|
|---|---|---|---|---|
|Work on EMADE Presentation|Complete|10/20/2021|10/13/2021|10/15/2021|
|Update Notebook|Complete|10/20/2021|10/13/2021|10/14/2021|
|Write Python scripts|Complete|10/20/2021|10/13/2021|10/17/2021|
|Run EMADE on preprocessed Titanic Data|Complete|10/20/2021|10/13/2021|10/17/2021|

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
    * A distance to the kth nearest neighbour is calculated and a fitness of R+1/(????k + 2) is obtained
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
5.	Selection: Represents ???survival of the fittest???; gives preference to better individuals, therefore allowing them to pass on their genes
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

## **Lab 1 ??? Genetic Algorithm with DEAP**

### a)	One Max Problem
1. 	The objective of this exercise was to find a bit string containing all 1s with a set length using the DEAP python library.
2.	After installing deap and importing this library, we defined the name of the class, the inherited class, and the objectives. We then created a class and defined a tuple that represents a single objective up fir maximization.
3.	The tournament selection of 3 individuals method was used to preserve more varied traits in this population.
4. Then, we evaluate our population according to each individual???s fitness
5. The algorithm is then set to run for 40 generations 
6.	After crossing over the generation and defining statistics for our population, we print out the result to check the progress over time. Max increased from 65.0 at generation 0 to 100.0 at generation 39. In addition, average increased from 53.82 in generation 0 to 97.81 at generation 39.
Thus, we conclude that after running the code multiple times, the generation did not always reach the optimal max and average due to the random nature of initialization, crossover, and mutation.

### b) The N Queens Problem
1.	This exercise revolves around the determination of a configuration of n queens on a nxn chessboard such that no queen can be taken by another.
2.	First, we create the fitness and individual classes (first we use n=20)
3.	Next, we define our crossover. This problem will consist of partially matched crossover mating. It represents swapping a pair of queens??? positions between 2 parent individuals which is more effective in this scenario.
5.	Then, we move into our mutation function
6.	Shuffling indexes in this exercise is crucial as this represents the position of the queens on the chessboard, but cannot mutate or duplicate as this may lead to an out of bounds result.
7.	After defining the loop and running it for 100 generations, we change from a max of 16.0 in generation 0 to 9.0 in generation 99. 
8.	Along with min, and avrg, , max has been gone down according to the following graphs
### Action Items
|Task Description|Current Status|Due Date|Date Assigned|Resolved Date|
|---|---|---|---|---|
|Lecture 1 Notes|Complete|09/01/2021|08/25/2021|08/29/2021|
|Lab 1 Notes|Complete|09/01/2021|08/25/2021|08/29/2021|
