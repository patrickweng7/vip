
***
* Introduction to VIP
* Syllabus 
* Allowing others to reproduce the work done (documentation)

* Phone number : 4049368747
* Personal e-mail : rayan.dabbagh@gmail.com
* Please find the reference materials by clicking on this link: https://github.gatech.edu/emade/emade/
                                                      
# **Automated Algorithm Design**

_GitHub: github.gatech.edu/emade/emade_

## Lecture 6: Presentation day

During presentation day, each group walked through his project in front of the class, and answered questions on diverse topics covered in the powerpoint. It was a good way to develop basic presentation skills while at the same time explain complex topics orally.
I benefitted a lot from the presentation in Lecture 5 which thought us the art of presenting information to an audience. 
I was able to apply what I learned in lecture 5 during my presentation, and I can definitely say that I improved as a speaker!

I worked on the data preprocessing part in the project : The process of transforming raw data into an understandable format for any audience. 

My team's preprocessing experimentation can be found in the link : https://drive.google.com/drive/folders/1lq6fycfuDPxNamEK6inOa1vt8-RddgiS

**Data preprocessing:**

![ML vs MOGP comparision](https://picc.io/0tb7Fw_.png)

**Critique from professor/students:** We were informed to try a crossover on the hall of fame individuals. He also explained that NSGA II truncates any individual past the "k"th index in the list of selected individuals. Therefore, shuffling the individuals would have let individuals to enter the hall of fame throughout the evolutionary loop.

**Action Items:**

|Task Description|Current Status|Due Date|Date Assigned|Resolved Date|
|---|---|---|---|---|
|Notebook Update|Complete|09/29/2021|10/06/2021|09/04/2021|

Link to my presentation: https://docs.google.com/presentation/d/1tK83vBU6uQFYQGAivnSjWEM4Ghw3qJaGR5Py14BocJk/edit#slide=id.gf4beb00e17_2_0

## Lecture 5:

In class, we discussed last week's project (Titanic Dataset). We got notified of this week's task. We also assisted to a presentation which revolves around the art of giving presentations. At the end of the class, we had a discussion with our group members to schedule future meetings.

**Prof's notes:**

* Manipulating the hyper parameters is an important factor required to obtain co-dominant algorithms.
* Be aware that the above may sometimes result in a reduction in the Pareto optimal fitness scores.
* Hit and trial was used to assure co-dominant algorithms. A few iterations for finding the optimal models and to achieve a successful outcome.
* Next week's goal is to use multiple objective genetic programming to find the Pareto optimal solution.
* Our goal now is to improve on the accuracy of our algorithms.
* We have to use basic primitives to generate desired solutions.
* During the weekend, my team and I have set meetings to catch up on our individual works.

**Group results showing the Pareto Front in Machine Learning and Genetic Programming (Approved by professor):**

![ML vs MOGP](https://picc.io/Efh2aD-.png)

One recommendation from the professor was to overlap the two graphs to better show the difference between them.

**Action Items:**
| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| --- | ----------- | --- | ----------- |----------- |
| Update Notebook & Review notes | Completed | 9/22/2021 | 9/29/2021 | 9/26/2021 |
| Team meeting 1| Completed | 9/22/2021 | 9/29/2021 | 9/27/2021 |
| Team Meeting 2 | Completed | 9/22/2021 | 9/29/2021 | 9/26/2021 |
| Complete Predicting Titanic Survivors Assignment and Presentation | Currently in Progress | 9/22/2021 | 9/29/2021 | 9/28/2021 |


## Lecture 4:

**Prof's notes:**

We were divided into groups depending on our mastery of the Python programming language and our ability to write efficiently machine learning algorithms. I rated myself a 4/5 in Machine Learning and a 4.5/5 in Python.

We got split into  project groups and then introduced to Kaggle Competitions. The project was called Titanic. For now, I am using Scikit for predictors and models. I am also consolidating my skills in Panda ( special library in Python ).

Our Results are scored based on objectives: false positive and false negative. I used train.csv, test.csv, predictions.csv, and the python notebook to structure our project. Within the python notebook, I used Panda in order to train and test my dataset. I needed to clean data sets, using "isna" to find N/A values and replace them with averages/modes of their columns.

The professor talked about encoding any strings to ints and replacing them in their columns (more useful in ML). He also introduced a new definition, "feature": something that describes data. In this case, not the "survived" column. We were then tasked to allocate training data for training and testing.

**Important:**
*         x_train = top x rows of train.csv
*         y_train = survived x rows of train.csv
*         x_test = bottom (n - x) rows of train.csv
*         y_test = survived (n - x) rows of train.csv
*         Use Scikit score function to evaluate predictions
Within our groups, our algorithms must be codominant. That being said, the train and test partitions must be the same. To ensure codominance, we have to check that the random state parameter is to be the same.

At the end, we will have to submit our final predictions file with results codominant within our groups.

**Project Notes:**
 
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

**Titanic ML Problem - Data Preprocessing**
* Created Google Colab/Slack notebook for group preprocessing
* Created ParetoFront.ipynb for group to input objective values for individual learner and confirm co-dominance
* Read in train and test sets as dataframes
* Set PassengerID as index
* One hot encoded Embarked feature values so as to not incorrectly assign a magnitude of value to each Embarked class. Created three columns, 0, 1, 2, each of which is assigned either the value 0 or 1 for each sample based on the Embarked class for that sample. 
* Replaced Sex feature categories with 1 for male and 0 for female
* We replaced null Age and Fare values with median values based on Pclass of passenger (see above). 
* Selected XGBoost learner due to its speed and ability to handle null data
* Split training data into training and testing sets (test_size=0.33, random_state=10)
* Initially ran XGBoost predictions with default hyperparameters 
* Gathered the confusion matrix for predictions 
* Changed XGBoost hyperparameters

Findings:
Comparing Charlie's and Aditi's learners, I noticed a discrepancy in the values of the FNR and FPR, given the same preprocessed data as an input. Charlie's performed much better in the FPR objective and Aditi's performed much better in the FNR objective. From the above, we can deduce that neural networks, specifically MLP classifiers, favor FP prediction at the risk of accuracy while XGBoost favors even distribution of the FNR and FPR. We have to further tweak the hyperparameters to achieve a particular FNR, FPR, and accuracy.

**Action Items:**
| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| --- | ----------- | --- | ----------- |----------- |
| Review Titanic Dataset and Preprocessing/Hyperparameter Tuning Techniques | Completed | 9/15/2021 | 9/22/2021 | 9/16/2021 |
| Titanic ML Learner Predictions| Completed | 9/15/2021 | 9/22/2021 | 9/17/2021 |
| Create Subteam Slack | Completed | 9/15/2021 | 9/18/2021 | 9/15/2021 |
| Meet to Discuss Individual Learners' Performance | Completed | 9/15/2021 | 9/18/2021 | 9/18/2021 |

## Lecture 3:

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
| " " | Level of detail: personal work and accomplishments |  |  | 15 |
| Useful resource | References (internal, external) |  |  | 8 |
| " " | Useful resource for the team |  |  | 14 |
| Total |  |  |  | 97 |

### Action items
|Task Description|Current Status|Due Date|Date Assigned|Resolved Date|
|---|---|---|---|---|
|Lecture 3 Notes|Complete|09/15/2021|09/08/2021|09/12/2021|
|Self-Evaluation|Complete|09/15/2021|09/08/2021|09/12/2021|
|Lab 2(part II) Notes|Complete|09/15/2021|09/08/2021|09/14/2021|


## Lecture 2: Genetic Programming

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

_We first start with importing the libraries, and then create individual classes. For lab #2, the individual class just created inherits from the DEAP library, as our individual will be represented in a tree. These programs commonly take the form of trees representing LISP s-expressions, and a typical evolutionary run produces a great many of these trees. For this reason, a good tree-generation algorithm is very important to genetic programming.When evaluating an individual, we compile the tree from its leaves to its roots.After that, we go ahead and instantiate a primitive set and add the ones that can be used. After having defined all the components of our code,we define our evaluation function._

After running our evolutionary algorithm for 40 generations, our fittest individual results as follows:
 *  `Best individual is add(multiply(x, add(multiply(add(multiply(x, x), x), x), x)), x), (1.2992604035586602e-16,)`

### Multi-Objective Genetic Programming with Symbolic Regression

* Multi-objective optimization is is an area of multiple criteria decision making that is concerned with mathematical optimization problems involving more than one objective.

_We add three new primitives, then re-instantiate our primitive set. Then, we define our Pareto Dominance, which returns true if the first individual is fittest (on all levels) relative to the second individual. Therefore, we end up with our new fit individual._

### Action items
|Task Description|Current Status|Due Date|Date Assigned|Resolved Date|
|---|---|---|---|---|
|Lecture 2 Notes|Complete|09/08/2021|09/01/2021|09/03/2021|
|Lab 2 Notes|Complete|09/08/2021|09/01/2021|09/03/2021|  
                                                   
## **Lecture 1: Genetic Algorithms**

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

### Action Items
|Task Description|Current Status|Due Date|Date Assigned|Resolved Date|
|---|---|---|---|---|
|Lecture 1 Notes|Complete|09/01/2021|08/27/2021|08/29/2021|
|Lab 1 Notes|Complete|09/01/2021|08/27/2021|08/29/2021|


