**Name:** Yisu Ma

**Email:** yma391@gatech.edu

**Threads:** Info-Networks && AI

**VIP:** Automated Algorithm Design

**Interests:** Fingerstyle guitar, Soccer, K-pop, Camping

# **Fall 2021**
## Week 4: September 16th - September 22nd (2021)
### Lecture Overviews
* Splited into different bootcamps (Bootcamp 2)
* Introduced Kaggle Competitions
* Talked about the Titanic disaster problem
* Introduced Scikit (scikit-learn.org) for machine learning prediction model
* The results are: 1 = survived, 0 = did not survive
* Talked about the difference between test.csv and train.csv
* Introduced pandas and NumPy in the reference notes
* Requested codominant results in groups final csv files
### Groupwork notes
* Sat up discord channel and met online during weekend
* The meeting records are here :https://docs.google.com/document/d/1WVhgmRNwyJxAAaGPhp5YT6-aHzeGc_kS8ewx94U4Myw/edit
* Dropped the parameters that we thought are not important
* Recorded codominant results in our csv files
* Chosen model: SVC_sigmoid

| Task | Current Status | Date Assigned |  Suspense Date | Date Resolved |
|------|----------------|---------------|----------------|---------------|
|Titanic problem lecture notes|complete|September 15th, 2021|September 22nd, 2021|September 22nd, 2021|
|Record Notebook|complete|September 15th, 2021|September 22nd, 2021|September 22nd, 2021|
|Data processed|complete|September 15th, 2021|September 22nd, 2021|September 22nd, 2021|


***
![](https://github.gatech.edu/yma391/VIP-AAD/blob/master/aad%20rubric.jpg)
## Week 3: September 9th - September 15th (2021)
### Lecture Overviews
* Talked about the self-grading rubric
* Introduced the multipole objectives
* Discussed sample problems

### Lecture Notes
* Gene pool is the set of genomes to be evaluated during the current generation
* GA = set of values
* GP = tree structure, string
* The Evaluation of a Genome associates a genome/individual (set of parameters for GA or string for GP) with a set of scores
* True Positive – TP 👍 ; False Negative - FN (type II error) 👎 
* False Positive – FP (type I error) 👍 ; True Negative - TN 👎 
* Review the fruit sample
* Sensitivity or True Positive Rate (TPR)  ||  TPR = TP/P 
* Specificity (SPC) or True Negative Rate (TNR)  || TNR = TN/N = TN/(TN+FP)
* False Negative Rate (FNR)
FNR = FN/P = FN/(TP+FN)
FNR = 1 - TPR
* Fallout or False Positive Rate (FPR)
FPR = FP/N = TN/(FP+TN)
FPR = 1 – TNR = 1 - SPC
### lab2 - partII
* added three new primitives and reinitialized toolbox
* defined pareto dominance function
* initialized a random population of 300 individuals, I changed the population later in for testing
* sorted our population
![](https://github.gatech.edu/yma391/VIP-AAD/blob/master/aad%20week3-1.PNG)
* defined and ran the main evolutionary algorithm
![](https://github.gatech.edu/yma391/VIP-AAD/blob/master/aad%20week3-2.PNG)
* _Learning point:_ we can see that at the end, DEAP's Mu plus Lambda evolutionary algorithm lead to the model to expected result
`49 	70    	[ 0.28332572 14.2       ]	[0.01730375 1.73205081]        	[ 0.27861333 11.        ]	[ 0.40364443 17.        ]
50 	66    	[ 0.28037786 15.3       ]	[1.21684778e-03 1.31529464e+00]	[ 0.27861333 10.        ]	[ 0.28500933 19.        ]`
![](https://github.gatech.edu/yma391/VIP-AAD/blob/master/aad%20week3-3.PNG)
**Action Items:**

| Task | Current Status | Date Assigned |  Suspense Date | Date Resolved |
|------|----------------|---------------|----------------|---------------|
|Multiple Objectives lecture notes|complete|September 9th, 2021|September 15th, 2021|September 14nd, 2021|
|Record Notebook|complete|September 9th, 2021|September 15th, 2021|September 14th, 2021|
|Lab2 part2|complete(questions remaining)|September 9th, 2021|September 15th, 2021|September 14th, 2021|


## Week 2: September 1st - September 8th (2021)
### Lecture Overviews
* Summarized the knowledge in the last class (Genetic Algorithm)
* Introduced Genetic programming
* Solved several examples
### Lecture Notes
* Instead of taking an individual and having a function evaluator to obtain objective scores…
* **Tree Representation:**
1. represent a program as a tree structure
2. Nodes are called primitives and represent functions
3. Leaves are called terminals and represent parameters
![](https://github.gatech.edu/yma391/VIP-AAD/blob/master/aad%20week2-1.PNG)
The tree for f(X) = 3*4 + 1 can be written as:
[+, *, 3, 4, 1]
* Crossover in tree-based GP is simply exchanging subtrees
* Start by randomly picking a point in each tree
* These points and everything below creates subtrees
**Mutation:**
1. Inserting a node or subtree
2. Deleting a node or subtree
3. Changing a node
* We discussed using the Taylor Series formula for sin(x) mutation

### Lab Overviews
**Symbolic Regression**
* Focusing on genetic programming
* created fitness and individual classes
* Initialized PrimitiveTree class && added primitives (below are the added primitives)

`pset.addPrimitive(np.sin, arity=2)
pset.addPrimitive(np.cos, arity=2)`

* Defined our toolbox, individual, population, and compiler.
* Defined our evaluation function
* Registered genetic operators
* Added tree height constraints
* Final evolutionary result(with a main evolutionary algorithm)

`-- Generation 37 --
  Min 0.0
  Max 4.0
  Avg 0.19622640892733842
  Std 0.6451516675079916
-- Generation 38 --
  Min 0.0
  Max 5.0
  Avg 0.25942077567399063
  Std 0.777762277528485
-- Generation 39 --
  Min 0.0
  Max 5.0
  Avg 0.14866815351562296
  Std 0.6378905904759974
-- End of (successful) evolution --
Best individual is multiply(x, x), (0.0,)
plt.plot(gen, avg_list, label="average")`

![](https://github.gatech.edu/yma391/VIP-AAD/blob/master/aad%20week2-lab.png)

**Reflection and Thoughts:**the original result is 
`negative(cos(multiply(add(cos(sin(cos(sin(cos(tan(x)))))), cos(x)), tan(x))))
with fitness: (0.2786133308027132, 15.0)`
I changed the primitives I use in the algorithm, but I didn't successfully lower our AUC, I may need some help after this. 
 
**Action Items:**
| Task | Current Status | Date Assigned |  Suspense Date | Date Resolved |
|------|----------------|---------------|----------------|---------------|
|GP lecture notes review|complete|September 1sth, 2021|September 8th, 2021|September 2nd, 2021|
|Record Notebook|complete|September 1sth, 2021|September 8th, 2021|September 7th, 2021|
|Lab2|complete(questions remaining)|September 1sth, 2021|September 8th, 2021|September 8th, 2021|


## Week 1: August 25th - September 1st (2021)
### Lecture Overviews
* Summarizing the logistics of the class: wiki page, general ideas, syllabus, notebooks.
* Started lecture on genetic programming.
* Jupyter notebook and lab 1 introduction
### Lecture Notes
* Genetic Algorithms: various evolutionary algorithms to create a solution or best individual.
* Key Words:
  1. Individual: one specific candidate in the population (with properties such as DNA)
  2. Population: a group of individuals whose properties will be altered 
  3. Objective: a value used to characterize individuals that you are trying to maximize or minimize (usually the goal is to increase objective through 
     the evolutionary algorithm)
  4. Fitness: relative comparison to other individuals; how well does the individual accomplish a task relative to the rest of the population? 
  5. Evaluation: a function that computes the objective of an individual
  6. Mate/Crossover: represents mating between individuals
  7. Mutate: introduces random modifications; the purpose is to maintain diversity
* One Max Problem
### Lab 1
**One Max Problem:** try to find a bit string containing all 1s.
* import deap
* define the fitness objective and individual classes
* define Toolbox
* _Learning point:_ Our single objective is a tuple -- (1.0,) for maximum; (-1.0,) for the minimum;
  For multi-objective: we can do something like (1.0, 1.0)
* define our genetic algorithm
* _Learning point:_ We can use the evaluate/mate/mutate/select function for our genetic algorithm. The current probability of bit flipping is defined as 5% in our example. Is there any standard for this number? So does the amount of tournament selection. If we increase the number of selections, would it affect how many generations we finally have?
* design our main algorithm
* run the main function

`-- Generation 39 --
  Min 87.0
  Max 99.0
  Avg 97.8
  Std 2.4522098876997482
-- End of (successful) evolution --
Best individual is [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], (99.0,)`


**Reflection and Thoughts:** Most of the time, we can reach maximum fitness within 40 generations. I changed the selection size to >=3 and the results run well, but if I decrease the selection size, the result will not end up with maximum fitness.

`-- Generation 39 --
  Min 40.0
  Max 67.0
  Avg 50.846666666666664
  Std 4.3935356554324
-- End of (successful) evolution --
Best individual is [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1], (67.0,)`

**The N Queens Problem:** determine a configuration of n queens on an nxn chessboard such that no queen can be taken by one another.
* creat fitness and individual classes.
* _Learning point:_ Since we want to minimize the number of conflicts between two queens, we wanna use the minimum objective for this model.
* define toolbox
* define a permutation function
* define evaluation function
* define crossover&&mutation function
* _My new mutation function:_ 


`
def newMutShuffleIndexes(individual, indpb):`

    size = len(individual)

    for i in range(size):
        if random.random() < indpb:
            swap_indx = random.randint(0, size - 2)
            if swap_indx >= i:
                swap_indx += 1

            q1 = (i + swap_indx) / 2
            q3 = swap_indx + (size - swap_indx) / 2
            individual[i], individual[int(q1)], individual[swap_indx], individual[int(q3)] = \
                individual[int(q3)], individual[swap_indx], individual[int(q1)], individual[i]
    
    return individual,`


* run main evolutionary function for 100 generations.
* _Result:_

`-- Generation 98 --
  Min 0.0
  Max 13.0
  Avg 1.372
  Std 2.9342147160697016
-- Generation 99 --
  Min 0.0
  Max 12.0
  Avg 1.291
  Std 2.8785967067305553
-- End of (successful) evolution --
Best individual is [18, 3, 8, 13, 11, 6, 1, 16, 5, 12, 10, 15, 0, 4, 19, 7, 14, 2, 17, 9], (0.0,)`

**Reflection and Thoughts:**
* After about 30 generations, the plot of the minimum would have a severe decrease. In the end, the average plot would not be exactly at 0, but very close. Here is the visualization of the graph.
* How could the mutation function affect our final result to reach a minimum?

![](https://github.gatech.edu/yma391/VIP-AAD/blob/master/vip%20lab1.png)

**Action Items:**
| Task | Current Status | Date Assigned |  Suspense Date | Date Resolved |
|------|----------------|---------------|----------------|---------------|
|Import deap library and set up Jupyter Notebook|complete|August 25th, 2021|September 1st, 2021|August 28th, 2021|
|Record Notebook|complete|August 25th, 2021|September 1st, 2021|August 31th, 2021|
|Lab1|complete|August 25th, 2021|September 1st, 2021|August 31th, 2021|





