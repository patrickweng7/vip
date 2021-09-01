**Name:** Yisu Ma

**Email:** yma391@gatech.edu

**Threads:** Info-Networks && AI

**VIP:** Automated Algorithm Design

# **Fall 2021**

***
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

![](https://github.gatech.edu/yma391/VIP-AAD/blob/master/vip%20lab1.png)

**Action Items:**
| Task | Current Status | Date Assigned |  Suspense Date | Date Resolved |
|------|----------------|---------------|----------------|---------------|
|Import deap library and set up Jupyter Notebook|complete|August 25th, 2021|September 1st, 2021|August 28th, 2021|
|Record Notebook|complete|August 25th, 2021|September 1st, 2021|August 31th, 2021|
|Lab1|complete|August 25th, 2021|September 1st, 2021|August 31th, 2021|





