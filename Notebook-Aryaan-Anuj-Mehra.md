== Team Member ==
<b>Team Member:</b> Aryaan Mehra <br>
<b> Major: </b> Computer Science <br>
<b>Email:  </b>aryaan@gatech.edu <br>
<b>Cell Phone:</b> (678)-907-4617 <br>
<b>Interests:</b> ML/DL, computer vision, finance <br>
=Fall 2020=

== December 2nd, 2020 ==
* Today we presented our final presentation to all the other subteams.
* Stocks:
** We spoke about issues we had with the CEFLANN paper and the inconsistencies that existed.
** We spoke about our approach with using genetic programming and obtained better results.
** We also spoke about the different technical indicator primitives that have been implemented in EMADE.
** Discussed how our EMADE run performed in comparison to the CEFLANN model.
* NN NLP:
** Spoke about the data sets they worked on. Used accuracy and AUROC to measure results.
** They also used tree based networks versus graph based learners.
** Created NNLearner, a new kind of primitive which fits a Keras model.
** They also created a BERT based primitive and places it at the beginning of networks.
** Spoke about the issues with PACE and using NVIDIA CUDA.
* EzCGP:
** Began by correcting the OpenCV primitives.
** Decided to focus on precision and recall.
** Since pace was taking more time they decided to seed the different individuals.
** They ran for 6 generation in the end but still has little diversity in the population.
** They believe using transfer learning less may help in the results and so would changing some of the primitives.
* Modularity:
** Finding combination of nodes based on their frequency and fitness.
** ARL's can wrap around each other and grow.
** Used differential fitness as a metric for optimization.
** Best p-values are found between generations 15 and 20.
* Link to Presentation - https://docs.google.com/presentation/d/1arplCjluOGjVm58LiMHV2zVwXl0GCvCsvgb2Ou7nSN8/edit?usp=sharing

=== '''Action Items''' ===
{| class="wikitable"
!'''Task'''
!'''Current Status'''
!'''Date Assigned'''
!'''Suspense Date'''
!'''Date Resolved'''
|-
|Last Presentation
|Completed
|December 2nd, 2020
|December 2nd, 2020
|December 2nd, 2020
|}
==November 29th, 2020==
* Over the last weekend, I met with the other first semester's in the Stocks team and we researched and implemented 3 new TI's so that we can run EMADE with more primitives.
** CCI - Commodity Channel Index​ (CCI) is a momentum-based oscillator used to help determine when an investment vehicle is reaching a condition of being overbought or oversold. It is also used to assess price trend direction and strength. This information allows traders to determine if they want to enter or exit a trade, refrain from taking a trade, or add to an existing position. In this way, the indicator can be used to provide trade signals when it acts in a certain way.
** Bollinger Bands - Bollinger Bands are envelopes plotted at a standard deviation level above and below a simple moving average of the price. Because the distance of the bands is based on standard deviation, they adjust to volatility swings in the underlying price. Bollinger bands help determine whether prices are high or low on a relative basis. They are used in pairs, both upper and lower bands and in conjunction with a moving average.
** MFI - The Money Flow Index (MFI) is a momentum indicator that measures the flow of money into and out of a security over a specified period of time. It is related to the Relative Strength Index (RSI) but incorporates volume, whereas the RSI only considers price.
* Link to Google Colab notebook - https://colab.research.google.com/drive/1wfLGQprIf4ILQBK4Ot-7lCl3zTO11m2o?usp=sharing
* Link to research website - https://www.centralcharts.com/en/gm/1-learn/7-technical-analysis/26-technical-indicator

=== '''Action Items''' ===
{| class="wikitable"
!'''Task'''
!'''Current Status'''
!'''Date Assigned'''
!'''Suspense Date'''
!'''Date Resolved'''
|-
|Implement Researched TI's
|Completed
|November 29th, 2020
|December 2nd, 2020
|December 1st, 2020
|-
|Create a slide to present during final presentation
|Completed
|November 29th, 2020
|December 2nd, 2020
|December 1st, 2020
|}
==November 9th, 2020==
*I've decided to join the stocks subteam for the remainder of this semester.  
**Jason joined us this meeting at spoke to us about some of the difficulties we were having implementing the model present in the CEFLANN paper. 
**We decided to do more error analysis and try to emulate their paper exactly. 
**As a new recruit I was tasked with researching about more technical indicators for now and try to work on converting them to EMADE primitives.

=== '''Action Items''' ===
{| class="wikitable"
!'''Task'''
!'''Current Status'''
!'''Date Assigned'''
!'''Suspense Date'''
!'''Date Resolved'''
|-
|Research different Technical Indicators 
|Completed
|November 9th, 2020
|November 29th, 2020
|December 29th, 2020
|-
|}
==October 19th, 2020==
*We presented our EMADE findings together as a team in front of the other bootcamp teams as well as other sub-teams. 
**We focused on correcting mistakes from past presentations.
**We ran 2 runs of EMADE and altered crossover rate as well as headless chicken.
**We analysed the different tree structures being formed and the reason why that might be the case as well.
**Comprehensively compared ML to MOGP to EMADE.
*Heard from Stocks, Modularity, NN, and ezCGP as well as the bootcamp teams. Decided to join stocks since I have a background and finance and see a lot of scope in understanding time series data better. '''Presentations from bootcamp groups and existing subteams:'''
** Stocks
*** Implementation based on paper that has CEFLANN architecture
*** Uses the following technical indicators price, volume, open interest
**** Inspired by actual traders
*** Intend to design novel technical indicators
** Modularity
*** Attempts to modularize common patterns (encodes common node and leaf combinations into a single function)
*** These new constructs are called ARLs (adapted representation through learning)
*** Selection algorithm biases towards individuals with more ARLs
** NLP/NN
*** Applies EMADE to neuroevolution and abstracts towards automation
*** Uses tree based representation to build and search neural architecture (NEAT uses graph)
*** Run using a dataset from literature using limited functionality performs almost at state of the art levels
** ezCGP
*** Represents GP using cartesian space
*** This compact representation allows for arbitrary connections between processing layers (block based)
*** Amenable to gaussian and other convolutions

=== '''Action Items''' ===
{| class="wikitable"
!'''Task'''
!'''Current Status'''
!'''Date Assigned'''
!'''Suspense Date'''
!'''Date Resolved'''
|-
|Research sub-team options and send preferences.
|Completed
|October 19th, 2020
|October 26th, 2020
|October 24th, 2020
|-
|}
==October 7th, 2020==
Objective this week is to get EMADE up and running and try a few different tests i.e with the number of generations, probabilities, mutation, etc. 

Try to obtain results and create the presentation by Sunday.
===Subteam #2 Meeting: October 13th, 2020===
* I met with Jason before the meeting to try and figure out why people were not able to get their MySQL connected to my server. We realised that it was mainly an IP address problem so we moved to the campus VPN and all the connections now work properly. 
* Unfortunately, when we run titanic using EMADE we are encountering an issue as the program seems to stop automatically. I think this is due to an error in running the program. I am unsure why though will ask Jason. 

===Subteam #2 Meeting: October 12th, 2020===
* Met with everyone on the team after we got EMADE up and running on our local PC's. 
* Tried to create a server and connect everyone but was unable to establish a connection.
* Set up a meeting with Jason to resolve these issues.

=== '''Action Items''' ===
{| class="wikitable"
!'''Task'''
!'''Current Status'''
!'''Date Assigned'''
!'''Suspense Date'''
!'''Date Resolved'''
|-
|Update notebook
|Completed
|October 7th, 2020
|October 13th, 2020
|October 12th, 2020
|-
|Connect eveyone on MySQL Server
|Completed
|October 7th, 2020
|October 13th, 2020
|October 11th, 2020
|-
|Run Titanic with base the implementation
|Complete
|October 7th, 2020
|October 12th, 2020
|October 11th, 2020
|-
|Try to verify results
|In Progress
|October 7th, 2020
|October 13th, 2020
|October 12th, 2020
|-
|}

==Presentation: September 23rd, 2020==
===Subteam #2 Meeting: September 20th, 2020===
*Did basic data preprocessing for the GP model so that the team could copy the .ipynb and experiment using different primitives, parameters, evaluation function
*Discussed some of the best parameter choices for the GP model
*Created Google Slides and split up work so that everyone can contribute equally to the work needed before the presentation

=== '''Some of our pre-processing and results:''' ===
[[File:Screen Shot 2020-09-28 at 5.58.53 AM.png|thumb|639x639px|Data Preprocessing|none]]

[[File:Screen Shot 2020-09-28 at 5.59.15 AM.png|thumb|374x374px|ML Results|none]]

[[File:Screen Shot 2020-09-28 at 5.59.32 AM.png|thumb|375x375px|GP Results|none]]

===Action Items===
{| class="wikitable"
!Task
!Current Status
!Date Assigned
!Suspense Date
!Date Resolved
|-
|Clone EMADE Repository
|Completed
|September 23, 2020
|September 30, 2020
|September 28, 2020
|}

==September 16th, 2020==
===Lecture Notes===
*Next assignment will be applying MOGP models to the titanic disaster survival dataset
**Individuals will be models that can predict survival given a data set of passengers
**Fitness will be measured in multiple objectives, false negative rate and false positive rate
===Action Items===
{| class="wikitable"
!Task
!Current Status
!Date Assigned
!Suspense Date
!Date Resolved
|-
|Complete Titanic Project with Genetic Programming
|Completed
|September 16, 2020
|September 23, 2020
|September 22, 2020
|}

== Self-graded Rubric ==
Total: 91

https://docs.google.com/document/d/1kwA4-OyRa6mlsr6WjcClNA8XigfiLK9E/edit

==September 9th, 2020==
===Bootcamp Lecture 4:===
*Split into sub-teams using our Pareto Optimality
*Discussed the titanic dataset Kaggle competition
*Went through a notebook showing example feature engineering and modeling with scikit-learn
*Team members: Xufei, Jonathan, Hannah, Bernadette
===Subteam #2 Meeting: September 14th, 2020===
*Decided to circle back on Monday in order to discuss findings and determine which models were working best
*Noticed that a significant difference was not visible by using the features I had engineered the prior week 
*Decided to remove features instead of formatting existing ones into usable features to see if they model was more predictive
*This did give us better results on the test set but did not generalize to the common case well, hence, we suffered from overfitting
*In the end, the optimized random forest classifier fit on the features engineered before the first meeting provided the best results (83.7%)
===Subteam #2 Meeting: September 12th, 2020===
*Created a GroupMe in order to stay in contact
*Created a Github in order to store all code and results effectively: https://github.gatech.edu/amehra37/Titanic_ML_Group2
*Discussed the problem at length and the basic steps to get started, namely feature engineering and model selection/tuning
*I had done some feature engineering beforehand in order to explain the process and get everyone up to speed in terms of finding and formatting features in the best possible to maximize results
*Decided to run some preliminary models on the feature set I made and do some individual exploration to see if we can get more predictive train-tests splits

===Performance metrics:===
*During feature engineering, I kept most, if not all of the numeric features in the dataset. For the categorical data, I removed the unnecessary terminology. For example, I converted the tickets into numerical values and the port of embarkment into numerical values
*I also converted the cabin section into single letter values. For example, if the cabin had a section assigned to it I kept only the first letter of the section and categorized people into those letters. I noticed that the people who did have cabin sections with letters, generally had a higher survival chance
*I learned to use the python 'dummy' function to convert these categorical types into numerics which the model could comprehend
*I also sectioned people by their titles and saw whether a particular title translated into a higher survival rate. I then used dummies to numerically represent them.
*The team trained the features on the following models:
**Tree, Neural Net, SVM, Random Forest, KNN, XGBoost
*I personally found the best results using the random forest classifier and the SVM
*I learned to use the GridSearchCV algorithm to test on multiple combinations of parameters to tune the model.
*Best results: 
**Random Forest classifier: 83.7% on our test split and 78% on Kaggle. FP: 28, FN: 18
**SVM: 83.7% on our test split and 76% on Kaggle. FP: 24, FN: 24 

**Confusion (Matrix)
***Inline result 83.7%, Kaggle result 78%[[File:Screen Shot 2020-09-18 at 5.30.26 PM.png|none|thumb]]
**Plot of member's model results
***The presence of one non-codominant result has resulted in the formation of two pareto fronts. Three of the models formed co-dominant results.[[File:Screen Shot 2020-09-18 at 5.30.05 PM.png|none|thumb]]
===Action Items===
{| class="wikitable"
!'''Task'''
!'''Current Status'''
!'''Date Assigned'''
!'''Due Date'''
!'''Date Resolved'''
|-
|Complete Notebook
|Completed
|September 9th, 2020
|September 16th, 2020
|September 15th, 2020
|-
|Complete Notebook Self Eval Rubric
|Completed
|September 9th, 2020
|September 16th, 2020
|September 18th, 2020
|-
|Conduct team meetings
|Completed
|September 9th, 2020
|September 16th, 2020
|September 14th, 2020
|-
|Create models and compare results with team
|Completed
|September 9th, 2020
|September 16th, 2020
|September 14th, 2020
|-
|}

== Lab 2 Part II Results ==

* Followed the lab to solve the multiple objective problem with symbolic regression
* Plotted Pareto Frontier to visualize how individual solutions can dominate each other
* <code>Area Under Curve: 6.311825374537857</code>
* Strongest individual: <code>subtract(negative(multiply(cos(add(x, multiply(tan(x), cos(cos(sin(multiply(subtract(tan(subtract(multiply(tan(x), multiply(tan(cos(x)), cos(multiply(x, x)))), x)), cos(add(x, cos(negative(subtract(multiply(x, x), tan(x))))))), cos(x)))))))), tan(sin(add(subtract(x, x), tan(sin(add(subtract(x, x), tan(x))))))))), cos(add(x, multiply(tan(x), cos(x)))) with fitness: (0.033164149955997646, 63.0)</code>
[[File:Screen Shot 2020-09-15 at 9.52.53 PM.png|frameless]]
[[File:Screen Shot 2020-09-15 at 9.52.46 PM.png|frameless]]
* Obtained by removing trigonometric identities
* <code>Area Under Curve: 0.6912744703006891</code>
* Strongest individual: <code>subtract(x, x)</code>
[[File:my-pareto-front.png|frameless|Pareto Frontier]]
[[File:evolution-graph-lab2-2.png|frameless|Evaluating fitness using SR]]

== September 2nd, 2020 ==
===Bootcamp Lecture 3 Notes:===
'''Multiple Objectives –  The MO in MOGA and MOGP'''

* We have multiple requirements when dealing with algorithms in this project.
* The lecture focussed on the translation of the vector of scores from evaluation into a fitness value.
* Given a gene pool, how can we use the results of our evaluation to arrive at fitness values with which we can select a new gene pool

* '''Gene pool''' is the set of genomes to be evaluated during the current generation
** Genome
** '''Genotypic''' description of an individual
** DNA
** '''GA''' = set of values
** '''GP''' = tree structure, string

* '''Search Space'''
** Set of all possible genomes
** For Automated Algorithm Design:
*** '''Set of all possible algorithms'''

* '''Evaluation''' of a genome associates a genome/individual (set of parameters for GA or string for GP) with a set of scores
* What are these scores?
** '''True Positive – TP'''
*** How often are we identifying the desired object?
** '''False Positive – FP'''
*** How often are we identifying something else as the desired object?

* '''Objective Space''' – Set of objectives
** '''Evaluation''' – Maps an genome/individual
*** From a location in search space:
**** '''Genotypic''' description
*** To a location in objective space:
**** '''Phenotype''' description

'''Important Evaluation Measures''' 
* We can view the evaluation results i.e. TP or TN by looking at the diagonal of a '''confusion matrix'''
* '''Sensitivity of True Positive Rate (TPR)'''
** AKA Hit Rate or Recall
** TPR = TP/P = TP/(TP+FN)
* '''Specificity (SPC) or True Negative Rate (TNR)'''
** TNR = TN/N = TN/(TN+FP)
* The algorithms with the '''highest''' '''TNR/TPR''' will have the '''highest''' chance of mating.

* '''False Negative Rate (FNR)'''
** FNR = FN/P = FN/(TP+FN)
** FNR = 1 - TPR
* '''Fallout or False Positive Rate (FPR)'''
** FPR = FP/N = TN/(FP+TN)
** FPR = 1 – TNR = 1 - SPC
* The algorithms with the '''lowest''' '''FNR/FPR''' will have the '''lowest''' chance of mating.

* '''Precision or Positive Predictive Value (PPV)'''
** PPV = TP / (TP + FP)
** Bigger is better
* '''False Discovery Rate'''
** FDR = FP/(TP + FP)
** FDR = 1 - PPV
** Smaller is better

* '''Accuracy (ACC)'''
** ACC = (TP+TN) / (P+N)
** ACC = (TP+TN) / ( TP + FP + FN + TN)
** Bigger is better

'''Optimization Measures'''
* We can use several measures to determine the error of a function which we need to minimize:
** Mean Squared Error
** Cost functions
** Complexity
* We can also use Pareto optimality:
** An individual is '''Pareto''' if there is no other individual in the population that outperforms the individual on all objectives
** The set of all Pareto individuals is known as the '''Pareto frontier'''
** Examples of such algorithms are:
*** Nondominated Sorting Genetic Algorithm II '''(NSGA II)'''
*** Strength Pareto Evolutionary Algorithm 2 '''(SPEA2)'''

===Action Items:===
{| class="wikitable"
!Task
!Current Status
!Date Assigned
!Due Date
!Date Resolved
|-
|Complete Notebook
|Completed
|September 2, 2020
|September 9, 2020
|September 4, 2020
|-
|Lab 2 Part II - Multiple Objectives
|Completed
|September 2, 2020
|September 9, 2020
|September 9, 2020
|}
__FORCETOC__

== Lab 2 Part I Results ==

Results after running the symbolic regression genetic program.
[[File:Screen Shot 2020-09-04 at 6.01.15 PM.png|thumb|Lab 2 Part I Results|none]]
In the plot above, we can see the program finds the optimal function with maximum fitness at generation 23-25.

The best individual was:  add(add(multiply(multiply(x, x), add(multiply(x, x), x)), multiply(x, x)), x), (1.0117659235419543e-16,)

== August 26th, 2020 ==
===Bootcamp Lecture 2 Notes:===
* In a '''genetic program''' the individual itself is considered to be the function. We use a parse tree in order to represent functions in a program.
** '''Key Terms'''
*** '''Nodes''' - it represents primitives or functions such as '''+, -, *, /'''
*** '''Leaves''' - they represent the terminals/end points of the parse tree
** In order to instantiate a parse tree in python we represent it in the '''lisp preordered parse tree''' format
** For example the operation '''3 * 4 + 1''' would be represented as '''[+, *, 3, 4, 1]''' 
** In the crossover operation in a genetic program, a portion of the tree is exchanged with that of another tree in the population
** For mutation, operations involve:
*** Inserting a node or a subtree
*** Deleting a node or a subtree
*** Changing a node
** We also discussed '''symbolic regression''' conceptually in order to approximate the function '''sin(x)'''
*** '''Function:''' y = sin(x)
*** '''Nodes(primitives):''' +, -, *, / 
*** '''Terminals:''' The variable 'x'
** In order to begin we created a parse tree modeled after the 3rd order Taylor series approximation of sin(x)
** The lisp preordered format is '''[- , x, / ,*, x, *, x, x, *, 3, 2]'''
** Finally to evaluate and optimize the genetic program we sampled a random set of values between 0 and 2pi
** After generating a function using the Taylor approximation we use a '''loss function''' in order to gauge '''error'''
** We then use crossover and mutation in order to get the best members of the population
** In order to make this process more efficient we can update our primitives to include function such as power(), factorial() or even sin(), cos(), tan() for more complex operations.

===Action Items:===
{| class="wikitable"
!Task
!Current Status
!Date Assigned
!Due Date
!Date Resolved
|-
|Complete Notebook
|Completed
|August 26, 2020
|September 2, 2020
|August 27, 2020
|-
|Lab 2 Part I - Symbolic Regression
|Completed
|August 26, 2020
|September 2, 2020
|September 1, 2020
|}
__FORCETOC__

== Lab 1 Results ==

Results after running the N-Queens genetic algorithm.
[[File:Screen Shot 2020-09-04 at 5.50.22 PM.png|thumb|Lab 1 Results|none]]

In the plot above, we can see the average and minimum reach a stable bend at around 20-30 generations.

Best individual was: [10, 1, 7, 9, 11, 3, 15, 4, 19, 17, 13, 6, 16, 0, 5, 14, 8, 18, 2, 12], (0.0,)

== August 19th, 2020 ==
===Bootcamp Lecture 1 Notes:===
* '''Genetic Algorithms:''' involves numerous matings/mutations of individuals in the previous population to create a new generation that will produce the best individual (one with the highest possible fitness).
** '''Key Terms'''
*** '''Individual''' - a candidate within a population with properties
*** '''Population''' - a group of individuals whose properties will be changed
*** '''Objective''' - a value to characterize individuals and whose value is increased through an evolutionary algorithm
*** '''Fitness''' - performance of completing a task relative to the rest of the population 
*** '''Evaluation''' - function to compute the objective of an individual
*** '''Selection''' - 'survival of fittest' by giving preference to better individuals
**** Fitness Proportionate - higher the value, the better for crossover
**** Tournament - winners from several rounds of tournaments are chosen for crossover
*** '''Mate/Crossover''': includes Single Point & Double Point
*** '''Mutate'''
**** random configurations made in children to maintain diversity
*** '''Genetic Algorithm Process'''
**** First a crossover is completed and then a mutation is made
*** '''Algorithms'''
**** various evolutionary algorithms are used to create the best possible individual
** We were introduced to the '''one max problem''' which will be clarified in the lab

===Action Items:===
{| class="wikitable"
!Task
!Current Status
!Date Assigned
!Due Date
!Date Resolved
|-
|Join Slack
|Completed
|August 19, 2020
|August 26, 2020
|August 20, 2020
|-
|Set Up Jupyter Notebooks
|Completed
|August 19, 2020
|August 26, 2020
|August 19, 2020
|-
|Create Wiki
|Completed
|August 19, 2020
|August 26, 2020
|August 20, 2020
|-
|Lab - DEAP
|Completed
|August 19, 2020
|August 26, 2020
|August 24, 2020
|}
__FORCETOC__
