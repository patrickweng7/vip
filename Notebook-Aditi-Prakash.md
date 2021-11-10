# Aditi Prakash
Name: Aditi Prakash  

Email: aprakash86@gatech.edu

Bootcamp Subteam Members: Charlie Bauer - cbauer32@gatech.edu, Rayan Dabbagh - rdabbagh3@gatech.edu, Rayan Khoury - rkoury7@gatech.edu

Cell Phone: 704-794-3924  

Interests: Machine Learning, Data Science, Software Development, Dance, Reading
[[files/aprakash86/'Image 1.png']]


# Week 10th: October 25th, 2021
## Overview
Final VIP Presentations - Bootcamp Teams and Main Teams. 

## Team Meeting Notes
* Presented final VIP presentations on Titanic problem solved with ML, MOGP, and EMADE. Received feedback from Dr. Zutty and Dr. Rohling that our revision of our MOGP approach based on Dr. Zutty's suggestion to use selDCD instead of selNSGA so binary selection is performed without truncation during selection was a good choice to maximize diversity of individuals being selected from during each generation, and thereby maximize diversity of the final pareto front for each generation (spreads individuals across tradeoff space between FNR and FPR). Received feedback from Dr. Zutty to ensure that all of our pareto fronts are placed on the same graph going forward for easy comparison. Heard from following teams on their current areas of focus and considered which teams best align with my research interests:

NLP:
* Trying to recreate BiDAF and BERT primitives in EMADE based on literature review 
* Uses work of NN team

Bootcamp Subteam 1:
* Jessi, Pranav, Eashan, Elan
* SVM, Gradient Decision Boost, NN, Random Forest, Gaussian
* Pareto optimal results - FPR increasing, FNR decreasing
* Add, subtract, multiply, cos, gp.mutNodeReplacement (individuals, pset), Evaluation - FNR and FPR, Selection - custom selection tournament
* Crossover - gp.cxOnePointLeafBased (ind1, ind2, termpb - random) 
* Custom algorithm - whichever individual has a lower sum of scores is the winner
* More diversity, lower AUC
* Headless Chicken rate - crossover with an individual with a randomly generated tree, decreased mutations except for ephemeral with need more randomness, reduced mutations
* Need to split train into train and test, got one pareto optimla individual because they didn’t do this
* 0.133 vs. 0.137 - preprocessed vs not preprocessing (42 vs. 115)
* Adaboost, too many inf, emade kills CPUs, check twice that reuse = 1
* MOGP was only able to beat ML due to the fact that it fills out the Pareto front 
* gp.cxOnePointLeadBiased - always random 

Neural Architecture Search:
* Create Neural Networks automatically using primitives
* Text processing for sentiment classification
* EMADE cannot evolve seeds very much 
* Most individuals are not NN learners in EMADE, have to restrict EMADE to only work with NN learners
* Want EMADE to move past seeded individuals and explore search space well
* Minimize accuracy error, minimize number of parameters
* Lowering training time would have same effect as limiting number of parameters
* Time stopping of 600
* Time stopping of 2500
* 600 time stop generated many more valid individuals, so we can create new individuals with genetic programming even though 2500 had a smaller AUC
* Modify original DataPair once, feed data into individuals 
* Preprocessing helped average evaluation time for individuals.
* CoDEEPNeat - additional class for EMADEDataPairs, limited main primitives set type access to primitives is only ADFS that has access to it, main primitive is blueprint, adfs are modules, represents CoDEEP NEAT structure
* Limited primitives in modules to just layers 
* Separate table to track NNLearner individuals over time, where to improve in encouraging more complexity
* Detect and reward novelty, novelty - dropout, embedding, convolutional layer
* Might have a highly optimized NN with only dense layers, helps with certain scenarios like images
* Trello board 
* Split layer list primitive into different classes based on Tensor dimension they take in 
* Novelty can make an objective become a subjective 
* 2500 was more distributed than the 600 one (600 is way too small) 

Bootcamp Subteam 2:
* Drop Name, PassengerID, TicketNumber, Fare
* Map Sex and Embarked to numerical
* Fill nulls with medians, mode
* AUC of 0.18129 with ML
* SPEA2 and NSGA tried, SPEA2 worked better
* Simple primitives 
* Evaluation - activation function 
* AUC 0.125
* Didn’t one hot encode embarked feature, would have improved 
* Conda + Python 3.7
* Struggled to get FNR and FPR methods working
* MySQL very slow - only 10 generations in 3 days, hard to connect 
* Ended with pareto optimal set of 65
* Trees grow slower than MOGP
* EMADE gave better generations even with less generations 
* Should have checked if certain number of generations takes a certain amount of time 

Image Processing:
* Multilabel image processing 
* CheXNet Paper - pneumonia classification on xray scans
* Image resizing, normalization, horizontal flipping 
* 30 Generations (Precision-Recall AUC), Number of Parameters
* NSGA-II, Lexicase, Tournament, Fuzzy 
* NSGA-III defines reference points to maintain diversity in solutions
* In theory, should outperform NSGA-II
* Only ran for one generation before stopping 
* EMADE master process would kill itself
* No errors in log
* Semantic crossover and semantic mutation 
* Individual * logistic *(random1 - random2)
* Primitives not set up to handle image data, majority of generated individuals not able to generate a valid fitness score
* Geometric crossover operators 
* Simulated Binary
* Blended Crossover
* Very little information about mutation and crossover for image processing problems 
* Gray level transform that increases contrast with image filter
* Hyperfeatures - two or more features which improve fitness 
* Enhancing contrasts helps with edge detection 
* Brainstorming-image-processing channel in Slack
* Lexicase picks a random objective 
* Loosely typed GP problem - everything is coming out as a float 
* Geometric crossover makes sense when you have a numeric genome 
* Simulated binary and blended crossover work more rigidly than single point crossover 
* NSGA II - works less well with more than 2 objectives 

Bootcamp Subteam 3:
* Started with a variety of learners, changed svm to gaussian because they couldn’t create Pareto optimal front 
* Added logic primitives and some arithmetic primitives 
* Added 3rd objective of tree size to evaluation function 
* Added FNR and FPR to input file 
* Allowed port forwarding on master’s home router if workers will be joining remotely 
* Weren’t able to run as many generations with EMADE as they would have liked, ML algorithm’s AUC was better 
* EMADE and MOGP - more diverse Pareto front 
* Grep -rl error string helped trace root cause of error 

Stocks:
* Primitives Analysis:
* Relative performance of primitives based on CDF metric (lower is better)
* Boosting regression across all technical indicator generally has a good performance 
* MyDeltaWilliamsR, modify seeded individuals to create better results in EMADE 
* Profit Percentage, Average Performance Per Transaction, CDF of Profit, Variance of Profit Per Transaction 
* Want to minimize CDF
* MyBollingerBand
* Create pool of objectives and pick out stocks to conduct EMADE runs 
* Run EMADE using all possible combinations of the objectives and compare the AUC across trials
* Objectives: Loss Percentage Objective, Average Loss Per Transaction, etc. 
* Takagi-Sugeno fuzzy, Support Vector Regressio and fuzzy logic 
* Cannot replicate paper
* Not able to replicate PLR-SVR 
* Explore approaches to portfolio optimization, stock price prediction 
* Improve EMADE’s time-series analysis (weather series, heart rate data)

Modularity:

**Action Items:**
| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| --- | ----------- | --- | ----------- |----------- |
| Submit Subteam Preferences  | Done | 10/30/21 | 11/1/21  | 10/31/21 |

# Week 9: October 20th, 2021
## Overview
Workday for EMADE and final presentations.

## Team Meeting Notes
* Prepared for final VIP Presentation by continuing to run EMADE and obtaining results for MOGP evolution of individuals on Titanic dataset. Received help from VIP Alumni in fixing minor bugs in selection_methods.py file (selDCD() method not being able to take individuals with a length not a multiple of 4) and began setting up pymysql to pull in MySQL data as pandas dataframes and conduct analysis on metrics like average analysis time across generations, most frequently occurring primitives in Pareto optimal individuals in each generation, and number of valid individuals over time. 

## Subteam Notes
Met with team to run through presentation and ensure we were ready to share our findings of using EMADE for the Titanic problem as well as hear from main VIP teams on their team focus areas and get a sense of which team we would like to join after bootcamp. Had to adjust to Rayan D. dropping the VIP by reassigning presentation parts close to the presentation date and revising our work for the Titanic ML and MOGP assignments. 

**Action Items:**
| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| --- | ----------- | --- | ----------- |----------- |
| Finish Worker Process Setup + Adding Preprocessing to titanic_splitter.py | Done | 10/6/21 | 10/20/21  | |
| Run EMADE on preprocessed Titanic Data | In Progress | 10/6/21 | 10/14/21  | |
| Write Python scripts to capture output in graphs (Pareto frontier, AUC, Individuals Over Time, T-Stats, etc.) | In Progress | 10/6/21 | 10/20/21  | |
| Work on EMADE Presentation | In Progress | 10/6/21 | 10/25/21  | |

# Week 8: October 13th, 2021
## Overview
Workday for EMADE and MySQL remote connection setup. 

## Team Meeting Notes
* Worked with Rayan, Rayan, and Charlie to help set up their EMADE engines and install all dependencies. They are working on installing all remaining dependencies, after which we can test MySQL remote connections to the server I have created . 

## Subteam Notes
Met with team on Friday to ensure team could run worker processes and connect to the MySQL server I created. Worked with Charlie during Saturday's hackathon to add our group's preprocessing to the titanic_splitter.py file and run EMADE with the updated train-test folds. Noticed Pareto front individuals gradually developing and being stored in titanic schema in my localhost. We are planning to remove the 3rd objective from the evaluation function to ensure a direct comparison to the Titanic ML and MOGP projects. 

**Action Items:**
| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| --- | ----------- | --- | ----------- |----------- |
| Finish Worker Process Setup + Adding Preprocessing to titanic_splitter.py | Done | 10/6/21 | 10/20/21  | |
| Run EMADE on preprocessed Titanic Data | In Progress | 10/6/21 | 10/14/21  | |
| Write Python scripts to capture output in graphs (Pareto frontier, AUC, Individuals Over Time, T-Stats, etc.) | In Progress | 10/6/21 | 10/20/21  | |
| Work on EMADE Presentation | In Progress | 10/6/21 | 10/25/21  | |

# Week 7: October 6th, 2021
## Team Meeting Notes
### Lecture on EMADE
* Introduced concept of EMADE (engine to perform multi-objective algorithm design using genetic programming and a primitive set containing both simple primitives and ML learners from Python packages).  
* Looked at EMADE repository on GitHub and got a view of input file that specifies MySQL database configuration for EMADE output and parameters of evolution, launchGTMOEP.py which initiates the evolutionary process, and the gp_framework_helper.py file that contains references to primitives used to create EMADE individuals).
* Received information about presentation on Monday, October 25th where bootcamp and returning students will present their EMADE presentations and hackathon on Saturday, October 16th, where new students can receive help from returning students for EMADE setup and analyzing output from running EMADE on Titanic dataset. 

## Subteam Notes
* Worked with rest of team asynchronously to set up master and worker processes for EMADE. I am running the master process (main evolutionary loop), while the others are running the worker processes (evaluation function and results). I was able to run a master process successfully after rewriting the selNSGA2 method to only perform selectDCD on lists of individuals whose length is a multiple of 4. Having done this, I ran the master process again and noticed that inf fitness values are being printed for certain individuals. This will likely be resolved when we replace the existing preprocessing in the titanic_splitter.py with our own preprocessing, which handles null and invalid values. We will also ensure that my other team members are able to run worker processes today during our team meeting, and if not, tweak any specifications of my localhost such that it accepts remote connections. 

**Action Items:**
| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| --- | ----------- | --- | ----------- |----------- |
| Finish Worker Process Setup + Adding Preprocessing to titanic_splitter.py | Pending | 10/6/21 | 10/14/21  | |
| Run EMADE on preprocessed Titanic Data | Pending | 10/6/21 | 10/14/21  | |
| Write Python scripts to capture output in graphs (Pareto frontier, AUC, Individuals Over Time, T-Stats, etc.) | Pending | 10/6/21 | 10/18/21  | |

# Week 6: September 29th, 2021
## Overview
Presented Titanic ML and MOGP assignments and received feedback from Dr. Zutty and Dr. Rohling. Watched other subteams' Titanic presentations and asked questions about their approach and design choices for their ML learners and MOGP individuals. Received instructions for peer evaluations (due 10/8/21, only complete peer evaluations for those you have interacted with frequently).

## Team Meeting Notes
### Notes on Titanic ML and MOGP Assignment + Presentation
* Dr. Zutty provided feedback on our presentation and told us that NSGA II truncates any individual past the kth index in the list of selected individuals, and that shuffling the individuals and/or using selTournamentDCD would have enabled new individuals to enter the hall of fame throughout the evolutionary loop. 
* In addition, Dr. Zutty mentioned that adding floats as terminals in our primitive set would have allowed us to perform operations on those constants as well as the inputs from our Titanic feature set, improving the fitness of our individuals. 

## Individual Notes
* Presented slides 1, 2, 4, 5 from slide deck: https://docs.google.com/presentation/d/1tK83vBU6uQFYQGAivnSjWEM4Ghw3qJaGR5Py14BocJk/edit?usp=drive_web&ouid=106540897889834720619
* Answered Dr. Zutty's questions regarding the primitives we chose to use (should look into using float primitives) and how using the HOF might have limited diversity in our resampling of the population during each iteration of the evolutionary loop. 
* Talked to Charlie and Dr. Zutty after class about selNSGA and how it truncates individuals after the kth individual when performing selection. Dr. Zutty suggested using selTournamentDCD and shuffling individuals so as to promote diversity in the Hall of Fame. 

**Action Items:**
| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| --- | ----------- | --- | ----------- |----------- |
| Titanic ML, MOGP, EMADE Presentation | Completed | 10/6/21 | 10/25/21  | - |
| Peer Evaluation | Completed | 9/29/21 | 10/6/21  | 10/4/21 |
| Install EMADE | Pending | 10/6/21 | 10/13/21  | - |
| Update Notebook | Completed | 9/29/2021 | 10/6/2021 | 10/3/2021 |

# Week 5: September 22th, 2021
## Overview
Discussed Titanic ML assignment and findings related to data preprocessing and hyperparameter tuning and their impact on minimization objectives. Began research for Titanic MOGP assignment, wherein our goal is to us DEAP and genetic programming to develop a Pareto frontier of trees with simple primitives and our dataset's features as inputs. Attended meta-presentation on good presentation techniques (detailed graphs, concise bullet points, key takeaways, etc.) in preparation for next week's Titanic ML and MOGP presentation. Decided to meet with team on Thursday to exchange initial findings and develop a plan of action for the week. 

## Team Meeting Notes
### Notes on Titanic MOGP Assignment 
* Loosely typed Gp, strongly typed gp, simple primitives, not allowed to use default algorithms in DEAP, no mu + lambda, have to code algorithm yourself, can use selection,crossover, mutation operations, but cannot use algorithms
* Have to write genetic program yourself
* Evaluation function - at least two objective, False Positives and False Negatives
* Can add additional objectives to make it better
* Comparison of Pareto Front that genetic programming finds against ML codominant set
* Once we’ve chosen preprocessed dataset, common valuation function, selection, mating, mutation, hyperparameters, can start doing any research on different operators, selection mechanisms, mutation - write your own, etc.
* Bare minimum is to get comparison between ML and GP. 
* Generate predictions for every pareto optimal point
* Each column is taking a tree structure (ex. Age < 18, predict true, run that alg on test.csv, another might be sex == female, survived, another algorithm)
* Present findings next week 
* In DEAP - sel_tournament is not a multi objective selection operator, just compares multi-objective fitnesses via tuples, (10, 11) < (11, 9) would be true with selection tournaments (only using first value)
* AUC will be useful to evaluation GAs
* Minimization - adding area, maximization - eating area
* Punctuated equilibrium - goes many generations before getting Pareto optimal, then stops, then keeps going ,etc. 
* Multiple dimensions - area under surface, computations are expensive, more solutions and interest in pop, interesting genetic diversity with more objectives, help mutation operators - shrink operation, if I have same FN and FP but favor one with less nodes, Occam’s razor, re-compute pareto optimal front on 2 objectives you care about 
* Diff is a fence posting problem 
* AUC will be more comparable between us, can take AUC of ML learners as well, always include trivial solutions for GP and ML solutions. Do FPR and FNR consistently. 

## Subteam Notes (Titanic MOGP Problem)
### Data Preprocessing
* Created Google Colab notebook with same preprocessing as Titanic ML assignment
* Notebooks with preprocessing and preprocessing experimentation here: https://drive.google.com/drive/folders/1lq6fycfuDPxNamEK6inOa1vt8-RddgiS
* Researched strongly typed GP in DEAP (reference: https://deap.readthedocs.io/en/master/tutorials/advanced/gp.html)
* Chose NSGA II as selection method (handles both objectives)
* Created hof using best individuals ever identified throughout evolution
* Created graph of fitness across generations - ordinality of average FPR and FNR changed after evolution
* Removed duplicate individuals from Hall of Fame for improve evolution
* Predicted Survived feature for test.csv 
* Titanic ML and MOGP Presentation: https://docs.google.com/presentation/d/1tK83vBU6uQFYQGAivnSjWEM4Ghw3qJaGR5Py14BocJk/edit?usp=drive_web&ouid=106540897889834720619

## Individual Notes
* Created outline of implementation - selecting primitive set, defining evaluation function (fp, fn tuple), determining selection, mutation, and mating methods and probabilities, writing evolutionary loop for a given number of generations, comparing Pareto frontiers for ML and MOGP
* Focused on simple primitives so as to be able to predict on each sample's features at a time, improving granularity (reference: https://numpy.org/doc/stable/reference/routines.math.html)
* Primitive Set:
![Genetic Programming Visualization](https://picc.io/_TMo_MD.png)
* Tried mutUniform and cxOnePoint, AUC improved when using mutNodeReplacement and cxOnePointLeafBiased with termpb = 0.1
* Change 30 generations to 50 generations for improved evolution
* Titanic ML and MOGP Presentation: https://docs.google.com/presentation/d/1tK83vBU6uQFYQGAivnSjWEM4Ghw3qJaGR5Py14BocJk/edit?usp=drive_web&ouid=106540897889834720619 (Created slides 1, 2, 4, 5)

Sample Learner: logical_and(not_equal(Sex, negative(multiply(multiply(C, Parch), Age))), greater(Ticket, SibSp))

Best Learner: FPR = 0, FNR =  0.9122807017543859

MOGP Pareto Front:

![Genetic Programming Visualization](https://picc.io/Uot-hXd.png)

Findings:
The AUC for MOGP was much better than that of ML. Evolution in MOGP favored diversity and individuals tended to cluster near both trivial points. MOGP also saw individuals with high FPR and FNR rates, while the learners we use for our ML Pareto frontier tended to favor higher FNRs and lower FPRs. We were also able to generate the same set of predictions each time we re-trained the classifiers using the random_state parameter, but the random probabilities of mutation and mating in MOGP led to different predictions on test.csv each time we ran the evolutionary loop. Finding Pareto-optimal solutions was more difficult with ML, and less individuals existed on both extremes of both objectives, but genetic programming created a diverse set of individuals and had a much lower AUC. We also ensured that we took a split of the data that was the same split that we used to train our ML classifiers for the Titanic ML problem. Dr. Zutty provided feedback on our presentation and told us that NSGA II truncates any individual past the kth index in the list of selected individuals, and that shuffling the individuals and/or using selTournamentDCD would have enabled new individuals to enter the hall of fame throughout the evolutionary loop. In addition, Dr. Zutty mentioned that adding floats as terminals in our primitive set would have allowed us to perform operations on those constants as well as the inputs from our Titanic feature set, improving the fitness of our individuals. 


**Action Items:**
| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| --- | ----------- | --- | ----------- |----------- |
| Meet with Team to Discuss Evolutionary Loop and Evaluation Function | Completed | 9/22/2021 | 9/29/2021 | 9/23/2021 |
| Plot MOGP individuals with Pareto frontier and compare to ML results | Completed | 9/22/2021 | 9/29/2021 | 9/23/2021 |
| Create Slide Deck for Titanic ML and MOGP Presentation | Completed | 9/22/2021 | 9/29/2021 | 9/25/2021 |
| Update Notebook for Week 5 | Completed | 9/22/2021 | 9/29/2021 | 9/26/2021 |

# Week 4: September 15th, 2021
## Overview
Received bootcamp subteam assignments (I am in Bootcamp Subteam 4) and explored Kaggle Titanic dataset. Discussed Titanic ML assignment wherein each member of our subteam is to select an ML learner, use it to predict the 'Survived' feature in the Titanic dataset, and determine the FNR and FPR of that learner. All of our learners must be codominant, meaning that no learner should outperform any other learner on both minimization objectives (FNR and FPR). Exchanged contact information with team and decided to meet throughout the week and create Slack channel for communication. Discussed preliminary ideas for data preprocessing and hyperparameter tuning.

## Team Meeting Notes
### Notes on Titanic ML Assignment 
* nans, strings, balance data, fold data, make sure everyone is using same X_train, y_train, X_test, y_test
* Post csv representing predictions of your model that was co-dominant with rest of group. 
* Sci-kit learn - classification (ex. Support Vector machine)
* Do Pareto graphing for minimization objectives
* Pandas documentation
* Why did the decision classifier perform so well when we didn’t do that much?
* Make sure submission samples are in the same order for everyone 
* Pandas, sci-kit learn - dig deep 
* Use n folds
* Look at cabin values and encode Embarked 
* Do k fold splits for all learners
* Cross val score - average of false negatives and false positive 
* Look at average for nan values across samples with similar features versus all samples
* Create csv files with data that we’re using for preprocessing 
* Create a jupyter notebook to graph pareto frontier - everyone inputs their values
* Don’t mix up the rows
* Undersampling/oversampling 

## Subteam Notes (Titanic ML Problem)
### Data Preprocessing
* Created Google Colab notebook for group preprocessing
* Notebooks with preprocessing and preprocessing experimentation here: https://drive.google.com/drive/folders/1lq6fycfuDPxNamEK6inOa1vt8-RddgiS
* Imported pandas, numpy, and sklearn methods 
* Mounted Drive to Colab and read in train and test sets as dataframes
* Dropped Name feature (irrelevance) and Cabin feature (too sparse to work with)
* Set PassengerID as index
* Replaced Sex feature categories with 1 for male and 0 for female
* Split training data into training and testing sets (test_size=0.33, random_state=10)

## Individual Notes
* Created ParetoFront.ipynb for group to input objective values for individual learner and confirm co-dominance
* Replaced null values of Embarked feature with mode of Embarked column and null values of Ticket feature with '100'. Held off on replacing Age and Fare null values here and replaced them later with median value of each respective feature for a given Pclass. This is so that the null values in the Age and Fare columns are not replaced with values that are not representative of the central value of those features for all samples of a particular type (in this case, a particular Pclass). 
* One hot encoded Embarked feature values so as to not incorrectly assign a magnitude of value to each Embarked class (ie. 'Embarked': {'C': 0, 'Q': 1, 'S': 2} might cause our learner to assume a relationship between Survived and Embarked for rows with an Embarked class of 'S' and no relationship between Survived and Embarked for rows with an Embarked class of 'C'). Created three columns, 0, 1, 2, each of which is assigned either the value 0 or 1 for each sample based on the Embarked class for that sample. 
* Extracted numerical part of Ticket feature and re-assigned Ticket column values to numerical portion (type=integer). This is so as to consider the relationship between ticket assignments and survival empirically (for instance, those with lower ticket numbers may have purchased their tickets earlier than those with higher ticket numbers, which could indicate residence in a particular location of the ship (ex. the upper or lower deck) at the time of the crash, impacting survival). This feature engineering had little to no impact on the FNR and FPR of the model. 
* Replaced null Age and Fare values with median values based on Pclass of passenger (see above). 
* Selected XGBoost learner due to its speed and ability to handle null data
* Initially ran XGBoost predictions with default hyperparameters 
* Obtained confusion matrix for predictions 
* Modified XGBoost hyperparameters
Final Learner: XGBoostClassifier(objective="multi:softprob", num_class=2,  eta=0.005, max_depth=10, subsample=0.98, colsample_bytree=0.9, eval_metric="auc", n_estimators=10000, scale_pos_weight=0.2). Setting the max_depth, subsample, and colsample_by_tree parameters to relatively high values allowed us to sample each row in the dataset multiple times as well as increase complexity of each decision tree, which led to higher accuracy as well as minimization of the FNR and FPR. The eval_metric parameter allowed us to determine the AUC for each gradient-boosted decision tree created by the XGBoostClassifier(), which enabled us to achieve Pareto optimality among the decision trees. The n_estimator value allowed us to build more trees in each level of the boosting process, which increased complexity, but it also reduced the efficiency of the algorithm significantly. This learner had 31 False Positives and 26 False Negatives. 
Interestingly, using booster="gblinear" as opposed to the default booster="gbtree" dramatically decreased the FPR and increased the FNR. This indicates that the boosting technique is really the strength of XGBoost, as a linear booster did not distribute its false predictions evenly between the FNR and FPR. 

Findings:
Charlie's multi-layer perceptron classifier and my XGBoost learner had vastly different FNR and FPR values, given the same preprocessed data. Charlie's performed much better in the FPR objective and mine performed much better in the FNR objective. This indicates that neural networks, specifically MLP classifiers, tend to favor false positive prediction at the risk of accuracy while XGBoost favors even distribution of the FNR and FPR as well as high accuracy. SVM and Decision Tree, the classifiers that Rayan D. and Rayan K. implemented, tended to distribute their false positives and negatives evenly prior to any hyperparameter tuning, and max_depth used without the appropriate splitter for SVM and max_iter and decision_function_shape for Decision Tree reduced the accuracy of each learner respectively. This indicates the loss function, number of iterations, and the depth of individuals are critical to the success of ML classifiers. These parameters are analogous to the evaluation function, number of generations, and max tree depth in MOGP, which are also important factors for evolving GP individuals. Moreover, mating, mutation, and selection are analogous to hyperparameter tuning as described above. When we solve the same Titanic problem next week using MOGP, we will be able to more closely determine the tradeoffs between and ML and MOGP approach in solving a classification problem and which problems call for which approach. Additional improvements can be made to our learners by continuing to tweak the hyperparameters to achieve a particular FNR, FPR, and accuracy, as well as more advanced preprocessing techniques (normalization, removing noise, principal component analysis, etc.). 

Pareto Front for ML Learners:

![Titanic ML Pareto Graph](https://picc.io/qjd1y20.png) 

**Action Items:**
| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| --- | ----------- | --- | ----------- |----------- |
| Review Titanic Dataset and Preprocessing/Hyperparameter Tuning Techniques | Completed | 9/15/2021 | 9/22/2021 | 9/16/2021 |
| Titanic ML Learner Predictions| Completed | 9/15/2021 | 9/22/2021 | 9/17/2021 |
| Create Subteam Slack | Completed | 9/15/2021 | 9/18/2021 | 9/15/2021 |
| Meet to Discuss Individual Learners' Performance | Completed | 9/15/2021 | 9/18/2021 | 9/18/2021 |
| Update Notebook | Completed | 9/15/2021 | 9/22/2021 | 9/19/2021 |

# Week 3: September 8th, 2021

## Overview
Attended lecture on multi-objective optimization and completed Lab 2's Multi-Objective programming exercise. Filled out survey sheet with ML and Python self-ratings.  

## Team Meeting Notes
### Lecture on Multi-Objective Optimization:
* Accuracy, speed, memory helps define metrics that an algorithm might look for in a mate
* Scalability, reliability, adaptability, consistency (tradeoff between precision and accuracy)

* Search space - all of the things that make up an individual (one max - 1,0 for each element in the list for how many elements there are)
* Limitations in memory limit how deep our algorithms can be. Full search space is the full set of possible algorithms. 
* How can we come up with good fitnesses to help us search these worlds?
* Objectives are our phenotype, evaluation function allows us to go from the genotype to the phenotype. 
* Binary classification vs. multiclass classification - 1 or 0 versus in a set of things

* Algorithm is for red objects, we are trying to find apples.

* Precision or positive predictive value - overlooked but very important
* Assessing algorithm’s consistency of performance with itself, regardless of truth 
* Accuracy - bigger is better
* Blue and green form a tradeoff space between each other against Objective 1 and Objective 2

* Dominated solution - there is an individual that would live in the space to the left and under a given point 
* Non dominated - there is no such individual 
* Co-dominant, none are dominated, form Pareto frontier 
* Want to keep diversity of genotypes, want all tree structures to stay in population, algorithms will continue to be diverse (their representations are diverse), want to reward this, stops algorithms from converging 
* Nondominated solution is called Pareto optimal in this class
* Would much rather have spread out points on Pareto frontier than clumped up individuals on either end in Pareto frontier
* Reward places that are off by themselves so we can keep that diversity 
* Higher crowding distance wins

* SPEA2: How many points does it dominate (look up and to the right)
* S is how many others in the population it dominates
* Rank is the sum of S’s of the individuals that dominate it 

* Tiebreakers: Fractional so serves as tiebreaker, one with higher distance is going to have a smaller effect on rank, if crowding distance is smaller, you’ll be closer to 1, almost at the next range, favor larger distance because it will get inverted 
* Niching - trying to spread diversity 
* Both algorithms favor nondomination of something more highly than how different it is from everything else. 
* Kth nearest neighbor - look at Euclidean distance in a space for all points to a kth neighbor 
* Larger the distance, the better, minimizes the 1/sigma, which minimizes the rank + 1/sigma 

## Lab 2 - Multi-Objective Genetic Programming
This lab explored the problem of optimizing a set of primitives based on more than one objective to achieve a target function model. Here, we minimize the mean squared error and the size of the tree. We also add the sin, cos, and tan functions to our set of primitives and reinitialize the toolbox. We then define a function to evaluate our symbolic regression and note that this new problem, with an evaluation function that takes the sin, cos, and tangent of the points into consideration when evaluating the individuals for fitness, cannot be solved within 100 generations like the ones we worked on previously. 

We then define the pareto dominance function, which compares two individuals and returns the individual which dominates the other in the objective space. We initialize 300 individuals and leave one individual as the comparison individual. We then sort the population we created by each individual's Pareto dominance as compared to the "spare" individual. Plotting the objective space, we are able to visualize the individuals that minimize both objectives and exist along the Pareto front using the Hall of Fame. Running the evolutionary algorithm, we identify the Best Individual: negative(cos(multiply(add(cos(sin(cos(sin(cos(tan(x)))))), cos(x)), tan(x))))
with fitness: (0.2786133308027132, 15.0). 

DEAP's Mu plus Lambda algorithm (reference: https://deap.readthedocs.io/en/master/api/algo.html), which takes in a mu and lambda value (number of individuals to select for each successive generation, and the number of children to produce at each generation), allows us to control the size of the population as well as the selection process between individuals. We identify that the size of our trees grows over generations, but the MAE quickly drops to a sub-1 value over generations. Visualizing our pareto front, we see that the Area Under Curve: 2.3841416372199005 indicates the amount of objective space that exists below our current Pareto front. 


Improvements:
Modifying the following hyperparameters reduced the AUC of the Pareto front to 0.3113. 
* NGEN = 50
* MU = 60
* LAMBDA = 75
* CXPB = 0.4
* MUTPB = 0

Original Hyperparameters:
* NGEN = 50
* MU = 50
* LAMBDA = 100
* CXPB = 0.5
* MUTPB = 0.2

Visualization:
[Screenshots](https://docs.google.com/document/d/1iIiZlL-WCdWpetdyYBEG_TXH59vqYatcfh7eqzxu6b8/edit)



Observations and Reflection: The original evolutionary loop produced individuals that were diverse but led to a large AUC (~2.38). In addition, the average and minimum tree size of individuals grew over the course of evolution, while the average and minimum mean squared error decreased almost immediately starting at evolution. With the modified hyperparameters for evolution, the average and minimum tree size of individuals stagnated quickly, and the average and minimum mean squared error decreased quickly as before. There were also fewer individuals in the Pareto front, but they were fairly diverse as before, and they had a much lower AUC (~0.31). As such, tuning the hyperparameters of evolution such as the number of individuals to select for each generation, the number of children to produce for each generation, and mutation and mating probabilities significantly improved the performance of our individuals. In particular,  decreasing the number of individuals selected at each generation, increasing the number of children produced at each generation, increasing crossover probability, and eliminating mutation altogether significantly improved the AUC. This indicates that starting with a fewer strong individuals and favoring information exchange between them as opposed to mutation/data imputation leads to a much fitter Pareto front than starting with many more individuals, several of which cannot be pushed to the Pareto Front easily with mutation, mating, and selection. In addition, the average tree size after modifying the hyperparameters and running the evolutionary loop was around 4, while the average tree size without modifying the hyperparameters was around 10. We are able to obtain smaller, more simple trees overall when we begin with stronger individuals and perform crossovers frequently between them so as to push simpler, fitter trees to the Pareto front. 

Additional improvements can be made to the current genetic programming algorithm such that we obtain an individual with the optimal fitness in a minimum number of generations. We can continue to tweak the probabilities of mutation and mating for offspring, change the tournament size, change our methods of mating, mutation, selection, etc., change the parameters of our mating and mutation (ex. points of mating, values that the data in our individuals can be mutated to), and change our evaluation function.

Strongly Typed Genetic Programming: 
* Used PrimitiveSetTyped object to create primitive set with strongly typed primitives and terminals 
* Ephemeral constants are produced by functions and therefore allow terminals which depend on other operations/randomness to be included in our trees, improving the diversity of our population.
* Used static limits and tree size as an objective to limit excessively large trees given Python's limitations and the fact that we usually favor simpler individuals over more complex individuals that have the same fitness. 

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

# Week 2: September 1st, 2021
## Overview
Attended lecture on genetic programming and completed Lab 2 on the same topic. Continued to discuss course expectations and direction after 10-week bootcamp. 

## Team Meeting Notes
### Lecture on Genetic Programming
Introduced concept of genetic programming with the goal of optimizing a function (represented as a tree structure) to achieve a particular target output.  
1. Nodes: primitives, represent functions 
2. Leaves: terminals, represent parameters 
3. Explored examples of lisp preordered parse trees that represent functions
4. Crossover in GP: exchanging subtrees
5. Mutation in GP: Inserting/deleting nodes and subtrees
5. Measuring error (ex. Mean Squared Error)
7. Identifying primitives that can make modeling a function easier 

## Lab 2 - Genetic Programming, Part 1 (Symbolic Regression)
This lab explored the problem of optimizing a set of primitives to achieve a target function model. This exercise is in contrast to typical machine learning or data modeling, wherein we attempt to fit a function to data. Here, we use the mean squared error to obtain the fitness of each individual in the population; that is, we determine the MAE between our primitives-based function and the target function.   

We first create our fitness and individual classes, where individuals are of the PrimitiveTree type. We then initialize the set of primitives our trees can draw from (add, subtract, multiply, and negative) and register our objects with the DEAP toolbox. We also define our evaluation function (which uses the MAE between the modeled function and the actual function) and register the evaluation, selection, mating, and mutation operators with the DEAP toolbox. We used selTournament, one-point corssover, uniform mutation, and gp.genFull for our functions. As we registered the expression for generating our population, we defined a minimum and maximum height for our tree. We passed points=np.linspace(-1, 1, 1000) to register the evaluation function with the toolbox in order to generate 1000 random points between -1 and 1 to pass to each tree in our population. We then programmed the same evolutionary algorithm that was used in Lab 1 for the n-queens problem and obtained the best individual after 40 generations. We set a 0.5 probability of mating and a 0.2 probability of mutation. We graphed the results and printed our statistics. 

Findings: The global maximum (a best individual with a fitness or MAE of 0) was almost reached. The best maximum individual reached a minimum fitness value of around 1.5. The average and minimum fitnesses approached a fitness of 0 closely (0 was an asymptote for these values). Further improvements can be made by changing the bounds of the random number generation for crossover, mutation, and selection.
  
The best individual was determined to be the following: Best individual is add(add(multiply(x, x), multiply(add(multiply(x, multiply(x, x)), multiply(x, x)), x)), x), (8.620776339403237e-17,). 

Visualization:
![Genetic Programming Visualization](https://picc.io/x91IjkA.png)

* We can see here that the maximum fitness value seems to oscillate around a fitness of about 2.0 and does not continue decreasing after about the 10th generation. 

To improve the fitness of the evolved individuals, I added the floor and maximum operations from the numpy library. I also registered the mutInsert function as an additional mutational function to use in the evolutionary loop. This mutation function randomly selected a node in the given individual and creates a new subtree using that node as a child to that subtree. The following graph of fitness over time reflects those changes:

![Genetic Programming Visualization after Floor and Maximum Primitives Added](https://picc.io/Un4_bet.png)

We see that modifying the primitive set and the mutation function being used in the evolutionary loop caused the maximum fitness line to decrease much more quickly than it did for the original evolution, indicating that floor, maximum, and mutInsert ensure optimal fitness for all individuals in a population by minimizing the fitnesses of higher-extreme individuals. 

Increasing the minimum and maximum tree depth when registering the expression function with the toolbox appeared to slow the improvement of the maximum, average, and minimum fitness across the same number of generations. This indicates that larger trees are not necessarily fitter or prone to becoming fitter through evolution than smaller trees. Evaluating tree size as an objective itself will reappear in the second part of Lab 2, at which point I will delve into the different outcomes that larger and smaller trees produce with the same evolution. 

In addition, I experimented with higher and lower values for the probability of mutation and mating, but all such combinations slowed the fitness improvement of the individuals and caused the maximum fitness line to begin at a higher fitness value than without changes to the mutation and mating probabilities. This indicates that excessive mutation and mating can actual lead to worse-performing individuals emerging from the evolutionary loop than those that entered it. 

Additional improvements can be made to the current genetic programming algorithm such that we obtain an individual with the optimal fitness in a minimum number of generations. We can continue to tweak the probabilities of mutation and mating for offspring, change the tournament size, change our methods of mating, mutation, selection, etc., change the parameters of our mating and mutation (ex. points of mating, values that the data in our individuals can be mutated to), and change our evaluation function.

**Action Items:**
| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| --- | ----------- | --- | ----------- |----------- |
| Continue to install DEAP and supporting libraries | Completed | 9/1/2021 | 9/8/2021 | 9/6/2021 |
| Complete Lab 2: Genetic Programming | Completed | 9/1/2021 | 9/8/2021 | 9/6/2021 |
| Review Genetic Programming Notes | Completed | 9/1/2021 | 9/8/2021 | 9/6/2021 |
| Update Notebook | Completed | 9/1/2021 | 9/8/2021 | 9/6/2021 |

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
6. Evaluate population to get objectives, then evaluate objectives to get fitness. 
7. Room for implementation choices, hyperparameters, crossover, mating, probabilities, mutating, objectives. 

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
For this problem, we followed many of the same steps that appeared in the Lecture 1 Notebook (see above). We define a main() function for the genetic algorithm, which evaluates the full population and initiates the evolutionary loop. Our population size was 300, and our probability of mating was 0.5 and our probability of mutation was 0.2. Within the evolutionary loop, we select individuals for each successive generation, clone them, and perform mutations/crossovers on them. We then evaluate the fitness of these offspring and replace the existing population with the offspring. Finally, we return the fitnesses of the individuals (based on the predefined fitness operation - the sum of the individual's entries) and print statistics such as the mean fitness, squared sum of the fitnesses, and standard deviation of the fitnesses). We loop for some number of generations (40, in this case) and report the best individual that has resulted from this evolution process. Within the DEAP framework, we used libraries like creator (including the create() method), tools (including the selBest() method and the selTournament, mutFlipBit, and cxTwoPoint attributes), and base (including the Toolbox(), register(), select(),  mate(), and mutate() methods). We kept track of our minimum, maximum, average, and standard deviation for the fitness values for each generation so as to determine if fitness was improving with evolution.

Findings:

The best individual was as follows: Best individual is [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], (100.0,). 

The global maximum (a best individual with a fitness equal to n, the number of entries in each individual) was reached within 40 generations about every 19 out of 20 times the algorithm was run; this indicates that our algorithm has an effectiveness of around 95%. Further improvements can be made by changing the bounds of the random number generation for crossover, mutation, and selection, increasing/decreasing the size of the population and the number of generations, as well as trying other crossover and mutation methods. 

![One Max Generations, Part 1](https://picc.io/pok5sgG.png)
![One Max Generations, Part 2](https://picc.io/ouFv77h.png)

**N Queens Problem:**
For this problem, we followed many of the same steps that appeared in the One Max Problem (see above). We define a size n = 25 for each individual and define a weight of -1.0 here, since we wish to minimize the number of conflicts between queens in our problem space. We then create a permutation function to populate the entries for each individual with numbers selected without replacement from range(n). For instance, the following individual will have a queen in the 1st row and the 8th, the 2nd row and the 1st column, etc.: [7,0,4,6,5,1,9,2,8,3]. We define our evaluation function as a measure of the number of conflicts along each diagonal of our board; with the creation process we defined for individuals, queens will not appear in the same row or column. [Describe evaluation function modification here w/ screenshots]. We then write the cxPartialyMatched() function for partially matched crossover, cxTwoPoint(), and mutShuffleIndexes() to shuffle values at different indexes within each individual (since we must remain within size n  = 25). We modified the mutation function to be a uniform int mutation, wherein randomly selected entries for each individual are replaced with a randomly selected value between 0 and n. The improvements seen with this new mutation function are described in the Findings section below. Finally, we run a similar evolutionary loop as the one described for the One Max Problem (see above) for 100 generations, return the fitnesses of the individuals (based on the predefined fitness operation - the number of conflicts between queens) and print statistics. We loop for some number of generations (100, in this case) and report the best individual that has resulted from this evolution process. 

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

I also modified the evaluation function to return a tuple (sum_left, sum_right), where sum_left is the number of conflicts between queens strictly on left diagonals and sum_right is the number of conflicts between queens strictly on right diagonals. This allowed us to attempt minimization of both objectives in the evolutionary loop such that the evolution did not produce many individuals that had a high number of conflicts on left diagonals or right diagonals but not distribute conflicts on both types of diagonals. 

4. With (left_diagonal_conflicts, right_diagonal_conflicts) as fitness tuple
![](https://picc.io/7PFcY8A.png)

As we can see in the above graph, the best individual was reached much more quickly using the left and right diagonal fitness tuple than simply the sum of the conflicts in the evaluation function. However, in the evolutionary function, the fitness being considered was only the first index in the fitness tuple (that is, the number of conflicts on left diagonals). As such, this indicates that our algorithm quickly minimizes left diagonal conflicts when our evaluation function contains both left and right diagonal conflicts. Next week, when we learn multi-objective genetic programming and an appropriate evaluation function to handle both left and right diagonals, I will be able to identify how fitness on both objectives improved over time in the fitness graph and obtain a best individual that is based on both objectives, not just the number of conflicts on left diagonals. 

Additional improvements can be made to the current n-queens algorithm such that we obtain an individual with the optimal fitness in a minimum number of generations. We can continue to tweak the probabilities of mutation and mating for offspring, change the tournament size, change our methods of mating, mutation, selection, etc., change the parameters of our mating and mutation (ex. points of mating, values that the data in our individuals can be mutated to), and change our evaluation function.

**Action Items:**
| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| --- | ----------- | --- | ----------- |----------- |
| Install DEAP and set up JupyterLab for Lab 1 | Completed | 8/25/2021 | 9/1/2021 | 8/30/2021 |
| Complete Lecture 1: GA Walkthrough | Completed | 8/25/2021 | 9/1/2021 | 8/30/2021 |
| Complete Lab 1 | Completed | 8/25/2021 | 9/1/2021 | 8/30/2021 |
| Set Up Notebook | Completed | 8/25/2021 | 9/1/2021 | 8/30/2021 |
| Review Genetic Algorithms | Completed | 8/25/2021 | 9/1/2021 | 8/30/2021 |