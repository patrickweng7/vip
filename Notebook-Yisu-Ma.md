**Name:** Yisu Ma

**Email:** yma391@gatech.edu

**Threads:** Info-Networks && AI

**VIP:** Automated Algorithm Design

**Interests:** Fingerstyle guitar, Soccer, K-pop, Camping

# **Fall 2021**
## Week 16: December 6th -  November 10th (2021)
### Lecture overview:
* Final presentation day
### Image Processing:
* Outcomes:
1. image resizing, normalization, rotation and especially sharping. 
* Methods:
1. Used 2 new geometric mating and mutation methods
2. Semantic Crossover: attempt to find subchild and offsprings
3. NSGA II and NSGA III
4. Talked a lot about Lexicase: perform worse than NSGA II and has worse AUC
5. using formula s = T(r)
* Some takeaways:
1. Overall. they have the best result from NAGA II
2. Test usinf hyper-features, I think they had a good performance with more parameters

### NLP:
* THIS IS MY TEAM!
* I talked about a little bit of our work in NNlearner2 and go through a refresh on NNlearner
* The audience asked some questions about QA model. We also made some clearance for the contents

### Stock
* Objective: optimize market trading
* Short term: do analysis on different papers
* Used 3 objective sets in EMADE
* Did a fancy profit percentage analysis
* Experiment beats the paper and SOTA
* Continued the work after midterm 

### NAS
* Goal: have more complex individuals on EMADE
* Added different generating methods for NNlearner
* Use resnet architectures in EMADE for weight Sharing which is implemented by storing weights in database
* Got more complex individuals by adding different generator functions.
* Did some paper review
* have clear goals: using pooling layers effectively.

### Modularity:
* Objectives: increase complexity of ARLs
* Updated the documentations
* moved information within tuples into classes
* rewrite some of the methods
* Increased the ARL size and they explained its relation with accuracy.
* The new ARLs perform better than old ARLs
* Merged Cache V2
* Future goal:
1. Change hyper parameters for ARL
2. check research papers for they data

### subteam notes:
* Joined the bluejeans meeting and went through all the slides we have and double confirmed my assignment on NNlearner2.
* Recap the information we need to present for NNlearner2 
* My teammates solved the dependency error in the input_squad.xml file, which enables me to run NNlearner2
* My runs are very short, because of the VPN issue. Though I didn't get enough generations for training, the runs were success and I accumulated some useful tips for working with PACE-ICE.
* Finish all the runs, and we found that the reuse should be sat to 1 in EMADE
* Upload the master.out files.

Task | Current Status | Date Assigned |  Suspense Date | Date Resolved |
|------|----------------|---------------|----------------|---------------|
|RUN EMADE on PACE-ICE using NNlearner2|finished|December 6th, 2021|December 10th, 2021|December 8th, 2021|
|Finalize our slides and tasks for presentation|finished|December 6th, 2021|December 10th, 2021|December 8th/9th, 2021|
|Collect all my knowledge  for NNleanrer2|complete|December 6th, 2021|December 10th, 2021|December 8th, 2021|
|Record notebook|complete|December 6th, 2021|December 10th, 2021|December 10th, 2021|


## Week 15: November 29th -  December 5th (2021)
### Lecture overview:
* Need to finish Peer Evaluation
* Final presentation details confirmed DEC. 10th
* NLP scrum:
1. NNlearner2 can now work on regression. 
2. Optimized some running time complexity

* NAS scrum:
1. Had code freeze

* Modularity:
1. Standardizing the coding conventions and comments

* Image processing:
1. reaching code freeze
2. got new results from baseline run
 
### subteam note:
* We are still focusing on running experiments on standalone and seeding test. Part of the process are paused due to PACE-ICE maintenance.
* Need to freeze the fork of code https://github.gatech.edu/sleone6/emadebackup

### Individual notes:
* I resumed my work and tried to use standalone_tree_evaluator to test our output. But then we need to reinstall EMADE using the back up fork, so I was dealing with that. This time I pay a lot of attention on the password setup part.
* Looked through my teammates's result for running seeding_qa.
* The team is focusing of wrap up the data we have and prepare for final presentation. Some of the teammates are running squad.

Task | Current Status | Date Assigned |  Suspense Date | Date Resolved |
|------|----------------|---------------|----------------|---------------|
|Reinstall EMADE using the new freezed fork|finished|December 1st, 2021|December 5th, 2021|December 1st, 2021|
|run standalone tree evaluator on PACE ICE|finished|December 1st, 2021|December 5th, 2021|December 2rd, 2021|
|Looked through my assigned part for presentation|in progress|December 1st, 2021|December 5th, 2021|December 5th, 2021|
|Record notebook|complete|December 1st, 2021|December 5th, 2021|December 5th, 2021|
|Peer evaluation|complete|December 1st, 2021|December 8th, 2021|December 6th, 2021|




## Week 14: November 22nd -  November 28th(2021)
### Lecture overview:
* Image processing:
1. On boarding new members
2. Had poor results on their baseline runs
* Modularity:
1. Bootcamp members working on data visualization
2. Dealing with Cache V2 integration

* NLP Scrum:
1. Got all the primitives for final presentations
2. NNlearner2 bugs on classification are fixed
3. Regression need to work on NNlearner2.
4. Bootcamp students can try to run experiment on data we have

### Subteam Notes:
* No meeting because of thanksgiving 

### Individual Notes:
1. Tried to run the experiment in PACE, but forgot my password for database.
2. Followed the instructions on Youtube and fixed the issue
3. Tried to use standalone and seeding to test our output, but meet some issues with my VPN.
Will do this again when I come back school.
4. Continued reading this paper: https://www.researchgate.net/publication/339980709_Sentiment_Analysis_with_NLP_on_Twitter_Data

Task | Current Status | Date Assigned |  Suspense Date | Date Resolved |
|------|----------------|---------------|----------------|---------------|
|read Sentiment Analysis on Twitter data paper|November 22rd, 2021|November 30th, 2021|pending|
|run standalone tree evaluator on PACE ICE|in progress|November 22rd, 2021|November 30th, 2021|in progress|
|database password issue|fixed|November 22rd, 2021|November 30th, 2021|November 25th, 2021|
|Record notebook|complete|November 22rd, 2021|November 30th, 2021|November 30th, 2021|



## Week 13: November 15th -  November 21st(2021)
### Lecture overview:
* Need to keep notebook updated
* NLP Scrum:
1. Specified tasks in different subteams
2. Had a lot of bug after merge
3. Encountered memory error from EMADE. The solution is to set the memory limit param in the xml

### sub-Team meeting note:
1. Our NNlearner2 team has new members 
2. modified the code base from NNlearner and try to make it accept 2 data pairs
3. Got NNLearner2 working on classification.
4. Still working on integrating 2 data pair 
5. Reduced the scope to only regression to for this semester, because of the complexity

### Individual Note:
1. Helped he subteam change modified our xml file together
![](https://github.gatech.edu/yma391/VIP-AAD/blob/master/aadxml.jpeg)
2. Helping Devan check if our layers are able to fit into Keras model
3. Looked trough our branch and this is our github link:https://github.gatech.edu/sleone6/emade/commit/77992e059d9d10b5174632a859c514b626d31d92
4. Read several articles and papers on Sentiment Analysis with NLP on Twitter data
https://medium.com/analytics-vidhya/introduction-bd62190f6acd

https://ieeexplore.ieee.org/abstract/document/9036670


Task | Current Status | Date Assigned |  Suspense Date | Date Resolved |
|------|----------------|---------------|----------------|---------------|
|Look through useful resource for twitter dataset NLP|in progress|November 15th, 2021|November 30th, 2021|pending|
|run EMADE on PACE ICE environment|complete|November 15th, 2021|November 21st, 2021|November 19th, 2021|
|check layers for Keras model|complete|November 15th, 2021|November 21st, 2021|November 20th, 2021|
|Record notebook|complete|November 15th, 2021|November 21st, 2021|November 21st, 2021|




## Week 12: November 8th -  November 14th (2021)
### Lecture overview:
* NLP Scrum:
1. Onboard new members
2. The old members tries to help new member build ICE-PACE environment
3. Integrate new primitives
4. Discussed if we shpould include Neural Network in Bootcamp

###Subteam Notes:
1. Placed into NNlearner2 data pairs team
2. the goal is adding the nnlearner2 method that works with 2 data pairs for text type data
3. Trying to successfully able to rum EMADE with amazoninput dataset.
4. Sat up MariaDB

###Individual notes:
1. NNLearner: type of learner in EMADE
Layers of neural networks are primitives in EMADE
2. Need to make sure NNlearner2 does not run into error
3 I tried to download twitter data set and ran the standalone tree evaluator, but error occurs in our nnlearner2.
Devan and Steven would try to fix the bugs in Hackathon

Task | Current Status | Date Assigned |  Suspense Date | Date Resolved |
|------|----------------|---------------|----------------|---------------|
|Create input.xml with twitter dataset|complete|November 8th, 2021|November 15th, 2021|pending|
|Record notebook|complete|November 8th, 2021|November 15th, 2021|November 15th, 2021|
|set up PACE ICE database|complete|November 8th, 2021|November 8th, 2021|November 15th, 2021|





## Week 11: November 1st -  November 7th (2021)
### Lecture overview:
* Introducing myself to NLP team
* Everyone greet with each other
* Get to know the some basics of NNlearners
### Team meetings:
* Joined NLP slack
* Talking about different layers: BIDAF ...
* Our leader present a Q&A during the first remote meeting
* Currently implementing  Modeling layer and output layer

 Task | Current Status | Date Assigned |  Suspense Date | Date Resolved |
|------|----------------|---------------|----------------|---------------|
|Join team slack|complete|November 1st, 2021|November 3rd, 2021|November 3rd, 2021|
|Record notebook|complete|November 1st, 2021|November 7th, 2021|November 7th, 2021|
|set up PACE ICE|in progress|November 1st, 2021|pending, 2021|pending|





## Week 10: October 25th - November 31st (2021)
### Lecture overview:
* Presentation on 10/25
* Slides link: https://docs.google.com/presentation/d/1ShDz-7hPoor3ExWA9BKqiSzqn-G4ufgBYWor-mtlzdU/edit?usp=sharing
* Watched presentations by different sub-teams

### Sub-team presentations:
* NLP:
1. The goal is to make machines understand natural language
2. Doing some literature reviews
3. Using SQUAD dataset
4. Meeting time: Wednesday 2:00 pm
* NAS:
1. Preprocessing:
Text tokenization and one hot encoding for multi-class target data
2. CoDEEPNeat:
Separated primitive sets and terminals between ADFs and MAIN primitive
(didn't really understand this part)
3. The goal is to improve the productivity of EMADE
4. Used new analysis methods: nn_ind_from_has(hash) and view_nn_statistics
5. Meeting time: Friday 2 - 3pm
* Image processing:
1. The goal is object detection, image segmentation
2. Focused on a CheXNet (chest X-rays)
3. 3 selected methods: NSGA-III, Hypervolume indicators and Lexicase:
Talked about the difference between NSGA III and NSGA II.
4. Used Tensorflow to normalize, flip and resize the images
5. Results are obtained on 30 generations
6. added new mutation and mating function.
7. Some generated individuals cannot generate better results
* Modularity:
I went to my lab so I missed it.
### Bootcamps presentation:
* Team1:
1. Did similar data preprocessing as us: Drop columns, delete irrelevant variables
2. used SVM classifier and other bunch of machine learning methods.
* Team2: (presented my part)
* Team3:
1. Did similar data preprocessing methods
2. Talked about the troubles they met
3. Compared the Pareto front
* Team4: Went to my lab session
### Individual note:
* Probably I will choose NLP, because the goal and presentation is very sound

 Task | Current Status | Date Assigned |  Suspense Date | Date Resolved |
|------|----------------|---------------|----------------|---------------|
|Presentation|complete|October 25th, 2021|October 25th, 2021|October 25th, 2021|
|Record notebook|complete|October 25th, 2021|October 25th, 2021|October 25th, 2021|



## Week 9: October 20th - October 26th (2021)
### Lecture overview:
* EMADE working session
* Solving questions related to MySQL an EMADE installation
### Team meeting notes:
* Figured out how to run master process and work process

`Server host command: mysql -h hostname -u username -d database_name -p`

* Encountered errors;

`Issue where fitness values for individuals were (inf, inf, inf)
 EMADE error stated that "Tree missing valid primitve for data type"`
* Connected to Mysql database
* Met difficulty on getting generations after 11
* Worked on slides for presentation on 10/25

### Individual notes:
* Rohan and Manas talked about the idea of using virtual conda environment to fix invalid version issue
* Trying to join worker process, but need to do it under gatech network
* Modified the code for plotting graphs
* Finalized our slides

| Task | Current Status | Date Assigned |  Suspense Date | Date Resolved |
|------|----------------|---------------|----------------|---------------|
|Sub-team meeting|complete|October 20th, 2021|October 26th, 2021|October 20th, 2021|
|Settle environment|pending|October 20th, 2021|October 26th, 2021|October 25th, 2021|
|Make Mysql work process working|in process|October 20th, 2021|October 26th, 2021|in progress|
|Record Notebook|complete|October 20th, 2021|October 26th, 2021|October 26th, 2021|




## Week 8: October 13th - October 19th (2021)
### Lecture overview:
* Work with the group
* Try to run EMADE properly
* Try to run Titanic dataset on EMADE and join the master process
* Need to have 1 master program and other worker nodes for MySQL.
* Meet version problem when setting up environment
### Lecture notes:
* Dr.Zutty answered questions from students
### Team notes:
* Successfully get EMADE to recognize our Mysql database
* Experiencing EMADE error - "Tree missing valid primitive for data type"
* Try to fix the problems evaluation functions
* Delete Python 3.9 in my computer and downgrade it to Python 3.7

### Individual notes:
* Investigated worker output file
* Faced some issues in joining worker process, need to access it in school network

| Task | Current Status | Date Assigned |  Suspense Date | Date Resolved |
|------|----------------|---------------|----------------|---------------|
|Sub-team meeting|complete|October 13th, 2021|October 20th, 2021|October 20th, 2021|
|Settle environment|pending|October 13th, 2021|October 20th, 2021|in progress|
|Make Mysql work process working|in process|October 6th, 2021|October 20th, 2021|in progress|
|Record Notebook|complete|October 13th, 2021|October 20, 2021|October 20th, 2021|







## Week 7: October 6th - October 12th (2021)
### Lecture overview:
* Notebook completion for mid term grading.
* projects introduction
* midterm Presentation date on 10.25
* Introduction to EMADE and MySQL.
* no bootcamp session on Wednesday
* Hackathon 10.16 or 10.17

### Lecture Notes:
* Introduced EMADE
* The basic concept for EMADE combines a multi-objective evolutionary search with high-level primitives to automate the process of designing machine learning algorithms
* Need to follow the install instructions and configure MySql server
* Input .xml file is required for fulfilling configurations in EMADE.
* Make sure the MySQL command works before troubleshooting EMADE
* Understand the EMADE structure and see the sample output.
evalFunctions.py is our evaluation function file
### team note:
* Everyone need to have EMADE downloaded and run it properly
* clone the repository 
* set up the package
* Set up MySql
* One person need to have Master process and other teammates need to join the work process
* Encode "Embarked" feature in dataset 
* Discussed our MOGP assignments
* Configure XML file to run Titanic Dataset
* Trying to resolve the problems during sql work process set up

| Task | Current Status | Date Assigned |  Suspense Date | Date Resolved |
|------|----------------|---------------|----------------|---------------|
|Sub-team meeting|complete|October 6th, 2021|October 13th, 2021|October 12th, 2021|
|Installed EMADE|complete|October 6th, 2021|October 13th, 2021|October 12st, 2021|
|Make Mysql work process working|in process|October 6th, 2021|October 13th, 2021|October 13th, 2021|
|Record Notebook|complete|October 6th, 2021|October 13th, 2021|October 13th, 2021|



## Week 6: September 29th - October 5th (2021)
### Lecture overview:
* Finish peer evaluation next week.
* Listened to different groups' presentation
* Presented my part in our group presentation:
 > The selection of our select, mutate, and evaluate function.
 > The improvement we made for the genetic loop.

**Link:** https://docs.google.com/presentation/d/1E5DIPJOt7uBeqUeYklg6TE7X7PTdOsaFdUjTDCrttkU/edit?usp=sharing
### Lecture note (I just record some of the ideas from each group):
* Group4: 
> AUC is around 0.2, a little bit higher than ours.
> They let us realized that Embark might not be a good column to be included in our data. Just as I thought in week 5, because this feature would not affect the cabin position as my team thought.
* Group3:
> Used KNN,  MLP,  SVM, logistic regression, random forests
* Group1:
> They seem to get a better result in GP than ours
### Individual notes:
* Finished peer evaluation
* Re-done through the data processing and take a note of the change I got

| Task | Current Status | Date Assigned |  Suspense Date | Date Resolved |
|------|----------------|---------------|----------------|---------------|
|Peer evaluation|complete|September 29th, 2021|October 5th, 2021|October 1st, 2021|
|Record Notebook|complete|September 29th, 2021|October 5th, 2021|October 1st, 2021|




## Week 5: September 23th - September 29nd (2021)
### Lecture overview:
* Discussed last week's project
* Learned how to derive Pareto Optimization Curves
* Learned how to deliver a great presentation
* Made a discussion with the teammate for the project next week.
### Lecture notes:
* Hyperparameters would highly likely affect the models and the codominant algorithm results.
* In order to examine our multi-objective solutions, we would need to use Pareto Optimization Curves 
* Try to write our own algorithm – using selection, crossover, mutation functions in Deap and try to figure out the best fit for our model.
* There were some constraints on MOGP given in the class: Only using basic primitives and 
No selTournament for selection, because it is not multi-objective
* Submit .csv with columns of passengerID
* Finally compare ML and GP.
### Presentation Skills:
* Use page numbers, allows the audience to target the page
* Label the graph and organize the information in the slides
* Do not read the whole slides
### Groupwork notes
* Meet on Discord on Weekends. (around 6 hours)
* We went through the code template in Lab2 first and to see if we can get some hints from it.
* We decided to keep our preprocessing data in our last lab. ie. same chosen parameters.
* Since we are not allowed to use default algorithms in Deap (Tournament selection) we tried multiple different algorithms and decided to use SPEA2.
* Cited the regression evaluation function from lab2
* Helped Manas develop our genetic loop
* Our final result:
Best individual is `multiply(cos(add(subtract(Sex, Age), add(add(Sex, Sex), Parch))), Sex) with fitness (fpr, fnr) = (0.0, 0.37966101694915255) and pareto front AUC of about 0.125653`

![](https://github.gatech.edu/yma391/VIP-AAD/blob/master/aad%20week5.PNG)
* So we can see from the graph that GP does have a better result than ML


### Individual notes:
* Got a deep understanding of what we were doing in the lab
* Sat a Max boundary for our mate and mutate function, so that our trees would not go beyond height 17. This would avoid our algorithum work crazy. Here are the codes:
`toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))`
* Tested different selection, mutation, and evaluation functions. Here are the functions we selected:
`toolbox.register("evaluate", evalSymbReg, pset=pset)
toolbox.register("select", tools.selSPEA2)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)`
* My teammates made the slides and I went through the part I would present. Thanks, Rohan
> **Link:** https://docs.google.com/presentation/d/1E5DIPJOt7uBeqUeYklg6TE7X7PTdOsaFdUjTDCrttkU/edit?usp=sharing

| Task | Current Status | Date Assigned |  Suspense Date | Date Resolved |
|------|----------------|---------------|----------------|---------------|
|Team meeting1|complete|September 22nd, 2021|September 29th, 2021|September 25th, 2021|
|Team meeting2|complete|September 22nd, 2021|September 29th, 2021|September 26th, 2021|
|Record Notebook|complete|September 22nd, 2021|September 29th, 2021|September 27th, 2021|
|Complete Titanic MOGP Notebook|complete|September 22nd, 2021|September 29th, 2021|September 26nd, 2021|
|Make presentation slides|complete|September 22nd, 2021|September 29th, 2021|September 27th, 2021|


## Week 4: September 16th - September 22nd (2021)
### Lecture Overviews
* Split into different Bootcamp (Bootcamp 2)
* Introduced Kaggle Competitions
* Talked about the Titanic disaster problem
* Introduced Scikit (scikit-learn.org) for machine learning prediction model
* The results are: 1 = survived, 0 = did not survive
* Talked about the difference between test.csv and train.csv
* Introduced pandas and NumPy in the reference notes
* Requested codominant results in groups final CSV files
### Groupwork notes
* Sat up discord channel and met online during the weekend
* Meeting on Saturday, 9/18 and Sunday, 9/19
* We went through ideas we have and send a conclusion to the Discord channel
* The meeting records are here:https://docs.google.com/document/d/1WVhgmRNwyJxAAaGPhp5YT6-aHzeGc_kS8ewx94U4Myw/edit
* Dropped the parameters that we thought are not important
* Recorded codominant results in our CSV files
* Chosen model: SVC_sigmoid

* Removed some parameters that are not useful to our model:
1. Name

2.PassengerID

3.Ticket Number

4.Fare

* The parameters we chose:
1. Pclass
2. Sex
3. Parch
4. Embark
5. SibSp (Because families would stay together)
* While we were trying different models on SciKit learn, many of the models would not produce codominant results. Therefore, we created a NaN_map to fill in missing Age and Embarked values, and modify our parameters. Some model requires special parameters in the constructor and we searched the documents. 
* Our final results are:
_      Aditya = AdaBoostClassifier. FP = 32, FN = 21.
    *  Rohan = DecisionTreeClassifier (min_samples_leaf=30). FP = 9, FN = 45. 
    *  Manas = RandomForestClassifier (parameters above). FP = 18, FN = 29. 
    *  Adithya =  MLP. FP = 26, FN = 26.
    *  Yisu = SVM (used svm.SVC, sigmoid kernel). FP = 0, FN = 104. _

![](https://github.gatech.edu/yma391/VIP-AAD/blob/master/aad%20week4.png)

### Individual notes:
* Although the whole team decided to keep consistency for the parameters we choose, I find that Cabin and Age could also significantly influence the final results and in this case, the Embarked parameter may not be important, because the cabin position may not be determined by the port that a passenger is from.
* After reviewing Scikit Documentation (scikit-learn.org), I tried different functions to come up with codominant results with my teammates. But most of them fail. I asked Manas for help and finally, I used svm.SVC, sigmoid = kernel (SVM) to fit our codominant results. The result is kind of extreme because I got FP=0, FN=104, which means all of them are false negative.The SciKit documentation tells me which parameters each constructor takes in. 
* This might be a manipulated result but we find that in order to achieve the codominance, we would have to intentionally modify the hyperparameters we have for some model either to lower FNR or raise FPR. Thanks to Manas's discovery.
* For example, in my chosen model the kernel type to be used in the SVM algorithm would cause huge disagreement with other teammates' results.
Below are the resources I found on the internet for different  kernel parameter:
`
 Linear Kernel: K(X,Y)=XTY
 Polynomial kernel: K(X,Y)=(γ⋅XTY+r)d,γ>0
 Radial basis function (RBF) Kernel: K(X,Y)=exp(∥X−Y∥2/2σ2) which in simple form can be written as exp(−γ⋅∥X−Y∥2),γ>0
 Sigmoid Kernel: K(X,Y)=tanh(γ⋅XTY+r) which is similar to the sigmoid function in logistic regression.
`
* Reviewed 5 predictions.csv files and helped add Pareto Optimal Frontier cells in our notebook


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





