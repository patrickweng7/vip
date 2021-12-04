# Team Member
<b>Team Member:</b> Steven Leone <br>
<b> Major: </b> Computer Science <br>
<b>Email:  </b>sleone6@gatech.edu <br>
<b>Cell Phone:</b> (412)-378-7253 <br>
<b>Interests:</b> Machine Learning, Natural Language Processing, Software Engineering, Algorithms <br>
<b>Sub Team:</b> NLP <br>
<b>Sub Team Teammates:</b> Karthik Subramanian, Devan Moses, Kevin Zheng, Shiyi Wang, George Ye, Rishit Ahuja


Original Notebook can be found at: https://github.gatech.edu/emade/emade/wiki/Notebook-Steven-Anthony-Leone

## Week 16, Dec 6th

### General Meeting Notes



### Sub Team Meeting Notes


### Action Items
Task | Current Status | Date Assigned | Date Resolved | Date Due |
--- | --- | --- | --- |--- |
,,,| Complete | 11/15/2021 | //2021 | //2021 |

This week: Runs, Final Presentation

## Week 15, Nov 29th

This week: codefreezed version, getting in runs in

### General Meeting Notes



### Sub Team Meeting Notes


### Action Items
Task | Current Status | Date Assigned | Date Resolved | Date Due |
--- | --- | --- | --- |--- |
Complete EMADE Runs on Squad| Complete | 11/29/2021 | 12/04/2021 | 12/06/2021 |
Create a branch for code freezed EMADE | Complete | 12/01/2021 | 12/01/2021 | 12/06/2021
Help other Team Members get set up on PACE | In Progress | 11/29/2021 | ... | 12/06/2021



## Week 14, Nov 22nd



### General Meeting Notes



### Sub Team Meeting Notes


### Action Items
Task | Current Status | Date Assigned | Date Resolved | Date Due |
--- | --- | --- | --- |--- |
Add Regression to NNLearner2 | Complete | 11/22/2021 | 11/23/2021 | 11/29/2021 |
Fix weird incorrect accuracy issue in eval_methods | Complete | 11/22/2021 | 11/29/2021 | 11/29/2021 |
Fix train/test data split in NNLearner2 | Complete | 11/25/2021 | 11/28/2021 | 11/29/2021



# Fall 2021

## Week 13, Nov 15th
### General Meeting Notes
* Output layer problem this week, decided with Karthik and Rishit the previous week that it was too much for this semester


### Sub Team Meeting Notes


### Action Items
Task | Current Status | Date Assigned | Date Resolved | Date Due |
--- | --- | --- | --- |--- |
Retask  Members from output layer team| Complete | 11/15/2021 | 11/17/2021 | 11/17/2021 |
Further Debug Primitives with Standalone Tree Evaluator | Complete | 11/15/2021 | 11/21/2021 | 11/21/2021 |
Meet w/ Devan on double NNLearners | Complete | 11/17/2021 | 11/21/2021| 11/21/2021 |
Have code ready for Code Freeze | Complete | 11/15/2021 | 11/29/2021 | 11/22/2021 -> Extended to 11/29/2021|


### Hackathon
* At the Hackathon, I worked with Devan to debug NNLearner2 and get it working


## Week 12, Nov 8th
### General Meeting Notes
* NLP: 
* * Having onboarded our new members and laid out what we need to accomplish before the end of the semester, we began tasking today.
* * We used the trello board to layout tasks.
* * I began leading the integration team to workout the bugs from the big merge that remained, in addition to getting the primitives to work, and debugging them
* * Karthic began leading a team to explore our issue with the output layer
* * Devan began leading a team to debug NNLearner2
* * Kevan would look at word embeddings

### Action Items
Task | Current Status | Date Assigned | Date Resolved | Date Due |
--- | --- | --- | --- |--- |
Implement new Primitives into EMADE | Complete | 11/08/2021 | 11/09/2021 | 11/10/2021 |
Debug Persistent issue with Merge | Complete | 11/08/2021 | 11/14/2021| 11/16/2021 |
Debug Primitives | In Progress | 11/08/2021 | Moved til next week | 11/16/2021 |


## Week 11, Nov 1st
### General Meeting Notes
* We received our New Members this week and began onboarding them.
* I gave them a brief intro to our project and gathered information about their background in order to decide what to present on Wednesday.
* NLP:
* * I updated the whole team on our progress. We believed our primitives should work at this point, and we listed out the remaining tasks left before we could get results.
* Modularity: 
* * They were reassessing and adjusting goals and timelines for the semester
* * The ARL group is still going to be doing runs and experimenting
* NAS:
* * Noticed a bug with their ADF's.
* * Already working on final presentation material.
* Image Processing
* * At this point, they have been divided into teams working on Selection Methods, Mating/Mutations, Hyperfeatures, and some general infrastructure work.
* * Some of their members were re assigned tasks this week.
* * Having some errors on PACE.

### Sub Team Meeting Notes
* At this sub team meeting, it was the first time we had the whole team with the new members together for a whole hour.
* First, we introduced ourselves to each other.
* We divided our meeting into two groups. Our returning members went into a breakout room to continue working on the Bidirectional Attention Layer, while I gave a presentation on NNLearners and Deep Learning. The presentations, which I edited from last year, are linked in my weekly work below.
* Then I switched with Devan, and went to work with the returning members while Devan taught our new members about NLP and QA specific topics.

### Action Items
Task | Current Status | Date Assigned | Date Resolved | Date Due |
--- | --- | --- | --- |--- |
Complete New Primitive Development | Complete | 11/01/2021 | 10/27/2021 | 11/08/2021 |
Prepare Presentations for New Members | Complete | 11/01/2021 | 11/02/2021 | 11/03/2021 |
Debug New Primitives with Standalone Tree Evaluator | Complete | 11/01/2021 | 11/08/2021 | 11/08/2021 |

### Preparing Presentations for New Members
* I modified last year's PowerPoints to inform the new members about deep learning and NLP.

### Debugging Primitives with Standalone Tree Evaluator
* I began using Standalone Tree Evaluator this week to debug the primitives we'd made.

## Week 10, Oct 25th

### Presentation Notes
* NLP
* * We presented our work so far: https://docs.google.com/presentation/d/1GviS4whmKxNpbxn2cMQgcUthRxp1hsmu_NLfDkY00b4/edit?usp=sharing
* Bootcamp 1
* * Learning and Implementing ML, MOGP, and EMADE for Titanic Dataset from Kaggle
* * First, pre-processing data
* * Replacing Null Values, One-Hot Encoding
* * SVM Classifier, Gradient Descent, Neural Network, Random Forest Classifier, Gaussian Process Classifier
* * Genetic Programming worked better than the ML for this group
* * EMADE -> doubled headless chicken rate
* * Biggest setback was messing up dataset
* Neural Architecture Search
* * Creates Neural Networks in EMADE with the NNLearner
* * Previous Semesters had individuals that could barely improve off of seeded individuals
* * Dimensionality Errors (no rule enforced on connecting layers)
* * Worked around Time Stopping
* * Preprocessing: text tokenization, one hot encoding for multi class target data
* * CoDEEPNeat, latest and most effective Neural Architecture Search
* * Some individuals are taking too long to train 
* * New analysis methods
* * Dimensionality issues between layers: they're still working on this, 
* * Throwing away non NNLearners: used strongly typed GP, made a new type of EMADE Data Pair that requires an NNLearner to be used
* Bootcamp 2
* * Data Preprocessing Procedure: dropped name, passengerID, ticket number, and far
* * Mapped alphabetical values of 'sex' and 'embarked'
* * MOGP: Used False positive and negative rates, minimizing these two objectives
* * EMADE: had a difficult installation process
* * Created a virtual conda environment, used python 3.7
* * Adaboostlearner was their best type of individual
* Image Processing 
* * New this semester
* * Wanted a narrower scope
* * Dataset Preparation: Image Resizing, normalization, horizontal flipping, newer augmentations
* * Augmented 1000 out of 5000
* * Having suspicious baseline results, including an incredibly low AUC
* * Implemented NSGA III to use
* * Pre defines set of reference points to maintain diversity among its solutions
* * In theory, should outperform NSGA II
* * Added two new mating and mutation methods based around semantic, Geometric Semantic Crossover and Geometric Semantic Mutation
* * Added "Hyper-features" for Image Processing and GP
* Bootcamp 3
* * Worked on titanic dataset
* * used one hot encoding for genders
* * dropped passengerID, ticket, and cabin
* * Added a gender_embarked column
* * started with a variety of machine learning models popular for classification problems, including svm, rf, logistic regression, and MLP neural networks (multilayer perceptrons)
* * GP evaluation function used False positive and negative rate
* * EMADE, had to modify sel_nsga2 to achieve selTournamentDCD()'s requirement of having an individuals array divisible by 4 (should be fixed by DEAP == 1.2.2... we normally run into this issue once a semester if I recall correctly).
* * AdaBoostLearner was the individual with the most appearances in their results
* * Takeaways: Important to use trial and error, it is crucial to connect worker nodes
* Stocks
* * Overarching Objectives: How to use EMADE for regression on time series data, and how to use emade to optimize market trading algorithms?
* * Idea: Model other papers and use technical analysis to predict future trigger points for the stock
* * Relative performance of primitives based on CDF metric shows that some primitives are working great
* * Objective functions include profit percentage, average profit per transaction, pdf of profit, and variance of profit per transaction
* * Designing new experiments based on AAPL and other stocks
* Bootcamp 4
* * Data preprocessing: dropped name and cabin.
* * One hot encoded for embarked feature
* * replaced null values
* * Then moved on to MOGP Evolution. Tried one point crossover but the results weren't great.
* * Revised MOGP by replacing selection with select
* * AUC for EMADE was .26380
* * Pareto Front for gen 20 had 314 valid individuals
* * Number of valid individuals increased with generations
* * Had more diverse individuals with MOGP
* * EMADE worked well with large primitive sets for them
* Modularity
* * Trying to introduce ARL's, or Adaptive Representation through Learning
* * A way to introduce reusability in EMADE
* * Evaluation -> Selection -> Genetic Operations -> ARLS -> Evaluation
* * ARLS are useful because they might improve a population's overall fitness
* * Should allow a population to converge faster
* * This semester, continuing their work on increasing the complexity of ARLS via increasing tree depth
* * Goals include allowing search functions to consider larger subtrees depth to increase the potency of ARLs
* * Improve ARL candidate selection via a new weighting function
* * Fixed a lot of bugs this semester, like individuals only having one of each ARL
* * Fixed Arity of parent nodes not being updated

### Sub Team Meeting Notes
* We dedicated this week to getting the necessary primitives to get results
* We decided that basing a model off of the BiDaf model was the best route to go.
* Our embedding layers currently in EMADE would suffice. We still needed a Bidirectional Attention Layer, a Modeling layer, and an Output layer.
* We assigned ourselves into three groups to implement these remaining layers. 
* Bidirectional Attention Layer: This will be the hardest layer to implement. We assigned myself, Karthik, Devan, and Rishit to implement this primitive.
* Modeling Layer: This is essentially just an LSTM layer. We assigned George to this.
* Output layer: This will take some effort, so we assigned Kevin and Shiyi to this task.

### Action Items
Task | Current Status | Date Assigned | Date Resolved | Date Due |
--- | --- | --- | --- |--- |
Assign Primitive Development | Complete | 10/25/2021 | 10/27/2021 | 11/01/2021 |
Email Dr. Zutty about 2 Data Pair NNLearners | Complete | 10/25/2021 | 10/30/2021 | 11/01/2021 |
Implement Bidirectional Attention Layer | Complete | 10/25/2021 | 11/07/2021 | 11/08/2021 |


### Implementing Bidirectional Attention Layer: Part 1
* Devan and I met on Friday to work on the Bidirectional Attention Layer this week
* As neither of has had experience implementing primitives prior to this, we first spent a lot of time reading through neural_network_methods.py to discover how to implement a new primitive.
* I discovered that NNLearner uses a new class defined in the file, called a LayerList, to keep track of the layers. Then, the NNLearner function iterates through these layers to build the deep neural network.
* So, if we wanted to add a layer, we would have to create a subclass of Keras.layers.Layer, and override the build() and call() functions.
* While I was doing this, Devan found a template from someone else who implemented BiDAF. It wasn't split into layers, but we could use it as a reference for learning keras and TensorFlow.
* After a little over an hour, and with the skeleton of the Bidirectional Attention Layer done, we finished up work for the time being, until Sunday.

### Implementing Bidirectional Attention Layer: Part 2
* I met with Kevin, Rishit, and Karthik on Sunday this week to work on the Bidirectional Attention Layer.
* (TODO: Finish in details of meeting)
* (TODO: Link result code as of this meeting)



## Week 9, Oct 18th

### General Meeting Notes:
* At this week, we had one week until our presentation
* NLP:
* * We talked with Dr. Zutty about our issue with standalone tree evaluator. We were able to resolve it. As it turns out, the EMADE-304 branch was only updated for base EMADE, and not standalone tree evaluator. So, we would have to move changes from 1 over to the other in order to use standalone tree evaluator.
* Modularity:
* * Finished fixing ARL implementation bugs
* * Found a logic bug with some ARL primitives that needs fixed, deals with "tuple index out of range"
* * Have many members from the stocks team now, are trying to move forward with that side of things
* NAS:
* * Prepping for midterm presentation and doing runs
* * Divided up slides
* * Looking into how to push all of their code changes into a single branch
* Image Processing: 
* * Made their input schema compatible with PACE
* * tested NSGA-III with their new implementation
* * Made some runs and bug fixes, but won't realistically be able to move them in until after their presentation

### Sub Team Meeting Notes
* This week was dedicated to debugging merge changes and preparing for our midpoint presentation
* Most of this meeting was dedicated towards making our midpoint presentation
* I created a slide deck, and we talked about what we wanted to include in it.
* The slides can be found here: https://docs.google.com/presentation/d/1GviS4whmKxNpbxn2cMQgcUthRxp1hsmu_NLfDkY00b4/edit?usp=sharing
* We would rehearse on Sunday.

### Action Items
Task | Current Status | Date Assigned | Date Resolved | Date Due |
--- | --- | --- | --- |--- |
Finish Eval Method Implementation | Complete | 10/18/2021 | 10/20/2021 | 10/25/2021 |
Debug Merge Bugs | Complete | 10/18/2021 | 10/21/2021 | 10/25/2021 |
Run EMADE on new branch with no errors | Complete | 10/18/2021 | 10/21/2021 | 10/25/2021 |
Make presentation slides | Complete | 10/18/2021 | 10/20/2021 | 10/25/2021 |
Rehearse Presentation | Complete | 10/18/2021 | 10/20/2021 | 10/25/2021 |



## Week 8, Oct 11th
### General Meeting:
* No General Meeting this week

### Sub Team Meeting
* At our Sub Team meeting, we began looking at the EMADE-304 branch and deciding how best to merge it over.
* We planned to use the Hackathon to finish merging EMADE-304 over.
* We would look over the code over the next 2 days, and then have a meeting on Friday to discuss our thoughts.

### Friday Meeting
* I put together a list of trivial vs non-trivial merge conflicts: https://docs.google.com/document/d/1B-0uHdawDfCY-5BLZQUiuz-dleC9MDwqVhYU6WmliRc/edit?usp=sharing
# We discussed it, moved files around, and decided to use it as a basis for which files to look at first on Saturday.

### Hackathon
* Karthik, Kevin and I met at the Hackathon on Saturday to work on the merges, while our Literature Review Team kept looking at papers for inspiration with different primitives and state of the art models.
* We used my document to keep track of which files we had to change still, and which we had already worked on.
* The three of us looked through each file together to determine which ones to edit.
* Most changes ended up being trivial. GP_framework_helper.py, general_methods.py, and data.py had the most non-trivial methods.
* Our branch is now up on my fork of EMADE: https://github.gatech.edu/sleone6/emade/tree/EMADE-304-allow-cachev2-to-consume-aligned-datapairs
* We merged the nn-vip branch into EMADE-304


### Action Items
Task | Current Status | Date Assigned | Date Resolved | Date Due |
--- | --- | --- | --- |--- |
Look over Merge Conflicts | Complete | 10/13/2021 | 10/14/2021 | 10/15/2021 |
Resolve Merge Conflicts | Complete | 10/13/2021 | 10/16/2021 | 10/16/2021 |

## Week 7, Oct 4th
### General Meeting Notes
* Stocks:
* * The stocks team was disbanded.
* Image Processing:
* * Worked on implementation of NSGA-III, and new mutation methods.
* Modularity: 
* * Will stick with Google Collab instead of PACE-ICE.
* * Fixed bug with incorrect arities.
* * Will focus on unit testing and code reviews to avoid the same or similar errors from re-occurring.
* NAS: 
* * Informed the team members about the evolutionary process in EMADE.py. Went through the master algorithm loop in EMADE. Discussed mutations and other functions needed.
* NLP: 
* * We discussed our progress on loading in the QA dataset to EMADE, and told the team about our blocker with multiple data to be represented in the tree. Dr. Zutty informed us that there has been some progress on this front outside of the VIP; code already existed for multiple EmadeDataPairs. We further elaborated on this in our breakout meeting, discovering that the dataPairArray is a list of lists of EmadeDataPairs. 

## Sub Team Meeting Notes
* As we are still blocked on merging in the new code to work with multiple EMADE Data Pairs, I took time in the sub team meeting to present on how to test changes to EMADE on PACE-ICE. This involved using scp to transfer files, having an equivalent local copy setup, and running bash reinstall.ish to reinstall GPFramework while in the anaconda environment.
* Next, we turned our attention to the tasks that will come forward.
* As a way to get our two new members from the stocks team familiar with QA and NNLearners, I assigned them, and Shiyi, with coming up with some seeded individuals in a tree structure that we could test in standalone_tree_evaluator.py, looking at BiDAF and the SQUAD leaderboard as examples. This would also help us figure out what primitives to write to make QA work.
* The rest of us would be on the team to get the changed EMADE, with support for multiple data pairs, to work.

### Action Items
Task | Current Status | Date Assigned | Date Resolved | Date Due |
--- | --- | --- | --- |--- |
Make an EMADE Branch with new CacheV2 and NNLearner functionality | Complete | 10/03/2021 | 10/10/2021 | 10/10/2021 |

## Week 6, Sep 27th
### General Meeting Notes
* Stocks: 
* * The stocks team was dissolved this week
* Image Processing:
* * Working on NSGA-III implementation, Hyper-feature & Primitive Packaging, and more mating and mutation functions.
* Modularity: 
* * Identified a bug that was causing crashes. 
* * Working through getting everyone setup to do PACE-ICE runs
* NAS:
* * Looking into writing to disk
* * Discussed introducing weight sharing to the existing NNLearner evolution process in EMADE
* * Completed the Run-Resume feature
* NLP:
* * We presented our plan to implement QA systems into EMADE.
* * We received feedback from Anish, and decided to just overload NNLearner with two EMADE data pairs.

### Sub Team Meeting Notes
* We received two new members, as the stocks team was dissolved
* We focused on our tasks, and assigned everyone to start trying to get the SQUAD dataset loaded in.
* I made a quick presentation on what we're trying to do this semester, and referred our new members to my notebook to look at the current state of research.

### Action Items
Task | Current Status | Date Assigned | Date Resolved | Date Due |
--- | --- | --- | --- |--- |
Create new XML and data load functions for QA in EMADE | Complete | 09/27/2021 | 10/03/2021 | 10/03/2021 |


### Creating new XML and data load functions for QA in EMADE

* To implement this functionality, I did another deep dive into EMADE, this time into some of the core functionality of creating the objects that would eventually be used for the evolutionary process and to evaluate individuals.
* To start, I added a method into data.py titled "load_qa_textdata_from_file". This function is similar to "load_textdata_from_file", but it accounts for two pieces of data that need loaded and represented in the individuals' trees separately, the context and query. This design isn't perfect; it isn't very re-usable for other projects that will use multiple data pairs. We can improve on this design as the semester progresses.
* The code I wrote for "load_qa_textdata_from_file" is shown below. The entire modified file "data.py" is accessible here: https://drive.google.com/file/d/10JjfNUzw98tY6Y_V4I_0IKjfos0wWdOO/view?usp=sharing
* <img width="826" alt="Screen Shot 2021-10-06 at 9 17 08 PM" src="https://github.gatech.edu/storage/user/27405/files/d4c43798-803c-49ca-9eb2-847f510e4cb4">
* Next, I began testing where the code would be ran from, in EMADE.py. It directly takes in a load_function from the xml file based off of the type of data. So, I set the type of data in a new xml file called input_squad.xml, based off of input_amzn.xml, to be "qatextdata", and had this type of data map to the "load_qa_textdata_from_file" function I made. The code I changed is visible below, and the entire modified file is accessible here: https://drive.google.com/file/d/11fuaMtMTbjcdmpRyq8LRHsEd6TZGSxUE/view?usp=sharing
* <img width="1022" alt="Screen Shot 2021-10-06 at 9 18 14 PM" src="https://github.gatech.edu/storage/user/27405/files/d1634596-7da5-4dd8-a46a-c7ba5d79b1bb">
* After this, however, when testing this with standalone_tree_evaluator.py, I did run into an architectural problem after this. EMADE only stores one list of EMADE Data pairs for each fold, stored as the "dataPairArray". We would have to modify many other parts if we were to add another array. This is an issue we will have to discuss at the coming meeting.

## Week 5, Sep 20th
### General Meeting Notes
* Stocks: 
* * Still determining the semester goal, whether it will be Stock generalization or fundamental analysis
* * Found a couple more interesting papers
* Image Processing: 
* * Looking into NSGA-III and Lexicase as selection models.
* * Will compare performance with the ChexNet paper
* Modularity: 
* * Achieved ARL's with depth > 2 for the first time, checking off a big semester goal
* * Looking into more improvement now.
* NAS:
* * Debugged EMADE issues
* * Members are responsible for finding and debugging work once EMADE is setup on PACE-ICE
* NLP: 
* * We informed the team of our progress, including selecting objectives. We decided that previous models using num params vs an accuracy objective weren't quite diverse enough. Adding other objectives will allow a better tradeoff space that will lead to diverse models.
* * Having decided on our objectives and analyzed the dataset, we then had to tackle the building blocks of a QA system. The biggest challenge with them is how there are two inputs, the context and the query.
* * I would research the current state of the nn-vip EMADE branch and present a best strategy for implementing a QA system.

### Sub Team Meeting
* I presented to the sub team the information I compiled on NNLearners, shown in the section underneath Action Items, and how we could use them for QA systems.
* We discussed several strategies for the layers we needed. 

### Action Items
Task | Current Status | Date Assigned | Date Resolved | Date Due |
--- | --- | --- | --- |--- |
Research NNLearners and how to implement QA | Complete | 09/20/2021 | 09/22/2021 | 09/22/2021 |
Research creating a new type of data pair | Complete | 09/20/2021 | 09/27/2021  | 10/03/2021 |


### Researching NNLearners and how to implement QA
* My assignment was to research the current state of NNLearners and see how we could add primitives to make the problem solvable
* Having never actually coded for the branch before, I started by looking at NNLearners in neural_network_methods.py.
* The NNLearner was setup so that each primitive was a layer from a Keras Neural Network. A lot of primitives we would need were already there. The output layer, attention layer, and embedding layers existed.
* The initial input for an NNLearner was a data_pair, referred to in the tree structure as ARG0. I figured out that the type is an EMADEDataPair by looking at the inputs and comparing methods called on it.
* From this information, I formed strategies for our two obstacles:
** To deal with the requirement of a different way of obtaining outputs from an outputted probability matrix, we could make a new data pair type, similar to the types 'textdata' and 'imagedata' that have already been implemented in EMADE. Then, at the end of the NNLearner code, we could implement an if statement to obtain the proper start and endpoints. 
** To deal with the requirement of two inputs per data point, the context and the query, we could make primitives that specifically return an embedding of exactly one of the inputs. For example, one could be named "ContextEmbeddingLayer" and another could be "QueryEmbeddingLayer".
* Analyzing these solutions from the mindset of attempting to increase and maintain diversity in individuals, I think these solutions will work well. As opposed to the original thought process of constraining NNLearners to always have one type of input or output, this will maintain the normal constraints of mating individuals, which should still allow for any individuals to be mated, increasing our diversity.

## Week 4, Sep 13th

### Self Evaluation Rubric
<img width="962" alt="Screen Shot 2021-09-13 at 7 11 24 PM" src="https://github.gatech.edu/storage/user/27405/files/6e678180-14c6-11ec-811b-bb9acf9a8258">

### General Meeting
* Stocks: 
* * Conducting a literature review to find new ideas of what to implement.
* * Their team also found a good resource on genetic programming generated trading.
* Image Processing: 
* * Defined goals and tasked themselves.
* * Focusing on multi-label image classification
* * Image classification or object detection are their overarching goals.
* Modularity:
* * Still coming up with goals for the semester.
* * Potential areas of improvement include new models, selection methods, ARL Database storage, and more depth in ARL's.
* Neural Architecture Search:
* * Presented information about neural architecture search to new members.
* * Came up with 6 different potential improvements, including triviality detection
* NLP: 
* * We updated the team on our progress.
* * 3/5 of our members have gotten EMADE on PACE setup, and we're moving into a literature review of Question Answering Systems.

### Sub Team Meeting
* We gave our updates on the assigned tasks so far
* Everyone has been able to get EMADE to run the Amazon dataset on PACE.
* We then went over the objectives we needed to figure out to get EMADE to work.
* We decided our objectives would be F1 and number of parameters, to strike a balance between a match/accuracy and complexity. My task was to look into F1 for QA systems and find out how they worked (we ruled out EM as it was similar to F1, but harder to train with).
* Karthik created a Google Collab notebook to begin looking at the dataset.
* We would familiarize ourselves with the different layers of a QA system before planning what primitives to make on Monday.

### Action Items
Task | Current Status | Date Assigned | Date Resolved | Date Due |
--- | --- | --- | --- |--- |
Research and Implement QA F1 | Complete | 09/15/2021 | 09/19/2021 | 09/20/2021 |
Familiarize self with Question Answering Systems | Complete | 09/15/2021 | 09/19/2021  | 09/20/2021 |
Self Evaluation Rubric | Complete | 09/20/2021 | 09/20/2021 | 09/20/2021 |

### F1 Research
* As opposed to other models, I discovered that F1 is used to score each answer provided individually. Then, the F1 score of each data point is averaged.
* There are two answers, which are the predicted answer and the ground truth answer. Words in the predicted answers are true positives if they appear in the ground truth answer, false positives if they don't. We have false negatives if words from ground truth don't appear in the data as well.
* I began to write code to write this as an objective function for EMADE, basing it off of the current accuracy objective.
* I finished writing the modified F1, based on the official SQUAD 2.0 evaluation dataset. The code can be found at this link: https://drive.google.com/file/d/1KNRPpFV2RZLSYcQArlSE341eNGa-sRVl/view?usp=sharing 
<img width="839" alt="Screen Shot 2021-09-19 at 1 31 24 PM" src="https://github.gatech.edu/storage/user/27405/files/01aa3500-1a3a-11ec-80fa-7324ee23d348">

### Question Answering Systems Research / Refresher
* A tasked we assigned to all members of the team was to get refreshed on Question Answering Systems, or learn about them if no previous knowledge existed. 
* I reread chapters and papers on QA systems, and linked some to my sub team members that I thought would be useful.
* In general, QA systems have at least 3 layers: the embedding layer, the attention layer, and the output layer. These layers are all required for the following reasons:
** Embedding layer: without this, there is just text. We cannot obtain numbers to work with.
** Attention layer: QA systems have two inputs. We need to have a way for the vector representing the final output to be aware of both the context and the query, and attention layers are currently the best way to do this.
** Output layer: we need to obtain our output. We must know which word in the Context is our answer.
* In general, the output provided includes a list of length 2N, where N is the number of words in the context. The first N values are the probabilities that the ith word is the start of the answer. The last N values are the probabilities that the ith word is the end of the answer.
* For example, if our context was "The Titanic sank in 1912" and we had the output vector [0, 0, 0, 1, 0, 0, 0, 0, 0, 1], then our answer would be "in 1912". 
* I linked a paper about BiDAF, a QA model that is relatively easy to understand compared to some other state of the art models: https://arxiv.org/pdf/1611.01603.pdf
* This is a diagram with BiDAF from the paper, which shows the 3 layers I described earlier (Embedding, Attention, Output) with a few extra ones:
* <img width="1008" alt="Screen Shot 2021-09-26 at 9 34 25 PM" src="https://github.gatech.edu/storage/user/27405/files/a1bbfe80-1f11-11ec-9e9e-dcd0d2610184">



## Week 3, Sep 6th
* There was no General Meeting this week
* We decided on Wednesday at 2 pm for our Weekly Sub Team Meetings

### Sub Team Meeting Notes
* We organized our first sub team meeting of the year
* We discussed QA systems, and I gave a brief presentation on how they work, producing probabilities for where the answer in a given paragraph starts and stops
* We then made a brief list of steps to explore our problem with Question Answering Systems
* We assigned tasks. For Monday, everyone was to setup EMADE on PACE and start runs with the Amazon dataset.
* We would 


### Action Items
Task | Current Status | Date Assigned | Date Resolved | Date Due |
--- | --- | --- | --- |--- |
Run EMADE on Amazon Dataset | Complete | 09/07/2021 | 09/12/2021 | 09/13/2021 |
Look into State of the Art Question Answering Systems | Complete | 09/07/2021 | 09/12/2021  | 09/13/2021 |


### Checking EMADE
* The first action I took was to ensure we could still run EMADE on PACE
* First, I ensured I still had PACE access by logging in with "ssh sleone6@pace-ice.pace.gatech.edu". This worked as expected.
* From memory and referring to https://github.gatech.edu/emade/emade/wiki/Guide-to-Using-PACE-ICE as a guide, I then worked my way through, ensuring the mysql database still ran upon submitting a job on PACE.
* I then tested seeding, running the seeding file with the 10 NNLearners we used for seeding the previous year. This also functioned properly.
* Finally, I tested running EMADE as a submitted job on PACE. I waited until there were results after 3 Generations. The Pareto front had slightly different results than the seeded values, meaning that EMADE was working.

### Literature Review for Question Answering Systems
* I found a paper with a recent Question Answering System focus, https://aclanthology.org/P17-1018.pdf . It achieved scores in the 80's on the SQuAD dataset, using gated-self matching networks. 
* As the linked paper describes, gated-self matching networks work as follows: the query and the context are embedded separately, then processed by a bidirectional RNN. Then, the outputs are combined through gated attention layers. Another attention layer is then used, which applies this output to itself in this attention layer. The paper diagrams this structure with the figure below:
* <img width="797" alt="Screen Shot 2021-09-26 at 9 05 33 PM" src="https://github.gatech.edu/storage/user/27405/files/3113e280-1f0f-11ec-95d2-1e7430ed8cf6">
* I don't believe this paper is useful in terms of finding inspiration for new primitives. As opposed to a new mechanism that could be used, the primary innovation in this paper is the structure of the QA system. The individual units, like GRU and Attention layers, are already implemented. The structure, or how these building blocks will fit together, is for EMADE to decide by evaluating generation after generation of individuals and looking at the ones on the Pareto Front for inspiration.
* However, I do believe this paper is helpful for our task in that it is a state of the art model for the SQUAD dataset. We can use it as inspiration to make a seeded individual, by writing this out in a tree structure. 


## Week 2, August 30th
### General Meeting Notes
* During the general meeting, I informed the whole team of ideas discussed in our brainstorm meeting. Devan also suggested added more primitives for more than embeddings.
* The sub team for NLP was officially formed with members decided.

### Action Items
Task | Current Status | Date Assigned | Date Resolved | Date Due |
--- | --- | --- | --- |--- |
Setup Weekly Sub Team Meetings | Complete | 08/30/2021 | 09/07/2021 | 09/13/2021 |
Finalize Semester Goals | Complete | 08/30/2021 | 09/07/2021  | 09/13/2021 |
Start Progress Towards Semester Goals | Complete | 08/30/2021 | 09/12/2021 | 09/13/2021 |


## Week 1, Aug 23rd
### General Meeting Notes 
 * We discussed potential sub team ideas
 * A brainstorming channel in the slack was created for the NLP sub team.
 * I think exploring how AutoML can be used to explore more complex NLP problems, like machine translation, would be interesting, especially as machine translations require multiple qualities to score, making it a natural choice for a multi objective framework.
 * I found a few papers on using AutoML for machine translation. They each express how AutoML hasn't been used much for machine translation, and neither of them used multiple objectives (both used BLEU).
 * * https://ieeexplore.ieee.org/abstract/document/9095246/ talks about using NAS for machine translation, without an evolutionary system (using gradients)
 * * http://proceedings.mlr.press/v97/so19a.html talks about NAS for machine translation, with an evolutionary system
 * We held a brainstorming meeting. We decided that the issue of complexity was best left to the NAS team if we were splitting this semester.

### Action Items
Task | Current Status | Date Assigned | Date Resolved | Date Due |
--- | --- | --- | --- |--- |
Brainstorm Ideas for Meeting | Complete | 08/23/2021 | 08/27/2021 | 08/30/2021 | 
Setup/Conduct Meeting | Complete | 08/23/2021 | 08/27/2021 | 08/30/2021 | 