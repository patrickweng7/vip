# Aditi Prakash
Name: Aditi Prakash  

Email: aprakash86@gatech.edu

Bootcamp Subteam Members: Charlie Bauer - cbauer32@gatech.edu, Rayan Dabbagh - rdabbagh3@gatech.edu, Rayan Khoury - rkoury7@gatech.edu

Cell Phone: 704-794-3924  

Interests: Machine Learning, Data Science, Software Development, Dance, Reading

# Week 15: November 29th, 2021
## Overview
Finalize QA model parameters and EMADE run parameters, test standalone tree evaluator on NNLearner2, begin 8-hour trials with NNlearner2 as seeded individual. 

## Team Notes:
* Members are continuing to run 8-hour trials in EMADE seeded with NNLearner2, we have decided to use continuous_mse and number of parameters as our evolutionary objectives. 
* Our continous_mse metric is currently having an error wherein all of the values it returns for individuals’ continuous_mse are about half of what they should be based on their predictions of the start index in the answer. Steven was able to fix this by dividing the Monte Carlo objective values by 2 (the number of input DataPairs NNLearner2 takes in). 
* Kevin is continuing to fix up our layers by refactoring their logic to be a bit cleaner and more readable as well as adding meaningful comments to each layer to indicate the flow of tensors throughout each individual and their respective sizes. 

## Subteam Notes:
* We met in person at the CULC to work on seeded runs using our codefreezed feature/nnlearner2 branch. I learned how to seed runs using python3 src/GPFramework/seeding_from_file.py templates/input_squad.xml seeding_qa. My runs as of now are remaining stuck in the queue for a long time, which is likely due to the large number of runs that are simultaneously in the queue. Other errors I am experiencing include MySQL connection timeout issues, which are just a matter of retrying to resolve.  

## Individual Notes:
* Which individuals are evaluating, are they seeded individuals, which individuals are pareto option, are they seeded individuals, how fast is evaluation happening, issues with PACE queue, how far off is MSE, what is exact match rate, how might we change trial if it is an abstractive problem, why might we be getting these results 
* Since we focused heavily on creating valid primitives this semester, our layers are wrappers around Keras models and are quite limited to the functionality that the Keras API offers. In addition, our custom logic has caused some inefficiencies in the runtime for individuals, which might be avoided with a different approach to implementing our layers (ex. graph execution instead of eager execution). 
* NNLearner2 is also more or less straight out-of-the-box, so we are not entirely certain that it is best suited for our NLP problem. 
* One of our biggest action items for next semester is to work making our embedding methods (ex. GloVe and Bert) as optimal as possible. Moreover, since our problem is currently extractive and not abstractive, we heavily punish individuals for not predicting the exact answer string in the context, even though an actual human might be off by a word or two in their answer to a particular question. This model is quite far removed from other state-of-the-art NLP models and implementing this would allow our model to be more scalable and solve a wider range of problems. 
* We should also focus on our contextual embedding and ensure the output layer does not “undo” the work  done by the embedding and attention flow layers when we distill our probability matrices into a single prediction for the answer string. 
* Finally, we should have our models predict both the start and the end index of the answer in the context to ensure that they are able to learn the length of the answer string as opposed to just its start. 

**Action Items:**
| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| --- | ----------- | --- | ----------- |----------- |
| Finalize classification vs. regression problem based on Keras activation parameter | Done | 11/22/21 | 11/24/21  | 11/24/21 |
| Resolve standalone tree evaluator bugs with NNLearner2 | Done | 11/1/21 | 11/21/21  | 11/20/21 |
| Begin 8-hour runs with NNLearner2 as seeded individual and correct objectives for regression | In Progress | 11/26/21 | 11/29/21  | - |

# Week 14: November 22th, 2021
## Overview
Finalize QA model parameters and EMADE run parameters, test standalone tree evaluator on NNLearner2, begin 8-hour trials with NNlearner2 as seeded individual. 

## Team Notes:
* Currently, NNLearner2 predicts 0’s and 1’s for all samples in our training set, which we believe is due to some regression parameters not being set properly in create_representation() and model.fit(). Dr. Zutty suggested that it is a problem with how our output layer is using the sigmoid activation function (ie. squeezing all of its predictions to 0’s or 1’s).  
* From this documentation, it seems as though passing in None for the activation parameter should remove the sigmoid activation altogether: https://keras.io/api/layers/activations/, giving us the model’s actual predictions instead of a binary classification for answerable/unanswerable questions. 
* If this change produces our expected results, we will turn our current classification problem into a regression problem with the target being the start index of the word in the answer. Otherwise, we will continue with our classification problem. 
* We will confirm the nature of our problem by Wednesday, at which point we will start 8-hour trials to test NNLearner2’s performance. 

## Subteam Notes:
* No Wednesday meeting due to Thanksgiving break. Team members are starting 8-hour runs seeded with NNLearner2 as well as testing standalone_tree_evaluator.py on NNLearner2 with the changes Devan has pushed recently. 
* We are currently resolving a bug with our evaluation function wherein results are being divided by 2, which is likely due to the size of our DataPair being counted twice (since we are passing in 2 DataPairs to NNLearner2). Steven is working on a fix for this and is close to pushing the correct evaluation function, which we can use for trials going forward. 

## Individual Notes:
* Devan’s feature/nnlearner2 branch is our up-to-date, nearly codefreezed branch that we will use to run trials with NNLearner2. I am currently getting a MalformedNodeError when running standalone tree evaluator on the following individual: NNLearner2(ARG0,ARG1,OutputLayer(DenseLayer(10, defaultActivation, 10, LSTMLayer(16, defaultActivation, 0, trueBool, trueBool,EmbeddingLayer(100, ARG0, randomUniformWeights, InputLayer())))), 100, AdamOptimizer). Adding the following lines to my input_squad.xml file resolved the error (insert numinputs, regression flags pic here: )
* Running standalone tree evaluator on NNLearner2, I get the following results: 
(Analyze NNLearner2 results here). Once the statistics team finalizes the hypothesis for our experiments, I will know whether my results indicate that AutoML can improve QA systems or not. 
* Our goal until final presentations is to get as many runs as possible from each team member and increase the sample size of our trials such that our hypothesis testing can produce statistically significant results. As such, we will be able to evaluate the performance of individuals on our Pareto front as compared to the performance of our seeded NNLearner2 individual (our implementation of the BIDAF model in EMADE). 
* One goal I have for next semester is to write unit tests for each BIDAF layer we implemented this semester to ensure our seeded individual is scalable and to remove any hacky fixes with parameters we had to make this semester in the interest of time. 
* Having started an 8-hour run without seeding, I noticed that most individuals are evaluating to (inf, inf) (our current objectives are accuracy and number of parameters). I also notice that my runs are getting stuck on a certain generation (ex. Generation 24), which is an issue I resolved during bootcamp by uninstalling and reinstalling GPFramework. Doing this repeatedly is quite impractical given that we want full 8-hour runs without reuse/reseeding, so I am looking into a permanent fix for this (ex. reducing instances in input_squad.xml, reducing the dataset size, etc.). The individuals that are evaluating are mostly NNLearner2 individuals, which indicates that NNLearner2 outperforms other individuals consistently and is therefore being placed on our Pareto front for comparison with our seeded individual. 
* A WindowKaiser individual also repeatedly appears in the Pareto front upon starting new runs with reuse: https://www.mathworks.com/help/signal/ug/kaiser-window.html. This individual performs extremely poorly, but since it is the only individual other than NNLearner2 that evaluates, it is pushed to the Pareto front in each subsequent generation. This will likely impact our calculations of the AUC for our Pareto front versus the AUC of our seeded individuals when we analyze our results.
* I will continue to run standalone tree evaluator on NNLearner2 and start seeded runs with NNLearner2 and accuracy and num_parameters as objectives, but our team might switch to a metric more suited for regression (ex. MSE) during our team meeting on Monday. 

**Action Items:**
| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| --- | ----------- | --- | ----------- |----------- |
| Finalize classification vs. regression problem based on Keras activation parameter | Done | 11/22/21 | 11/24/21  | 11/24/21 |
| Resolve standalone tree evaluator bugs with NNLearner2 | Done | 11/1/21 | 11/21/21  | 11/20/21 |
| Begin 8-hour runs with NNLearner2 as seeded individual and correct objectives for regression | In Progress | 11/26/21 | 11/29/21  | - |


# Week 13: November 15th, 2021
## Overview
Debug standalone_tree_evaluator.py to work with NNLearner2 (2 DataPairs). 

## Team Notes:
* After researching Keras’ current capabilities, we realized that it does not currently support multi-output regression, which is a big limitation for us given that the output layer is a Keras library function with some custom logic added to it. Moreover, EMADE’s architecture directly passes individuals’ predictions to its evaluation functions for scoring, which means that we would have to make changes to EMADE’s master process in order for our individuals to predict indices and for our evaluation functions to take in strings. As such, we have modified our problem to be a regression problem, where our target value is just the start index of the answer in the string. Rishit had the idea of training the model twice, once to predict the start index of the answer and once to predict the end, but this would cause a few issues: our models would evolve differently in each evolutionary process, and we would lose information about the start index while predicting the end index. Moreover, our models could simply learn a pattern in the length of the answer strings as opposed to their semantic location in the vector space, which would defeat the purpose of predicting the end index altogether. 
* The output layer team has now dissolved, and I have been placed on Devan’s NNLearner2 team along with David, Jessi, and Rishit. 
* Our goal is to debug standalone_tree_evaluator.py to work on our NNLearner2 model (which takes in a context datapair and query datapair). 
* Currently, running standalone_tree_evaluator.py on NNLearner2 fails because NNLearner2 uses the load_environment() method in EMADE.py to set up a single individual for evaluation (the following line is from standalone_tree_eval.py: database, pset_info = load_environment(xml_file, train_file=train_file_name, test_file=test_file_name). 
* Our edits to EMADE.py (mainly changing the number of inputs to each individual in this line: self.pset = gp.PrimitiveSetTyped('MAIN', [EmadeDataPair]*datasetDict[0]['numberinput'], EmadeDataPair)) mean that we have to make changes to load_environment() so that EMADE's evaluate_individual() method still has the expected behavior when we call it on standalone tree evaluator with two EMADE datapairs as inputs. We have already ensured that the evaluation function is compatible with the changes to EMADE.py, and our goal is to do the same for load_environment().  
* We will all look into the codebase to determine if fixing load_environment() will be worthwhile or if we should proceed with full runs instead of standalone_tree_evaluator.py on NNLearner2. 

## Subteam Notes:
* Our team members were retasked in preparation for the hackathon on Saturday. 
* Returning members are working on debugging the BidirectionalAttentionLayer, which currently outputs a tensor with dimensions that are mismatched with the input tensor dimensions from other primitive layers. Keras layers taken in 3-dimensional input tensors with (batch_size, timestep, dimensions), but the BidirectionalAttentionLayer does not output a tensor of this size due to its lack of a batch size definition.
* New team members will continue to get set up on PACE in preparation for 8-hour runs with our seeding NNLearner2 individual. 
* I presented my analysis of whether it would be feasible to fix standalone_tree_evaluator.py tp work with NNLearner2 before the next full-team meeting. See individual notes below for my analysis. 

## Individual Notes:
* From what I can make of the current codebase, we define creator.create("Individual", list, fitness=creator.FitnessMin, pset=self.pset, age=0,  elapsed_time=0, retry_time=0,                        novelties = None, hash_val=None, **fitness_attr) in the setDatasets() method. However, we call create_representation() before we call setDatasets(), which means we haven't yet registered an Individual with two EMADE datapairs as arguments when we call create_representation(). 
* The following is the order of method calls in general_methods.py in load_environment() for standalone_tree_evaluator:
emade.create_representation(datasetDict, adfs=3, regression=regression)
emade.setObjectives(objectiveDict)
emade.setDatasets(datasetDict)
emade.setMemoryLimit(misc_dict['memoryLimit'])
emade.setCacheInfo(cache_dict)
emade.set_statistics(statisticsDict)
emade.buildClassifier()
* create_representation() is called on the emade instance prior to setDatasets(), and setDatasets() is where the datasetDict with two input datapairs is defined. This is what I believe to be the main discrepancy between load_environments() datapair ingestion logic and EMADE.py’s logic for normal runs. 
* I believe the following line should be modified in general_methods.py so that individuals can take in 2 datapairs in standalone_tree_evaluator: pset = gp.PrimitiveSetTyped('MAIN', [EmadeDataPair]*datasetDict[0]['numberinput'], EmadeDataPair), instead of gp.PrimitiveSetTyped('MAIN', [EmadeDataPair] EmadeDataPair), which is what it currently contains. 
* At the hackathon on Saturday, I continued debugging the Already Exists error I was having with Tensorflow and Keras installation in my PACE conda environment. A quick fix for this was to simply comment out all references to keras.backend (the package whose import command was throwing the error) in neural_network_methods.py and gp_framework_helper.py. After these changes, the same error was being thrown, and I noticed that the error message pointed to code at a line number that no longer existed in neural_network_methods.py. With this, I realized that EMADE was not actually running with my updated code, and I subsequently ran bash reinstall.sh and was able to resolve all errors at that point. I’ve learned from this that a simple bash reinstall.sh can prevent several errors related to package installation, which is something I will definitely keep in mind going forward. 
* We will start setting up 8-hour runs in Monday’s meeting (and possibly standalone_tree_evaluator.py runs with NNLearner2 if Devan and Steven are able to debug load_environment() by then). 

**Action Items:**
| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| --- | ----------- | --- | ----------- |----------- |
| Determine if Karthik's run produces individuals that can predict start index | Done | 11/8/21 | 11/15/21  | 11/15/21 |
| Finish EMADE setup on PACE | Done | 11/1/21 | 11/21/21  | 11/20/21 |
| Analyze discrepancies between EMADE.py and load_environment() for standalone_tree_evaluator.py | Done | 11/15/21 | 11/22/21  | 11/20/21 |

# Week 12: November 8th, 2021
## Overview
Research output numerization methods and finalize output layer output format. 

## Team Notes:
* In today’s NLP team meeting, members were assigned to one of our four remaining tasks for the semester: https://trello.com/b/6lcDBEj1/distilling-qa-systems-with-emade. I was placed on the Output Layer task along with Karthik, Rishit, and Jessi.  
* The current output layer primitive that the team has written for QA outputs probability matrices for each word in the context being the start and end index of the answer string (this is the same behavior as the BIDAF model’s output layer). In order to evaluate individuals’ predictions with our new F1 score evaluation function (which compares the number of words that match in the prediction and target strings), we need to reduce these matrices to single values that the evaluation functions can take in as input. 
* Ideally, we would like each individual to output a tuple of the form (prediction_start_index, prediction_end_index) that our F1 evaluation function can then take in and compare to the start and end index of the actual answer. 
* If we determine that a tuple of the form (prediction_start_index, prediction_end_index) does not produce individuals that give good results, we will look into other numerical representations of the string (ex. hashing the string in some way, vectorizing the string, etc.) and determining which representation produces the best-performing individuals. 
We will meet on Wednesday to determine the exact output format we want our models to predict, and we will proceed with preprocessing our dataset and modifying our output layer accordingly. 

## Subteam Notes:
* During Wednesday’s meeting, Rishit, Karthik and I talked to Steven about our problem definition a bit further and received clarification about which part of the architecture we should modify to adjust the output layer’s output. 
* Our neural_network_methods.py file contains a model.fit() call which compiles and trains an individual containing the BIDAF primitives we have implemented. This call generates trained individuals that will then output values of the label type we specify in the context and query DataPairs we pass in. Our goal is to make changes to model.fit() and preprocess our dataset such that we can have individuals predict a numerical representation of the answer string that our evaluation function can then use in evolving individuals.

Model Fitting:
```
history = model().fit(train_data_list, target_values, batch_size=batch_size, epochs = epochs, validation_data=(test_data_list,truth_data), callbacks=[es])
```

* Rishit, Karthik and I discussed different numerical representations of our target strings that we could pass into our models. We are currently considering passing in word/character/contextual embedding vectors to the models, but this could easily devolve into replacing the embedding layer’s responsibility in the model prematurely, which could easily lead to overfitting of our models on our training set.
* After running a quick Python script to ensure that the start index column in our dataset matched the string labels (insert script here), we decided to proceed with having our models predict the start and end index of the answer as a tuple, as this would be the easiest format for our evaluation functions to use. Script can be found here: https://colab.research.google.com/drive/1dx66YZSFYXimJAcZtU35yTkt_PYpiVL9.
* Karthik is currently testing out an EMADE run on PACE with the start index of the answer as the target variable (as a result of our output layer changes, we have modified our problem from a classification problem to a regression problem). 
* Squad Queries and Answers Sample:
![Screenshot (493)](https://github.gatech.edu/storage/user/47031/files/6beced0b-6810-493d-ac5a-e34bd27a78cb)

## Individual Notes:
* Since Jessi attended Wednesday’s meeting virtually, she asked if I could set up some time with her to help her get caught up with the output layer team’s progress and next steps. We met Friday afternoon at 3:00 PM for the same. I explained our team’s discussion on Friday as well as our new task of determining how Keras models can output tuples. 
* I found this article to contain the best explanation of multi-output regression implementation in Keras: https://towardsdatascience.com/multi-output-model-with-tensorflow-keras-functional-api-875dd89aa7c6. However, I foresee limitations with EMADE’s handling of our DataPairs that may cause this approach to not work for individuals’ predictions, even if we can implement Keras layers that output tuples of the form (start_index, end_index). I believe EMADE directly passes off individuals’ predictions to the designated evaluation functions in the .xml for a given run, which might cause the start and end indices to get separated from one another when being evaluated, defeating the purpose of a tuple output for our prediction string. I will confirm with Steven on Monday. 
* When I look at the EmbeddingLayer, it seems like the main thing we're doing there is tokenizing the datapair using texts_to_sequences(). I believe a starting point for us would be to look at other tokenization methods to use in this layer such as Penn TreeBank https://catalog.ldc.upenn.edu/LDC99T42 (which is good at identifying predicate-argument structure, a particularly common structure in answers for QA systems) and Gensim https://radimrehurek.com/gensim/, which already has pretrained embeddings for datasets of a variety of domains, including QA datasets. 

Current Embedding Layer taken from feature/nnlearner2:
```
    maxlen = MAXLEN 
    numwords=NUMWORDS   
    out_dim = abs(out_dim)
    data_pair = data_pair
    # data_pair = data_pair[0]
    if data_pair.get_datatype()=='textdata':
        data_pair, size, tok  = tokenizer(data_pair, maxlen, numwords)
    else:

        size = len(data_pair.get_train_data().get_numpy())
        maxlen = 1
    initializer = initializer.value()  
    layerlist.mylist.append(Embedding(size, out_dim, input_length=maxlen, embeddings_initializer=initializer))    
    return layerlist
```

* With these trained embeddings, we could embed the output once it is predicted and pass that to the evaluation function as a vector to compare to our embedded target vectors, allowing us to capture the entire strings as a numerized vector and allowing single-output regression at the same time. Moreover, if we were to move to a different evaluation metric that is more forgiving of answers that are slightly off in index but answer the question correctly, vectorized outputs would allow us to make this qualitative comparison directly (F1 score, which is what we are using correctly, merely compares the number of words in the prediction and target string that match). 
* With this research in mind, our output layer team will finalize our output format during Monday’s meeting. 

**Action Items:**
| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| --- | ----------- | --- | ----------- |----------- |
| Finalize Output Format from Output Layer After Team Discussion | Done | 11/8/21 | 11/15/21  | 11/15/21 |
| Research embedding methods and Keras multi-output regression | Done | 11/8/21 | 11/15/21  | 11/12/21 |
| Determine if Karthik's run produces individuals that can predict start index | In Progress | 11/8/21 | 11/15/21  | - |

# Week 11: November 1st, 2021
## Overview
First NLP team meeting, understanding NLP problem, EMADE setup on PACE. 

## Team Notes:
* Today, bootcamp students were placed into their subteams, and I was placed on the NLP team. In our weekly scrum, each subteam (NLP, Image Processing, Modularity, NAS) provided updates on their progress. While each team is working on solving different problems, some teams use the same GitHub branches for their work and/or share their commits with other teams in case they prove useful. I find this interesting because it points to EMADE’s ability to be repurposed for a variety of tasks, since it is a generic architecture for genetic programming which presents new capabilities whenever its baseline functionality is expanded or improved. 
* After scrums, I joined the NLP team and was introduced to its goals and current tasks. In order to meet and outperforming state-of-the-art QA systems, the team is currently implementing custom primitives based on the BIDAF model for QA systems as outlined in the following paper: https://arxiv.org/pdf/1611.01603.pdf. Returning team members are also working on debugging issues related to merges of these primitives into the existing codebase for NLP that were done earlier in the semester. I asked questions about Scikit primitives vs. Keras primitives and learned that a key distinction between NLP with BIDAF vs. other NLP or NAS systems is the usage of a 2-datapair input of the context and the query to make predictions as opposed to a single EmadeDataPair input (which is what most other teams are working with). 
* My tasks for this week include getting EMADE set up on PACE (scp-ing our working branch (https://github.gatech.edu/sleone6/emade/tree/EMADE-304-allow-cachev2-to-consume-aligned-datapairs) over to PACE, setting up a conda environment, ensuring I can login to a MySQL instance and submit jobs to the queue, etc.) using Cameron’s set-up video: https://www.youtube.com/watch?v=LashYCCJF3E&feature=youtu.be and this setup guide: https://github.gatech.edu/emade/emade/wiki/Guide-to-Using-PACE-ICE. 
* I will also read the following paper to better understand the purpose of the BIDAF model, the layers it includes, and how our custom-built EMADE primitives each map to one layer in the BIDAF model: https://arxiv.org/pdf/1611.01603.pdf. 

BIDAF model overview:

![Screenshot (491)](https://github.gatech.edu/storage/user/47031/files/46ce9df9-21cd-460c-a222-4a9510d54f02)

Notes on BIDAF Paper:
* Character Embedding Layer uses tokenization techniques to create vector representation of words in context and query using character-level CNNs.
* Word Embedding Layer similarly embeds the context and query at the word level using a pre-defined corpus and a word embedding model (ie. GloVe).
* Contextual Embedding Layer is an LSTM that considers the words surrounding each word to improve the accuracy of each word’s vector representation. 
The attention flow layer (the specialty of the BIDAF model) creates a similarity matrix that represents the similarity between the t-th context word and the j-th query word, where t is the row number and j is the column number. This layer outputs the query-aware vectors for the context words using the context words’ similarity values to the words in the query. 
* Modeling layer is a bidirectional LSTM which takes in the query-aware context vectors and outputs vectors that represent each word with respect to both the context and the query.
* Output layer takes in the modeling layer’s output and predicts the probability of each word in the context of being the start and end index of the answer. 

## Subteam Notes:
* In today’s meeting, Steven gave a short presentation on NLP and different types of neural networks in order to get new students acclimated with the Keras models that our team uses to build primitives for QA. 
* Devan also gave an introduction to the BIDAF model, the functionality of each of its layers, and the evaluation metrics currently being considered by the team (precision, recall, Mean Average Precision, Mean Reciprocal Rank, etc.): https://docs.google.com/presentation/d/1E1DZyeGYXwsT8WTRPRaiwko9gRsr3q1u/edit?usp=sharing&ouid=113962999086036620588&rtpof=true&sd=true. 

## Individual Notes:
* As PACE is down from November 3rd-5th for maintenance, I will continue to read the BIDAF paper and take a look at our team’s codebase for further clarity into how custom primitives are introduced to EMADE, how evolution remains stable with these custom primitives, and what the control flow for individual creation, evolution, and evaluation looks like with our primitives. 
* Currently, most of the primitives the team has built reside in neural_network_methods.py, with much of the logic for primitive set creation residing in EMADE.py (which passes inputs to individuals as EmadeDataPairs). Also, as is the case for all EMADE runs, we have an input_squad.xml file which contains the evolutionary parameters to be used for our team’s runs (our current evaluation metrics are accuracy and number of parameters, both minimization objectives), and unlike the Titanic problem, we pass in two inputs: the train-test split of the question dataset, and the train-test split of the context dataset. The multi-input and multi-output nature of our problem is leading our team to investigate EMADE’s architecture closely and make changes that meet our requirements, and I am excited for the opportunity to understand EMADE better as a result. 
* Dr. Zutty had suggested unit testing for our custom layers in our first subteam meeting. I think writing unit tests would be a great way for myself and other new team members to get familiarized with the functionality of each layer and contribute to the codebase given the relatively short amount of time remaining in the semester. However, based on direction from Steven, we will table unit testing until later if at all, given the amount of debugging currently taking place to integrate existing primitives. 

EMADE PACE Setup Issues
* I am running into an Already Exists error with Keras backend in gp_framework_helper.py. I believed this was due to the fact that both Tensorflow and Keras install Keras from our conda_env.yaml file, so I manually deleted Keras from my conda bin folder. However, this did not resolve the issue, so I will get further guidance during our Monday meeting from team members who are successfully set up. 
* Apart from this error, I am fully set up with a conda environment on PACE, have scp-ed our EMADE-304 branch from Steven's fork of the EMADE repository to my PACE account, and am able to login to a MySQL instance via PACE and submit a MySQL job to the PACE queue via "qsub pbsmysql.pbs." My next step will be to attempt to run standalone tree evaluator with NNLearner to ensure that I can submit EMADE jobs to the queue, and then run an EMADE master process on the SQUAD dataset to ensure I can help with runs to test that individuals containing our custom primitives evaluate properly. 

**Action Items:**
| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| --- | ----------- | --- | ----------- |----------- |
| Finish EMADE setup on PACE | In Progress | 11/1/21 | 11/8/21  | - |
| Read BIDAF paper and understand function of each layer (particularly inputs and outputs) | Done | 11/1/21 | 11/8/21  | 11/3/21 |
| Look through NLP codebase and make sense of control flow with custom primitives | Done | 11/1/21 | 11/8/21  | 11/5/21 |

# Week 10: October 25th, 2021
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
* Using arity of primitives to create "super-primitives" with same functionality as individuals created from base primitive set
* Observing interesting spikes in fitness of pareto optimal individuals during EMADE runs
* Want to ensure that goal of minimal size does not become a criteria (ie. not evolving only simple modular primitives and foregoing complex modular primitives) 

**Action Items:**
| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| --- | ----------- | --- | ----------- |----------- |
| Submit Subteam Preferences  | Done | 10/30/21 | 11/1/21  | 10/31/21 |

# Week 9: October 20th, 2021
## Overview
Workday for EMADE and final presentations.

## Team Meeting Notes
* Prepared for final VIP Presentation by continuing to run EMADE and obtaining results for MOGP evolution of individuals on Titanic dataset. Received help from VIP Alumni in fixing minor bugs in selection_methods.py file (selDCD() method not being able to take individuals with a length not a multiple of 4, resolved by pulling recent commit from EMADE git repository) and began setting up pymysql to pull in MySQL data as pandas dataframes and conduct analysis on metrics like average analysis time across generations, most frequently occurring primitives in Pareto optimal individuals in each generation, and number of valid individuals over time. 

## Subteam Notes
Met with team to run through presentation and ensure we were ready to share our findings of using EMADE for the Titanic problem as well as hear from main VIP teams on their team focus areas and get a sense of which team we would like to join after bootcamp. Had to adjust to Rayan D. withdrawing from the course by reassigning presentation parts close to the presentation date and revising our work for the Titanic ML and MOGP assignments. 
* As all of us were experiencing the OperationalError("MySQL Connection not available.") when trying to connect to my MySQL host with the pymysql package in a Colab notebook after some time, we decided to conduct our remaining analysis by pulling our EMADE databases as .csv files into our Colab notebook and working with them as Pandas dataframes. 
* Charlie worked on revising our MOGP results with Dr. Zutty and Dr. Rohling's suggestions from our earlier ML and MOGP presentation, as well as making the graphs for the FPR and FNR predicted by EMADE on the Titanic dataset. Since EMADE gave us the average FP and FN predicted across all 5 folds of our train and test sets, we multiplied this value by 5 to get the total FP and FN prediction count across all folds and divided by the total number of positive and negative values of our target feature to get the FNR and FPR across our entire dataset. Charlie worked on this conversion as well as creating a graph with our final generation's individuals' fitnesses plotted and our final EMADE Pareto front.
* My team and I met to practice the presentation on Sunday and discussed ways to best showcase our unique approach to the Titanic EMADE problem as compared to other bootcamp groups. We decided to focus most heavily on our analysis of EMADE's capabilities, limitations, and performance, as well as our results: our final Pareto front, average evaluation time for individuals over generations, learners which took in the ARG0 datapair directly, and our discussion of the tradeoffs between the ML, MOGP, and EMADE approaches to problems like Titanic as well as more complex problems with larger datasets.

## Individual Notes:
* I worked on extracting the primitive which directly took in the EMADE datapair (ARG0) for each individual in our final generation using Python regex expressions and string parsing, which gave us a sense of the primitives that can handle the features in our Titanic dataset directly most optimally. AdaBoostLearner() was the most frequently occurring such primitive. I also pulled the evaluation time data from the individuals table in our MySQL titanic database and plotted the average evaluation time for individuals across each generation of our final run, which showed that individuals on average evaluated more quickly in later generations than earlier ones. The script for the datapair string analysis and evaluation time analysis is as follows: https://colab.research.google.com/drive/1dEGTJ-ia7fWnXCvt0tP-CcQYgr1L6xTd.
* I worked on slides 2-5, 11, 13-15 of our final bootcamp presentation: https://docs.google.com/presentation/d/1Rgt1bLAuUg87MrD0WF8Mro7PKvBSm4EcFExPrzp5_wU/edit?usp=sharing. 
* Our Titanic project Colab notebooks can be found here: https://drive.google.com/drive/folders/1lq6fycfuDPxNamEK6inOa1vt8-RddgiS. 

Observations
Our revised MOGP Pareto frontier (using selDCD()) had an AUC of 0.1514, an improvement from our original Pareto frontier AUC of 0.2071 when we used selNSGAII() as our selection method, which failed to select individuals based on crowding distance correctly. A much greater number of individuals existed in the final generation of our MOGP run with selDCD(), indicating that selNSGAII() was weeding out individuals unjustifiably based on its truncation of individuals with similar ranks but different crowding distances. 
* We had a bug in our MOGP final generation graph with the selDCD() selection method where only the Pareto front was appearing on the graph, but not the rest of the valid individuals for the final generation. We saw that we had accidentally reset the population to the initial offspring list in each iteration of the evolutionary loop instead of setting the population to be the evolved offspring (after mating, mutation, etc.), so every individual was Pareto optimal in each successive generation. We fixed this and saw that all individuals were appearing on the fitness graph as desired. 
* EMADE's pset contained both simple and complex primitives (ex. logical operators and decision tree classifiers), which allowed for selection of individuals after several generations when many individuals consisted of the same primitives and therefore performed similarly to one another.
* We ran 20 generations of EMADE with an initialPopulationSize of 200 and a minQueueSize of 50. We initially started out with an initialPopulationSize of 512 and a minQueueSize of 250, but evaluation was taking around 1-2 hours with these changes (especially with inconsistent worker processes running on my master process), so we decreased both of these values such that we could at least get 10-12 generations to complete in a reasonable amount of time. The starting population for each generation grew steadily with time, such that by the 20th generation, evaluation was taking around 2 hours per generation as with the starting generation for EMADE with an initialPopulationSize of 512. 
* Our master process eventually ended when the following error was thrown in master.err:
```
sqlalchemy.exc.OperationalError: (pymysql.err.OperationalError) (2013, 'Lost connection to MySQL server during query ([WinError 10053] An established connection was aborted by the software in your host machine)')
```
* Both the number of Pareto optimal individuals and the number of individuals in total for each successive generation increased steadily, with a final Pareto front size of 23 and 314 valid individuals in this generation. 

```
Sample Individuals in Generation 13:
Received: AdaBoostLearner(ARG0, learnerType('LogR', {'penalty': 0, 'C': 1.0}), 2, 0.01)
	With Hash 076a79ddfb8f0527e074f92c3d3f63aa84ab18cbdb71f9973dd2e8823d333940
	With Fitnesses: (8.4, 64.0)
	With Age: 1.0
Received: AdaBoostLearner(myArcTangentMath(ARG0, 1), ModifyLearnerList(learnerType('Trees', {'criterion': 0, 'splitter': 0}), [15, 5]), passInt(0), myFloatDiv(100.0, 10.0))
	With Hash 45b1b86644bd9269f11e27b0676d8b0347b50f5675069eed50e34feaf7e2dc09
	With Fitnesses: (18.2, 20.6)
	With Age: 1.0
Received: lpfBlur(sp_div(contMaskRangeAspect(ARG0, -4.617371386969362), my_pow(ARG0)), lessThanOrEqual(myFloatAdd(myFloatAdd(myFloatDiv(myIntToFloat(myIntAdd(32, 5)), 1.0), 0.01), 100.0), myIntToFloat(1)))
	With Hash 52140884e55c36acfd6970647fe4e62e52dc62481351f9c6271865282020b81f
	With Fitnesses: (inf, inf)
	With Age: 0
Received: AdaBoostLearner(myBartlettHann(ARG0, 0), learnerType('Bayes', None), 6, ifThenElseFloat(notEqual(myFloatAdd(myIntToFloat(50), ifThenFloat(falseBool, passFloat(myIntToFloat(6267)))), ifThenElseFloat(falseBool, myFloatSub(myFloatSub(ifThenElseFloat(trueBool, 0.01, 0.01), myFloatDiv(0.1, 100.0)), ifThenFloat(ifThenBool(falseBool, trueBool), myFloatDiv(10.0, 1.0))), myFloatAdd(myFloatMult(myFloatAdd(myFloatAdd(passFloat(myFloatSub(0.1, 10.0)), ifThenFloat(ifThenBool(falseBool, falseBool), ifThenElseFloat(trueBool, 100.0, -3.448215911748788))), 0.1), ifThenElseFloat(falseBool, 0.01, 10.0)), myFloatAdd(ifThenFloat(trueBool, 3.396996932591346), myFloatSub(0.01, 10.0))))), 0.01, 1.4335719305516967))
	With Hash 228b12c29f95df6ee0a61ad4306ea8357541bf79518622e918e8c88f230a5307
	With Fitnesses: (inf, inf)
	With Age: 0
```

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

## Team Meeting Notes - EMADE Work Session
* We had a work session today to resolve any EMADE installation issues and ensure MySQL connection was possible for remote clients so that we could run additional master and worker processes in EMADE on our Titanic dataset and increase the size of our output data in preparation for fitness/evaluation time graphs for our final presentation. 

## Subteam Notes:
* Charlie is set up with a conda environment and the EMADE git repository cloned, but is having some issues with installing packages (ex. opencv-python) in his conda environment. Installation is highly laggy, which may be due to his eduroam connection; he will attempt to install these packages again and set up a MySQL connection to my host server after successfully doing so. 
* Rayan D. is continuing to experience lag issues with the build wheel while both cloning the EMADE git repository and installing packages in his conda environment; I am planning to set up a meeting with him later this week to help him debug these issues.
* Rayan K. has successfully created a conda environment with all of the required packages to run EMADE and is currently installing MySQL workbench to attempt remote connection with the changes to my my-cnf file from last week. If this is successful, we will continue to use the MySQL workbench to read and write to our EMADE databases, as the GUI it provides is easy to work with and write SQL queries in to retrieve evaluation data for the individuals in each of our runs.

SQL configuration meeting:
* Rayan K., Charlie, and I met to ensure that consistent remote connection to my SQL server was possible.
As Charlie was able to successfully connect to my database, we determined that the IP address returned from different sites for my computer were different from one another; as such, we experimented with each such IP address until we identified that this site returned the address that my team mates were all able to input into their input_titanic.xml files for successful connection: https://whatismyipaddress.com/ip/128.61.41.136. 

## Individual Notes
* I am fully set up with the EMADE git repository and a conda environment with all packages necessary to run EMADE. I have run master processes successfully 2-3 times, and during each run master.out indicates that the vast majority of our individuals have initial fitness values of (inf, inf). In addition, most of our trees have primitives which take in invalid inputs; we will wait and see if this issue is persisting and if it is after 10 generations, we will modify our primitive set in gp_framework_helper.py and ask Dr. Zutty if this is expected behavior for EMADE on the Titanic dataset in particular. 
* I started a master process Saturday morning and noticed that the evolution process was extremely slow as compared to the first trial run I did during the hackathon the previous Saturday. During this hackathon, we ran EMADE and obtained around 15 individuals in our Pareto front after about five generations. In this new run, each generation was taking approximately an hour to complete, with the number of individuals left for evaluation in the queue hanging approximately mid-way through each generation. This led my team to believe that their worker processes were not actually running on my master process as we had initially thought, or workers were periodically disconnecting throughout my master process. My team members continued to re-connect to my MySQL host until we were able to run a process which appeared to evolve individuals much more quickly than before.  

Meeting with Rayan
* I met with Rayan on Friday to help him get past the build wheel issue when installing packages in his conda environment. We initially tried running pip install --upgrade pip setuptools wheel and re-installing based on a suggestion from a Stack Overflow post. As this did not resolve the issue, we tried troubleshooting his system to identify any storage limitations/junk files to delete, and having cleared his Anaconda bin file of all old environments, he was able to install all conda packages and proceed with the rest of EMADE setup successfully. 

My Titanic Dataset Preprocessing:

```
import sklearn.model_selection
import sklearn.feature_extraction
import pandas as pd
import random
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from numpy import array
from numpy import argmax
from sklearn.metrics import confusion_matrix

train_data = pd.read_csv('train.csv')

train_data.drop(columns=['Name', 'Cabin'], inplace=True)
train_data.set_index(keys=['PassengerId'], drop=True, inplace=True)

train_nan_map = {'Embarked': train_data['Embarked'].mode()[0], 'Ticket': '100'}

train_data.fillna(value=train_nan_map, inplace=True)

values = array(train_data['Embarked'])
# print(values)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)

enc = OneHotEncoder(handle_unknown='ignore')
enc_df = pd.DataFrame(enc.fit_transform(train_data[['Embarked']]).toarray())
train_data = train_data.join(enc_df)
del train_data['Embarked']

columns_map = {'Sex': {'male': 0, 'female': 1}}
train_data.replace(columns_map, inplace=True)

#train_data

import re

train_data['Ticket'] = train_data['Ticket'].apply(lambda x: int(re.findall(r'\d+', x)[len(re.findall(r'\d+', x)) - 1]) if len(re.findall(r'\d+', x)) > 0 else 0)
train_data.head()
# test_data.head()

train_map = {}
for x in train_data['Pclass'].unique():
  train_map[x] = train_data[train_data['Pclass'] == x]['Age'].median()  

train_data["Modified Age"] = train_data["Pclass"].apply(lambda x: train_map[x])

train_data["Age"] = train_data["Age"].fillna(train_data["Modified Age"])

del[train_data['Modified Age']]

train_map = {}
for x in train_data['Pclass'].unique():
  train_map[x] = train_data[train_data['Pclass'] == x]['Fare'].median()  

train_data["Modified Fare"] = train_data["Pclass"].apply(lambda x: train_map[x])

train_data["Fare"] = train_data["Fare"].fillna(train_data["Modified Fare"])

del[train_data['Modified Fare']]

train_nan_map2 = {0: 0.0, 1: 0.0, 2: 0.0}

train_data.fillna(value=train_nan_map2, inplace=True)

train_data.rename(columns={0:'A', 1: 'B', 2:'C'}, inplace=True)

survived = train_data["Survived"]
train_data = train_data.drop('Survived', axis=1)

print(train_data.isnull().sum().sort_values(ascending=False))
vectorizer = sklearn.feature_extraction.DictVectorizer(sparse=False)
train_data =  np.hstack( (vectorizer.fit_transform(train_data.to_dict(orient='records')), survived.values.reshape((-1,1 ) ) ) )

kf = sklearn.model_selection.KFold(n_splits=5)

for i, (train_index, test_index) in enumerate(kf.split(train_data)):
    np.savetxt('train_' + str(i) + '.csv.gz', train_data[train_index], delimiter=',')
    np.savetxt('test_' + str(i) + '.csv.gz', train_data[test_index], delimiter=',')
```

Evolutionary Parameters Specified in input_titanic.xml:
```
<initialPopulationSize>200</initialPopulationSize>
<elitePoolSize>200</elitePoolSize>
 <launchSize>100</launchSize>
<minQueueSize>50</minQueueSize>
<outlierPenalty>0.2</outlierPenalty>
```


We observed that most individuals evolving in the first few generations of our run were either returning with an error or evaluating to (inf, inf). Only around 5-10 individuals were actually evaluating to finite fitness values in the first 10 generations of our run; we expect to see more individuals evaluating successfully in successive generations as individuals that are evaluating properly are rewarded by our selection methods. 
When we restarted EMADE once with reuse=1, we observed that the number of individuals at the beginning of generation 0 was drastically greater than the starting number of individuals in the initial run (~750 individuals vs. ~200 individuals for an initialPopulationSize of 200). This indicated that seeding increased population size as well as the number of individuals that evaluate successfully at the end of each generation, but this also meant that each generation took much longer to run. As such, we avoided reuse going forward and saw an EMADE run to completion with reuse=0. 

![Screenshot (489)](https://github.gatech.edu/storage/user/47031/files/aaaf8d1c-1d35-47e8-a3b5-447fc2f6e889)

**Action Items:**
| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| --- | ----------- | --- | ----------- |----------- |
| Initiate EMADE run with worker processes and databases in MySQL workbench | Done | 10/13/21 | 10/20/21  | 10/15/21 |
| Assist Rayan D. with conda environment setup | Done | 10/13/21 | 10/20/21  | 10/16/21 |
| Understand EMADE database outputs, ideate graphs to track EMADE activity over time | Done | 10/13/21 | 10/20/21  | 10/17/21 |

# Week 7: October 6th, 2021
## Team Meeting Notes
### Lecture on EMADE
* Introduced concept of EMADE (engine to perform multi-objective algorithm design using genetic programming and a primitive set containing both simple primitives and ML learners from Python packages).  
* Looked at EMADE repository on GitHub and got a view of input file that specifies MySQL database configuration for EMADE output and parameters of evolution, launchGTMOEP.py which initiates the evolutionary process, and the gp_framework_helper.py file that contains references to primitives used to create EMADE individuals).
* Received information about presentation on Monday, October 25th where bootcamp and returning students will present their EMADE presentations and hackathon on Saturday, October 16th, where new students can receive help from returning students for EMADE setup and analyzing output from running EMADE on Titanic dataset. 


## Subteam Notes
* Worked with rest of team asynchronously to set up master and worker processes for EMADE.  
* Rayan D. is having trouble setting up his conda environment and installing all of the required packages, as the build wheel is hanging every time he runs the "conda install" and "pip install" commands for his conda environment with Python 3.6. We believe this may be due to his WiFi/VPN connection and he will try again on eduroam and guest to eliminate any address-pinging issues he is currently having.

## Individual Notes
* I am running the master process (main evolutionary loop), while the others are running the worker processes (evaluation function and results). I was able to run a master process successfully after rewriting the selNSGA2 method to only perform selectDCD on lists of individuals whose length is a multiple of 4. 

![Screenshot (485)](https://github.gatech.edu/storage/user/47031/files/9f782859-88eb-4db2-accb-47f3dd3f2b53)

* Having done this, I ran the master process again and noticed that inf fitness values are being printed for certain individuals. This will likely be resolved when we replace the existing preprocessing in the titanic_splitter.py with our own preprocessing, which handles null and invalid values. We will also ensure that my other team members are able to run worker processes today during our team meeting, and if not, tweak any specifications of my localhost such that it accepts remote connections.
* I used https://whatismyipaddress.com/ip/128.61.41.136 to identify my IP address and share this with my team members so they could replace the host name string in their input_titanic.xml file and connect to the server I had created. I initially modified the value of the "Limit to Hosts Matching" field under the Users and Privileges tab in MySQL workbench to '%' so that any host would be able to connect to the instance of my MySQL server. However, my team members were still unable to connect to my instance with this change. After much research on MySQL's permission configurations, I changed the bind address specified in my.cnf in the MySQL bin folder to 0.0.0.0, so that any IP address would be allowed to connect to my MySQL server. Remote connection was still failing with these changes, at which point I found this article: https://linuxize.com/post/mysql-remote-access/ which indicated that access needed to be granted via the GRANT command to all users with the following command: GRANT ALL ON database_name.* TO user_name@'ip_address' IDENTIFIED BY 'user_password'; and that any firewalls would have to be removed. With these changes, my team workers were able to run EMADE worker processes with our own preprocessed Titanic data from the ML and MOGP assignments and write to and read from my MySQL instance. 

![Screenshot (487)](https://github.gatech.edu/storage/user/47031/files/a20c73e5-42be-430c-8acc-412a2d287479)
 
**Action Items:**
| Task | Current Status | Date Assigned | Suspense Date | Date Resolved |
| --- | ----------- | --- | ----------- |----------- |
| Finish Worker Process Setup + Adding Preprocessing to titanic_splitter.py | Pending | 10/6/21 | 10/17/21  | 10/16/21 |
| Run EMADE on preprocessed Titanic Data | Pending | 10/6/21 | 10/17/21  | - |
| Write Python scripts to capture output in graphs (Pareto frontier, AUC, Individuals Over Time, T-Stats, etc.) | Pending | 10/6/21 | 10/18/21  | - |

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
![Screenshot (450)](https://github.gatech.edu/storage/user/47031/files/7d8e9d40-2ef0-45e1-b974-f49e2b94d69a)
* Tried mutUniform and cxOnePoint, AUC improved when using mutNodeReplacement and cxOnePointLeafBiased with termpb = 0.1
* Change 30 generations to 50 generations for improved evolution
* Titanic ML and MOGP Presentation: https://docs.google.com/presentation/d/1tK83vBU6uQFYQGAivnSjWEM4Ghw3qJaGR5Py14BocJk/edit?usp=drive_web&ouid=106540897889834720619 (Created slides 1, 2, 4, 5)

Sample Learner: logical_and(not_equal(Sex, negative(multiply(multiply(C, Parch), Age))), greater(Ticket, SibSp))

Best Learner: FPR = 0, FNR =  0.9122807017543859

MOGP Pareto Front:

![Screenshot (426)](https://github.gatech.edu/storage/user/47031/files/4366dfb9-079b-4b4f-be5a-8b5d9d5a67e6)

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

![Screenshot (392)](https://github.gatech.edu/storage/user/47031/files/c7badbb1-d7c1-4088-a0d9-6049b9e73379)

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

Before Modifications:

<img src="https://github.gatech.edu/storage/user/47031/files/e863eae0-0aaf-4c6f-8793-4dc667a68ea4">

<img src="https://github.gatech.edu/storage/user/47031/files/ae21bdb9-e155-4ee1-83a8-bbe7695dc0d5">

<img src="https://github.gatech.edu/storage/user/47031/files/b3dd72ee-5e49-47ff-b7d2-abd60044662b">

After Modifications:

<img src="https://github.gatech.edu/storage/user/47031/files/90f6f99f-567b-462d-b8c1-b8f22290176a">

<img src="https://github.gatech.edu/storage/user/47031/files/bb1f8b60-37d2-464f-a3d5-d89c7e9e41f2" width="50%">

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
![Screenshot (421)](https://github.gatech.edu/storage/user/47031/files/e3de3a9c-1523-40d1-8d58-138294f30d2d)
![Screenshot (426)](https://github.gatech.edu/storage/user/47031/files/c17c900b-e0ef-4bd6-a16b-e74a29a6ca0a)

* We can see here that the maximum fitness value seems to oscillate around a fitness of about 2.0 and does not continue decreasing after about the 10th generation. 

To improve the fitness of the evolved individuals, I added the floor and maximum operations from the numpy library. I also registered the mutInsert function as an additional mutational function to use in the evolutionary loop. This mutation function randomly selected a node in the given individual and creates a new subtree using that node as a child to that subtree. The following graph of fitness over time reflects those changes:

<!--![Genetic Programming Visualization after Floor and Maximum Primitives Added](https://picc.io/Un4_bet.png)-->
![Screenshot (424)](https://github.gatech.edu/storage/user/47031/files/04f8edce-2f1d-4f1a-93fc-084b77d2cad7)

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

![Best Ind One Max](https://github.gatech.edu/storage/user/47031/files/aa90dcca-d670-412d-b190-e076e8641170)
![OneMax Generations](https://github.gatech.edu/storage/user/47031/files/f25111cf-38b3-4533-9d5c-07361438c066)

The best individual was as follows: Best individual is [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], (100.0,). 

The global maximum (a best individual with a fitness equal to n, the number of entries in each individual) was reached within 40 generations about every 19 out of 20 times the algorithm was run; this indicates that our algorithm has an effectiveness of around 95%. Further improvements can be made by changing the bounds of the random number generation for crossover, mutation, and selection, increasing/decreasing the size of the population and the number of generations, as well as trying other crossover and mutation methods. 

**N Queens Problem:**
For this problem, we followed many of the same steps that appeared in the One Max Problem (see above). We define a size n = 25 for each individual and define a weight of -1.0 here, since we wish to minimize the number of conflicts between queens in our problem space. We then create a permutation function to populate the entries for each individual with numbers selected without replacement from range(n). For instance, the following individual will have a queen in the 1st row and the 8th, the 2nd row and the 1st column, etc.: [7,0,4,6,5,1,9,2,8,3]. We define our evaluation function as a measure of the number of conflicts along each diagonal of our board; with the creation process we defined for individuals, queens will not appear in the same row or column. [Describe evaluation function modification here w/ screenshots]. We then write the cxPartialyMatched() function for partially matched crossover, cxTwoPoint(), and mutShuffleIndexes() to shuffle values at different indexes within each individual (since we must remain within size n  = 25). We modified the mutation function to be a uniform int mutation, wherein randomly selected entries for each individual are replaced with a randomly selected value between 0 and n. The improvements seen with this new mutation function are described in the Findings section below. Finally, we run a similar evolutionary loop as the one described for the One Max Problem (see above) for 100 generations, return the fitnesses of the individuals (based on the predefined fitness operation - the number of conflicts between queens) and print statistics. We loop for some number of generations (100, in this case) and report the best individual that has resulted from this evolution process. 

Findings:

1. With Shuffle Indexes/Uniform Int Mutation:

![Screenshot (296)](https://github.gatech.edu/storage/user/47031/files/d6f6ab76-9aaa-4a4c-87f3-d65612a2fa0c)

![Screenshot (295)](https://github.gatech.edu/storage/user/47031/files/df8e7c1a-31c0-4e4d-aa9a-549a3c5236a8)

* We can see here that the maximum fitness value decreased much more quickly with the Uniform Int mutation than the Shuffle Indexes mutation. We also see that the average and minimum fitness values tended towards 0 more closely than they did with the Shuffle Index mutation. 

2. With 85 Generations and 10% Mutation Rate (Shuffle Index Mutation):
![Screenshot (413)](https://github.gatech.edu/storage/user/47031/files/718b57d0-f301-41a1-861e-970e07ee8180)
<!--![N Queens Visualization with 85 Generations and 10%  Mutation Rate](https://picc.io/MZtm5UD.png)-->

* We can see here that with a 10% mutation rate as opposed to the initial 20% mutation rate and with 85 generations as opposed to 100, we obtain a best individual with a fitness of 0 more consistently than we did previously. The maximum fitness also trends towards our best fitness more quickly than before. This also points to the fact that Shuffle Index Mutation may not be the best mutation for this particular problem, since a lower percentage of that mutation led to more consistent results in fewer generations. 

I also modified the evaluation function to return a tuple (sum_left, sum_right), where sum_left is the number of conflicts between queens strictly on left diagonals and sum_right is the number of conflicts between queens strictly on right diagonals. This allowed us to attempt minimization of both objectives in the evolutionary loop such that the evolution did not produce many individuals that had a high number of conflicts on left diagonals or right diagonals but not distribute conflicts on both types of diagonals. 

4. With (left_diagonal_conflicts, right_diagonal_conflicts) as fitness tuple
![Screenshot (299)](https://github.gatech.edu/storage/user/47031/files/5513ddc1-8c91-4380-b8b9-31bd1fb80ec2)

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