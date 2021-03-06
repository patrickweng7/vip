# Team Member
<b>Team Member:</b> Steven Leone <br>
<b> Major: </b> Computer Science <br>
<b>Email:  </b>sleone6@gatech.edu <br>
<b>Cell Phone:</b> (412)-378-7253 <br>
<b>Interests:</b> Machine Learning, Natural Language Processing, Software Engineering, Algorithms <br>
<b>Sub Team:</b> NLP <br>
<b>Sub Team Teammates:</b> Karthik Subramanian, Devan Moses, Kevin Zheng, Shiyi Wang, George Ye, Rishit Ahuja, Jessi Chen, Aditi Prakash, David Zhang, Manas Harbola, Yisu Ma


Original Notebook can be found at: https://github.gatech.edu/emade/emade/wiki/Notebook-Steven-Anthony-Leone



# Fall 2021

My team's final presentation can be found here: https://docs.google.com/presentation/d/1mnFnhxyJnRowr6T-qh05yUMT50rSYqUQig7FIiPekWI/edit?usp=sharing

## Final Presentations
* Image Processing
* * Main Objective: How can we improve EMADE on image processing tasks
* * Leverage GP, selection methods, mating and mutation, new primitives, and deep learning
* * Normalized, conducted Horizontal flipping, and newer augmentations
* * Size of dataset is 4000 images
* * Changed problem to not be multilevel (still multiclass)
* * Comparing TPR vs FPR and Num Parameters
* * Baseline Results: AUC was 0.21895
* * NSGA II was used by baseline, then used NSGA 3 and Lexicase
* * In actuality, NSGA II ended up performing significantly better.
* * Trees started to become too complex to yield meaningful results come generation 4 and beyond.
* NLP
* * We presented our results: https://docs.google.com/presentation/d/1mnFnhxyJnRowr6T-qh05yUMT50rSYqUQig7FIiPekWI/edit?usp=sharing
* Stocks
* * Comparing their results to a paper
* * Changed objective functions this semester
* * Experiment status: not all trials were completed due to time constraints, but can do some analysis
* * Compared their results to a paper
* * Originally, the plan was to replicate the paper's logic in EMADE.
* * Ultimately couldn't replicate it fully
* * Now seeing if they can outperform the paper.
* * Best Individual is extremely complex.
* * Managed to beat the paper in every stock, except for Johnson and Johnson
* Modularity
* * Focus on ARL's
* * Created a script called CloudCopy for easier setup
* * Still using the same titanic dataset
* * Compared ARL Runs 
* * Had unreliable data in midterm presentation
* * Effects of ARL complexity on performance was high 
* NAS
* * Started off by defining what a search space was
* * Had a short term goal of producing complex individuals, and a long term goal of producing individuals
* * Produced extremely complex individuals
* * Added weight sharing
* * Added module as a primitive
* * Introduced Leveraging Modules
* * Introduced Weight sharing this semester
* * Weight sharing is: if a module was trained, load old weights.
* * Reading and writing shared weights to a database was bad (errors, very slow)
* * Instead, they stored weights in a file, which was much faster. 
* * Added Max Pooling

## Week 16, Dec 6th

### General Meeting Notes
* At this week's general meeting, we didn't do the normal structure of a scrum meeting. 
* Instead, we focused on defining what success would look like for the semester. 
* Before going into that, I checked in with the team to make sure everyone was good with PACE and on track to get runs in. Everyone was fine, either currently doing runs or in the last stages of debugging making EMADE work on PACE. 
* We then briefly transitioned to the final presentation. I made a new doc for it and shared it with the team: https://docs.google.com/presentation/d/1mnFnhxyJnRowr6T-qh05yUMT50rSYqUQig7FIiPekWI/edit?usp=sharing 
* We divided up who would say what, and would flesh it out on Wednesday. 
* We then transitioned to our experiment, and I talked to Rishit and Shiyi regarding our statistic.
* Our objective was to prove that AutoML could be used to improve Question Answering models. We had changed to MSE and num params as our evaluation metrics, and would compare our singular seeded individual to our Pareto Front individuals at the end, using AUC as a metric.
* We talked further about this with Dr. Zutty and Dr. Rohling. After further discussion, we ended up defined outperforming the seeded individual as having that individual no longer on the Pareto Front. In other words, how many individuals could we get in a region of interest on the Pareto chart that outperformed the seeded individual in both objectives? 
* We could then look at the problem with a Bernoulli distribution. Did it succeed or not?

### Sub Team Meeting Notes
* At this week's sub team meeting, we worked on the final presentation and clarified anything anyone was confused on.
* The final presentation can be found here: https://docs.google.com/presentation/d/1mnFnhxyJnRowr6T-qh05yUMT50rSYqUQig7FIiPekWI/edit?usp=sharing
* A lot of team members had one off, individual issues, so I set up meetings with them to clarify things and attempt to fix their issues.

### Meeting with Devan
* First, I met with Devan. This meeting was just to finish planning out our discussion of NNLearner2, as we had both debugged parts of it.
* We also assigned him to talking about multiple data pairs in the presentation.

### Meeting with Jessi
* Jessi was having issues getting runs in. I met with her and debugged our way through. The core of the issue was that her environment was that needed for the master branch, not the branch we were on.

### Other miscellaneous conversations
* Most of the rest of these conversations post sub team meeting were on clarifications about their parts in the presentation.
* George also still had issues with getting the seeds to show up properly. We eventually found that, if you are close to reaching Disk quota on PACE, too many epochs can cause an individual to fail for memory reasons.
* I clarified our problem with multioutput regression to Aditi.

### Action Items
Task | Current Status | Date Assigned | Date Resolved | Date Due |
--- | --- | --- | --- |--- |
Get at least one more EMADE Run with SQUAD| Complete | 12/06/2021 | 12/08/2021 | 12/08/2021 |
Create final presentation | Complete | 12/06/2021 | 12/09/2021 | 12/10/2021 |
Analyze EMADE runs on SQUAD Experiment Results and create Visualizations with Rishit and Shiyi | Complete | 12/06/2021 | 12/09/2021 | 12/10/2021 |


### Getting Runs in
* I continued getting runs in this week. As each one was to be 24 hours long, I figured out that, even with the debugging required by setting up a run, I would have time for 2 more runs.
* I got one more run in successfully.
* After that, however, PACE was getting very crowded. By Wednesday most members on our team, myself included, would add jobs to PACE's queue, and they would remain there for several hours.
* Thus, I was only able to get 1 good run in. 
* I added it to the folder of runs, which I had given to the team earlier this week: https://drive.google.com/drive/folders/16CnEwE8sH6MpxYL1BXF3heFVYUPTZwxS?usp=sharing
* This is our record for keeping track of details regarding runs: https://docs.google.com/document/d/1JqUTAvFtVeUjeLXaQQLYsbzi8vrLYKBosCeDAg9nEHM/edit?usp=sharing

### Analysis and Visualizations
* On Thursday, I met with Rishit and Shiyi to analyze the results of our runs, linked in the section
 above.
* We sorted through run results, and found that the best we could analyze (without only having one result to look at) would be 4 sets of two continuous 8 hour runs, for a total of four 16 hours runs.
* We compiled their Pareto fronts and began to analyze them.
* Our goal was to see if auto machine learning could be used to improve an existing question answering model. To do this, we would look at our runs, which we seeded with an individual, and see if individuals that beat the existing model in all objectives could be produced. 
* We could measure our success by looking at the area of a region that we dubbed the "region of interest". This would be the total area bounded by our Pareto Front and the rectangle made by the origin and seeded individual as opposite corners.
* An example (taken directly from our results). Is shown below. The yellow shaded region represents the region of interest. 
* <img width="361" alt="Screen Shot 2021-12-12 at 4 54 42 PM" src="https://github.gatech.edu/storage/user/27405/files/466630e0-c264-48d5-8366-cd304dc195dc">
* We could compare and easier understand our numbers by looking at region of interest's area as a percentage of the area bounded by the seeded individual.
* I wrote up code in one of our Collab Notebooks to calculate the area of the region of interest. The code is at the bottom of this link:
https://colab.research.google.com/drive/1S5ojJMDKG8L0aNYrzHFjqhDeA19H18SI?usp=sharing
* You can also find a picture of it here: 
*  <img width="1040" alt="Screen Shot 2021-12-12 at 5 26 12 PM" src="https://github.gatech.edu/storage/user/27405/files/376ec182-e2f9-4209-a07f-2fe4597d1434">
* I plugged in numbers from our Pareto fronts of our four different runs to produce visualizations from various code Rishit and Shiyi wrote. These are the results:
* <img width="331" alt="Screen Shot 2021-12-12 at 5 27 53 PM" src="https://github.gatech.edu/storage/user/27405/files/2d838e02-93de-4313-b819-7a97b6a3eaee">
* <img width="289" alt="Screen Shot 2021-12-12 at 5 28 13 PM" src="https://github.gatech.edu/storage/user/27405/files/b9f6e4fc-1efd-4133-a8e5-c6024bafdaff">
* <img width="281" alt="Screen Shot 2021-12-12 at 5 28 35 PM" src="https://github.gatech.edu/storage/user/27405/files/63344db7-a48b-454d-93f4-1ce074bd8930">
* Our fourth run did not produce a run better individual. The seeded individual remained on the Pareto Front.
* My overall analysis of the results, which I discussed with Rishit and Shiyi, is as follows. We had four runs, and 3 of them by our definition of success resulted in success. 2 of them reduced area bounded by the seeded individual by 10%, and another one was by only .1%. From one perspective, the run with .1% could realistically be seen as a failure. With four runs, only half of them managed to improve a seeded individual, indicating auto machine learning had trouble improving a seeded individual on this data, with these primitives. From another perspective though, in 75% of our runs, we were able to improve from a seeded individual, 2 of them being rather significant. Overall, I think the second case is stronger, as we definitely can't rule out that auto machine learning can improve seeded individuals- our results mostly showed improvements. Yet, the strongest factor here is definitely our limited amount of data. We only got four runs in. We only had 3 runs that were successful, but we also only had one failure. We would need many more runs before we can get a very strong sense of results here.
* Thus, overall, we decided that our results indicated that we could use auto machine learning to improve an existing question answering model, but we lacked the data to make any firm conclusions.
* One of the most significant factors I saw here, as discussed with many of our other members, was that our individuals weren't performing well at all. Accuracy was about 1% and MSE was around 400,000 to 500,000. MSE is less of a visually appealing statistic (we will have to go back and double check that this is a sum of squared errors), but, from our final presentation, it is possible that our models are only off by two every time. As accuracy requires an exact match, this is very well possible. 
* Next semester, we need to hand train a model for sure, using all primitives in EMADE. This would involve Keras instead of PyTorch, as the model we trained with our Collab notebook couldn't be exactly compared.
* We also need to fix the BERTInputLayer, which crashes our models. Not much debugging has been done here.

## Week 15, Nov 29th

### General Meeting Notes
* NLP: 
* * We had an error in NNLearner2 with the test and train data being sent in wrong. It is fixed now.
* * Accuracy still rather weird, one half of what the values should be. Working on fixing this.
* * The above described error turned out to be because the length of the test data was not being divided by num_inputs in EMADE.py. I spoke with Dr. Zutty about this, and he mentioned it was likely an issue with the length of the test data being doubled, as we were now passing in two data pairs. I fixed this for now by just dividing the length of the test input by 2, but this will need fixed later to be division by the number of inputs as opposed to a fixed 2. This is a picture of my commit: 
<img width="810" alt="Screen Shot 2021-12-12 at 7 33 37 PM" src="https://github.gatech.edu/storage/user/27405/files/96f79272-e78b-4f90-b460-4e7f8b502144">

* NAS:
* * Not a very heavy week
* * Have been working on the final presentation, will rehearse it soon
* * Making a few pull requests
* Modularity:
* * Continuing their runs
* * Hoping to start their stock runs soon
* Image Processing:
* * Not a very heavy week for this team
* * Not quite at a code freeze, but will start runs this week.
* * NAS has pushed some updates they may want to add in before they officially code freeze.



### Sub Team Meeting Notes
* At this week's sub team meeting, we worked on getting our code freezed codebase onto everyone's computer to start getting runs in. 
* I had already pushed my updates to our fork: https://github.gatech.edu/sleone6/emade/commit/bf8992725e7107d4f134db2c4a3e63623ec4f075 and https://github.gatech.edu/sleone6/emade/commit/79d92c48f945b20591b3f2425620d7aba023ae2c are some of my most recent commits at this point in time.
* However, the seeding wasn't working for others. Devan ran an 8 hour run and had no NNLearner2 individuals on the Pareto Front throughout the entire run.
* I ran again to ensure seeding still worked for me, and I was able to get NNLearner2's that I seeded immediately. Thus, we determined there was probably a file that I hadn't updated on our branch. 
* To resolve this, we used a temporary workaround. I uploaded my entire codebase as is to a new repo: https://github.gatech.edu/sleone6/emadebackup. Thus, they could just run git clone on this entire repo to easily move all of my code in.
* For the rest of the meeting, I helped others move files over to PACE and start getting runs in.


### Action Items
Task | Current Status | Date Assigned | Date Resolved | Date Due |
--- | --- | --- | --- |--- |
Complete EMADE Runs on Squad| Complete | 11/29/2021 | 12/04/2021 | 12/06/2021 |
Create a branch for code freezed EMADE | Complete | 12/01/2021 | 12/01/2021 | 12/06/2021
Help other Team Members get set up on PACE | Complete | 11/29/2021 | 12/06/2021 | 12/06/2021

### Completing runs
* I carried on with runs this week, as we resolved our last issue with accuracy with this commit (though we will have to change it somehow to num_inputs later on): https://github.gatech.edu/sleone6/emade/commit/79d92c48f945b20591b3f2425620d7aba023ae2c




## Week 14, Nov 22nd



### General Meeting Notes
* NLP:
* * We updated the team on what we accomplished leading up to and after the Hackathon. We also mentioned our current error, with the results looking like classification.
* * This week, we focused a lot in our meeting about general procedures for our experiments
* * Dr. Zutty helped us more formalize our Hypothesis and to make clear what our experiment should look like, as we changed from trying to just distill QA models and are trying to see if we can use AutoML just to improve them.
* * New Hypothesis: We can use Auto Machine Learning to improve QA models.
* NAS:
* * This week they focused on model and preprocessing improvements, and internal updates among members.
* * The final presentation is currently being created.
* * Improved the model by getting rid of pooling layers.
* * Lots of preprocessing improvements, including subtracting by the pre pixel mean.
* Modularity:
* * Documenting more of their code better.
* * Preparing to start runs 
* * Working on data visualizations, 
* Image Processing:
* * Have been working on their assigned teams still.
* * Most of the team, even new members, have EMADE set up and their database running.
* * Currently getting runs in, but results are bad (AUC is pretty high).



### Sub Team Meeting Notes
* Thanksgiving break was this week, so we had no sub team meeting. We would only continue to work on items over the break.


### Action Items
Task | Current Status | Date Assigned | Date Resolved | Date Due |
--- | --- | --- | --- |--- |
Add Regression to NNLearner2 by removing hardcoded activation in neural_network_methods.py | Complete | 11/22/2021 | 11/23/2021 | 11/29/2021 |
Fix incorrect accuracy issue in eval_methods.py | Complete | 11/22/2021 | 11/29/2021 | 11/29/2021 |
Fix train/test data split in NNLearner2 in neural_network_methods.py | Complete | 11/25/2021 | 11/28/2021 | 11/29/2021

### Debugging NNLearner2 - Classification to Regression
* I talked with Devan a bit before the General Meeting. NNLearner2's output seemed to be stuck on classification.
* We both went through EMADE to see if we could find support for regression with NNLearner's. Devan did discover the <regression> and <multilabel> flags in the XML file, but this didn't have an effect.
* I did see that our output layer added a dense layer, but that should still support regression.
* I stayed after the meeting and talked with Dr. Zutty. His best suggestion for debugging was to try removing the activation function.
* This makes sense as an activation function is modeled to squeeze everything to either a 1 or a 0. Either it activates, or it doesn't. This could easily be switching the final output to match a binary classification. 
* I checked some other Keras documentation, and found out that we shouldn't be using accuracy either to fit our models. I switched it to MSE (mean squared error).
* With these changes implemented, NNLearner2 started working on Devan's laptop. He pushed to our Fork, and we were now at least getting varying results, and not just 1's and 0's.

### Debugging NNLearner2 - Train/Test not being passed into model().fit() correctly
* Several things still seemed wrong with NNLearner2 even after fixing the classification/regression issue, primarily regarding the outputs.
* Most of our final accuracies were around 0.5, even though almost all of theme were wrong, and it should have been 1.0. Devan, Kevin and I discussed this. We assumed early on that this was correlated to the num inputs (1.0 / 2 - 0.5, and our num inputs was 2). 
* I went through NNLearner2 at an almost line by line level to check and validate what was going on through the flow of our code.
* That was when I discovered that our code was quite off regarding passing in the train data and test data into our final Keras model.
* Essentially, it looked like we were only taking a concatenation of the context and question and passing them into our models. This didn't allow our model to take advantage of the multiple input layers or the one to one mapping of contexts and questions. We wanted to allow any of the inputs to have learning done on them completely separate from the others. 
* I found a resource that guided me on how to pass in and split up the test, train, and validation data correctly in a Keras model with more than one input: https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/
* Essentially, when calling model().fit(), input should be passed in as a list, where the first element goes into the first InputLayer(), and the second element into the second InputLayer().
* Then, the validation data had to be defined as a tuple, with the list of validation x's and then the y.
* My code for defining these values is shown below: 
* <img width="685" alt="Screen Shot 2021-12-07 at 11 38 52 PM" src="https://github.gatech.edu/storage/user/27405/files/7efdf845-fd73-4193-a8df-614106399eec">
* I then updated our line where we called model().fit() to reflect these changes:
*  <img width="874" alt="Screen Shot 2021-12-07 at 11 41 36 PM" src="https://github.gatech.edu/storage/user/27405/files/03159b5b-fe64-49fe-bc37-84b772c839cc">
* I tested these changes and noted some improvements. Accuracy looked a little better, but the main confirming factor that our code worked was when I actually got an error for accidentally switching around some values in x and y. The error displayed showed me the exact sizes of my data passed in, and I was able to confirm after switching the values that NNLearner2 was passing in the correct values to train, validate, and test on.


## Week 13, Nov 15th
### General Meeting Notes
* NLP:
* * I updated the team on how we were reducing the scope of our problem to just one index for now, due to issues with sending a tuple of ground truth values through the flow of emade in addition to issues with keras and multi output regression, as discussed with Karthik and Rishit the previous week.
* * There was some confusion on this, I had to explain this deeper to one of our members working on a separate task who had not been told about this.
* * Thus, we ended up disbanding the output layer problem team and splitting up the team. Rishit and Shiyi would start working on designing our experiment, while Jessi and Aditi went over with Devan on NNLearner2. I continued working on integrating our work into EMADE to get runs going for the time being, and Karthik joined me on this.
* NAS:
* * Had a meeting with updates, and then a 30 minute work session.
* * New members are all on PACE and ready to work.
* * Working on ADF's.
* Modularity:
* * Introduced new members to the team and what work they do.
* * Discovered another new bug, this time in the match_arl algorithm.
* * Still working on integrating Cache V2 features.
* Image Processing:
* * Having some team re assignments, especially with new members
* * Mating/Mutations sub team is trying to adapt some functionality for strongly typed genetic programming.
* * Working on some general bug fixes.


### Sub Team Meeting Notes
* At this week's sub team meeting, we did a bit of a sync up first, getting everyone on the same page.
* As the output layer team was dissolved for now, and we were minimizing the task for the time being, we split up the members amongst the remaining teams.
* Now, we just had the NNLearner2 team and the Integration Team.
* Devan had looked at NNLearner2 and the differences between standalone tree evaluator and launchEMADE's load environment methods. We decided it would be best to ignore standalone tree evaluator's for now.
* We split up, and worked on our separate tasks.
* At the end of the meeting, we decided our best bet for completing NNLearner2 would be the Hackathon. I would work on it with Devan there.
* For the Integration Team, we still worked on testing different models with the BidirectionalAttention and BidirectionalModeling layers. 


### Action Items
Task | Current Status | Date Assigned | Date Resolved | Date Due |
--- | --- | --- | --- |--- |
Retask  Members from output layer team| Complete | 11/15/2021 | 11/17/2021 | 11/17/2021 |
Further Debug New Primitives with Standalone Tree Evaluator until working or major code flaw is found | Complete | 11/15/2021 | 11/21/2021 | 11/21/2021 |
Meet w/ Devan on  status of NNLearner2 | Complete | 11/17/2021 | 11/21/2021| 11/21/2021 |
Have code ready for Code Freeze | Complete | 11/15/2021 | 11/29/2021 | 11/22/2021 -> Extended to 11/29/2021|

### Further debugging of Primitives with Standalone Tree Evaluator
* With my memory issues from last week resolved, I continued with debugging the integration of our primitives into neural_network_methods.py.
* With a better understanding now of NNLearner, I figured that an easier way to debug might be to actually build our seeded model in Keras outside of EMADE.
* I opened up the Google Collab used by Karthik, Kevin, Rishit, George, and Shiyi for debugging the building of some of the primitives in order to attempt to build the model in Keras: https://colab.research.google.com/drive/1sZfLfxzt1IF904cKh6FmRwLJQbN1nh-w?usp=sharing#scrollTo=8cVHrbqkZCxd
* When I built the model, it did go through. However, the code specifically relied on batch size being none. I noted that this wouldn't always be the case when we ran model.fit() in NNLearner and NNLearner2. Consequently, it was causing crashes.
* Furthermore, the output of bidirectional attention was in the wrong shape. The difference in shapes was (2,0) vs (2,). This was evident from my runs with standalone tree evaluator, where the individuals would be written to failedones2.txt instead of running the evaluation in the method accuracy().
* This is the code I used for debugging bidirectional attention's integration: 
* <img width="1296" alt="Screen Shot 2021-12-07 at 12 10 23 AM" src="https://github.gatech.edu/storage/user/27405/files/576a8b23-bd47-4388-adfe-bb41fbbdb7be">
* Fortunately, while doing this, I did discover that Keras's regular Attention layer could also build a model that could apply the question to the context and context to the question in a meaningful manner. Thus, we could substitute this layer in, adding more potential diversity and allowing our NNLearner2 team to make more progress.

### Hackathon
* At the Hackathon, I worked with Devan to debug NNLearner2 and get it working.
* I also introduced Karthik and George to the bugs I discovered in Bidirectional Attention, as I discovered above. I gave them my code above as a way of testing progress: a model should build with bidirectional attention as a layer instead of attention.
* I sort of swapped between talking with their team and debugging our first semester students who were attempting to Run EMADE on PACE.
* However, most of my time at the Hackathon was spent on NNLearner2.
* The first bug we found was that the load_environment method was crashing. We couldn't exactly figure out the error, as it seemed to be an issue with a lambda call, and we weren't getting much info beyond that. We talked to Dr. Zutty to resolve this one.
* After that, we moved back to working through the flow and resolving errors along the way. 
* Our next major error was with the splitting of the train and test data. The code we were given for NNLearner2 only worked on image data, and it used a method we didn't have access to to split up train and test data lists. We therefore ended up basing most of our code off of a raw workaround that looked right from the image processing code we had as a reference. NOTE: this later proved to be incorrect, as discovered later on. When calling model().fit(), input should be passed in as a list, where the first element goes into the first InputLayer(), and the second element into the second InputLayer().
* Our next issue was with the Embeddings. As the two inputs were different sizes, the tokenization of words was slightly off. So, we had to find the max vocabulary between the two inputs and use that. NOTE: Our trivial fix for this proved to be incorrect, and we had to resort to a temporary fix of multiplying the vocabulary size by a constant of ten for the smaller input vocabulary. This workaround still needs fixed.
* At the end, we pushed our code to the Github, and worked on looking over NNLearner2 until the next day, where we were running into semantic errors. That is, our code seemed built for only solving classification problems, as our output was all 1's and 0's.



## Week 12, Nov 8th
### General Meeting Notes
* NLP: 
* * Having onboarded our new members and laid out what we need to accomplish before the end of the semester, we began tasking today.
* * We used the trello board to layout tasks.
* * I began leading the integration team to workout the bugs from the big merge that remained, in addition to getting the primitives to work, and debugging them
* * Karthic began leading a team to explore our issue with the output layer
* * Devan began leading a team to debug NNLearner2
* * Kevan would look at word embeddings
* NAS:
* * Came up with some new tasks, such as automating time stopping and now working on novelty detection
* * Setup a new video for setting up EMADE locally
* * Making SQL visualization improvements
* Modularity:
* * Changing some of their overall semester goals
* * Part of the old stocks team will work on adding modularity code to the CacheV2 branch
* * Still doing experiments with ARL's.
* Image Processing:
* * Having some PACE Errors
* * PACE was offline this week
* * Fixing environment errors 

### Sub Team Meeting
* We held this week's sub team meeting in person
* With our teams fully setup, we held a work session.
* I broke off with the integration team, trying to get what we've built to actually work in EMADE.
* George was having disk memory quota issues that I helped him resolve by resetting his conda environment.
* I helped Manas get setup on PACE.
* I ran into disk memory quota issues myself, and had to delete some datasets I wasn't using anymore to resolve them.
* I ended the session with a memory issue in standalone tree evaluator. Anish later told me about setting reduceInstances to solve this, and Dr. Zutty later informed me that we actually set this memory limit in the XML file. Increasing the memory limit there solved the issue.

### Action Items
Task | Current Status | Date Assigned | Date Resolved | Date Due |
--- | --- | --- | --- |--- |
Implement new Primitives (Bidirectional Modeling, Attention, and Output) into EMADE in neural_network_methods.py | Complete | 11/08/2021 | 11/09/2021 | 11/10/2021 |
Debug Persistent issues with Merge of EMADE-304 and nn-vip | Complete | 11/08/2021 | 11/14/2021| 11/16/2021 |
Debug New Primitives such that they are called when running standalone tree evaluator | Complete | 11/08/2021 | Moved til next week | 11/16/2021 |

### Debugging Big Merge and Primitives
* This week, I finished debugging big bugs left over from the big merge. My commits can be at these links:
* * https://github.gatech.edu/sleone6/emade/tree/8cae9f7b858b69f7981e51444af932103b0a3f9c
* * https://github.gatech.edu/sleone6/emade/tree/b4187b52a653597446b880638578c52abc373fe3
* * https://github.gatech.edu/sleone6/emade/tree/cc9fa59233312a3f5e187c3d4dfc0bb4be8ef1b4
* At this point, I was able to run an individual with the Amazon dataset with the regular NNLearner, and get accurate results. As a result, all other teams were now unblocked. They pulled the changes, and each of our three teams got to work.
* With the new primitives now in the pset, I began debugging this individual with standalone tree evaluator: NNLearner(ARG0,OutputLayer(BidirectionalOutput(BidirectionalModelingLayer(BidirectionalAttentionLayer(EmbeddingLayer(100, ARG0, randomUniformWeights, InputLayer()), EmbeddingLayer(100, ARG0, randomUniformWeights, InputLayer()))))), 100, AdamOptimizer)
* It ran for quite a while, eating up about an hours worth of debugging time. At the end, it crashed with a memory error, specifying that it had exceed 24000 MB.
* I later learned two ways to resolve this from Anish and Dr. Zutty. I could set reduce instances to a value less than 1 to reduce the size of the dataset, or I could increase the number of memory allotted in the XML file. In other words, it, turned out the error was an issue with the memory I was allotting in the XML file.

### Meeting with Karthik and Rishit
* Karthik, in charge of the multiple outputs team, and I met on Saturday to talk about the output layer problem.
* In short, the problem was that NNLearner and NNLearner2 both rely on Keras to train and fit the models. However, Keras doesn't support training for two inputs. We would have to not only write our own layers for a double output, but also write customized loss functions that could handle this, for any potential layer used in neural_network_methods.py.
* This was a pretty massive problem, and not one we could likely solve within a few weeks. We consulted the rest of Karthik's team to double check, and we all agreed it would be best to reduce the scope of the problem for now. We could change it to a pure regression problem, only trying to find the start index of the answer. This is similar to Google Search's answer feature, finding the answer within a document.
* EXAMPLE:
* * Question: When did the Titanic Sink?
* * Context: The titanic was a big ship. It sank in 1912. It hit an iceberg.
* * Output: 8, the index of "in" as in "in 1912".


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
Complete New Primitive (Bidirectional Modeling, Attention, and Output) Development outside of EMADE | Complete | 11/01/2021 | 10/27/2021 | 11/08/2021 |
Prepare Presentations for New Members | Complete | 11/01/2021 | 11/02/2021 | 11/03/2021 |
Debug New Primitives with Standalone Tree Evaluator | Complete | 11/01/2021 | 11/08/2021 | 11/08/2021 |

### Preparing Presentations for New Members
* I modified last year's PowerPoints to inform the new members about deep learning and NLP.
* The presentations I made can be found here: 
* * Deep Learning and NLP overview: https://docs.google.com/presentation/d/10R9PEClxceFbCFclzHBMbZSowvBlLb1-0zKQ8haParc/edit?usp=sharing
* * NNLearner overview: https://docs.google.com/presentation/d/17KfH14LP1ToNQA1u4b8WiWEcwr9bDY6KNd0Xws8iOdk/edit?usp=sharing

### Debugging Primitives with Standalone Tree Evaluator
* I began using Standalone Tree Evaluator this week to debug the primitives we'd made. Most of the code was finished out by Karthik and Kevan, but they had only tested it in a Collab notebook, not in EMADE. 
* Initially, I was obtaining issues with malformed individual strings. I used the error messages to debug this.
* Upon digging deeper, I discovered that the error message was a result of trying to turn a layer (BidirectionalAttention) into a literal. I asked in the slack about this, and discovered we had to edit gp_framework_helper.py.
* IMPORTANT FUTURE NOTE FOR ADDING PRIMITVES: all new primitives added have to be added to the set in gp_framework_helper.py. This is an example, which is the code I wrote to add our primitives, at the end of the addPrimitives() method:
* 
<img width="971" alt="Screen Shot 2021-12-06 at 10 18 19 PM" src="https://github.gatech.edu/storage/user/27405/files/d38b4082-b592-43a1-af5c-6f88ecd46455">
* However, even after adding this in, we still couldn't exactly complete runs. There were a couple discrepancies throughout this branch, like data pairs being passed in as a list. These were also fixed, though some of them bled into the next week.

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
Assign Primitive Development tasks to all team members | Complete | 10/25/2021 | 10/27/2021 | 11/01/2021 |
Email Dr. Zutty about 2 Data Pair NNLearners | Complete | 10/25/2021 | 10/30/2021 | 11/01/2021 |
Implement Bidirectional Attention Layer | Complete | 10/25/2021 | 11/07/2021 | 11/08/2021 |


### Implementing Bidirectional Attention Layer: Part 1
* Devan and I met on Friday to work on the Bidirectional Attention Layer this week
* As neither of has had experience implementing primitives prior to this, we first spent a lot of time reading through neural_network_methods.py to discover how to implement a new primitive.
* I discovered that NNLearner uses a new class defined in the file, called a LayerList, to keep track of the layers. Then, the NNLearner function iterates through these layers to build the deep neural network.
* So, if we wanted to add a layer, we would have to create a subclass of Keras.layers.Layer, and override the build() and call() functions.
* While I was doing this, Devan found a template from someone else who implemented BiDAF. It wasn't split into layers, but we could use it as a reference for learning keras and TensorFlow.
* After going through Layerlist, I wrote the primitives that would add these subclass lists to the layer list.
* <img width="713" alt="Screen Shot 2021-12-06 at 10 00 02 PM" src="https://github.gatech.edu/storage/user/27405/files/6959c0f7-94b0-4a32-944d-7a92caf5b297">
* After a little over an hour, and with the skeleton of the Bidirectional Attention Layer done, we finished up work for the time being, until Sunday. 
* <img width="481" alt="Screen Shot 2021-12-06 at 10 02 39 PM" src="https://github.gatech.edu/storage/user/27405/files/20a60fe9-ea61-45a9-9a1b-368d13d54a67">

### Implementing Bidirectional Attention Layer: Part 2
* I met with Kevin, Rishit, and Karthik on Sunday this week to work on the Bidirectional Attention Layer.
* We didn't make much progress due to a slightly different team make up than the Friday meeting, but I filled them in on how adding primitives to NNLearner's worked.
* I uploaded the most recent code to the slack (screenshot in Part 1 above), and some of them worked on it later this week.



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

### General Debugging
* I made some more commits with general bug fixes:
<img width="1266" alt="Screen Shot 2021-12-06 at 10 24 29 PM" src="https://github.gatech.edu/storage/user/27405/files/84422596-546c-4cc0-81ec-e6119efc888d">

## Week 8, Oct 11th
### General Meeting:
* No General Meeting this week

### Sub Team Meeting
* At our Sub Team meeting, we began looking at the EMADE-304 branch and deciding how best to merge it over.
* We planned to use the Hackathon to finish merging EMADE-304 over.
* We would look over the code over the next 2 days, and then have a meeting on Friday to discuss our thoughts.

### Friday Meeting
* I put together a list of trivial vs non-trivial merge conflicts: https://docs.google.com/document/d/1B-0uHdawDfCY-5BLZQUiuz-dleC9MDwqVhYU6WmliRc/edit?usp=sharing
* We discussed it, moved files around, and decided to use it as a basis for which files to look at first on Saturday.

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

### Make an EMADE Branch with new CacheV2 and NNLearner functionality
This is our fork, which we would merge CacheV2 (EMADE-304) with nn-vip: https://github.gatech.edu/sleone6/emade/tree/EMADE-304-allow-cachev2-to-consume-aligned-datapairs

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
Research creating a new type of data pair in EMADE | Complete | 09/20/2021 | 09/27/2021  | 10/03/2021 |


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