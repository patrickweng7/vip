= Spring 2021 =

== Explanation ==
Due to size issues with my notebook and taking too long to render, Spring and Fall 2021 notebook entries have been moved to their own separate notebooks. The following entry is for Spring 2021.



== Week 14: April 26, 2021 ==

=== General Meeting Notes ===
*NLP/NN
**Cameron pushed a large update to the Git
**Generations run much faster now
*Modularity
**Have some new objectives, doing 3 experiments
**Week is mostly focused on runs
**Using precision and recall on them
**Using 3 objectives, unsure if DEAP will still work w/ hyper volume; they will flatten to 2d to evaluate.
*EZGCP
**Did work on researching primitives
**scraped some common pre trained layers
**visualized them all to see which ones would be a priority to implement
**Adding batch normalization
**Mating can be very destructive, so not usually used w/ cartesian
*Stocks
**For the past week, they did a run w/ emade using new technical indicators
**Modified some hyper parameters in EMADE for this run; ran it for longer
**One individual outperformed seeded individual by a considerable amount
**That individual outperforms the mean by a considerable amount
**No other individuals performed this well.
**Will be using seeding functions that first semesters wrote
=== Sub Team Meeting ===
*Cameron put out a rather large new update
*I showed the large improvement we got from seeded mediocre individuals
*We'll meet on Wednesday to discuss our final presentation

=== Friday Presentations ===
*Stocks
**Main goal: look at how EMADE can optimize Stock Trading
**Semester Objectives: Implement TA-Lib Indicators, increase "evolvability" of EMADE individuals, test on larger datasets, analyze specific individuals, work on objective functions and evolutionary parameters, implement technical indicators researched
**Based methodology off of a paper. It discussed Piecewise Linear Representation and Exponential Smoothing.
**Created 3 main subgroups and general interest areas: Literature Review and Research, Data Analysis of runs, EMADE implementation
**Groups were not mutually exclusive
**Removed TriState and Axis parameters from TI primitives
**TA-Lib makes adding primitives much easier
**Wrote new technical indicators
**New indicators: Volume Weighted Moving Average, Volume Weighted Average Price, Fibonacci Retracement
**Also implemented Money flow index, Chaikin money flow, and Klinger volume oscillator
**Run 1 Results with 4 objectives, 328 generations:
***AUC: 0.002915
***Pareto Fronts seem to decrease, especially on var profit per transaction
**Notable Individual: the best performing one was a variant of MyBollingerBand
**Another Notable Individual: wasn't as well in performance, but not as volatile
**Team is utilizing Monte Carlo Simulations
**Individuals are consistently above average, and more than just in terms of profit
*EZGCP
**Refresher: EZGCP uses a graph based instead of tree based structure
**Midterm Recap: Removed augmentation and preprocessing
**Midterm Accuracy: Training had high accuracy, but validation was low (overfitting)
**Objectives: Recreate similar results on CIFAR-10 without relying on transfer learning, improve ability to visualize genomes, research and make new mating methods for cartesian GP
**Results: Analysis showed that individuals produced matched the target distribution
**Deeper individuals were required to produce results on CIFAR-10
**Finished max pooling, average pooling, and dropout
**After 50 generations, they reached 68.5% (compared to 56.3% without the primitives)
**Experimented with the introduction of dense layers after convolutional layers; had issues with low diversity and poor performance compared to SOTA and transfer learning.
**Read Cartesian Genetic Programming Paper to improve EZGCP
**Ran regression defined on 4 objectives as defined in the paper (Koza-3, Nguyen-4, Nguyen-7, Pagie-1)
*NLP/NN
**We presented our findings with regards to evolving complex individuals on the Amazon Dataset this semester.
*Modularity
**Overview: Background, Experiments, Analysis
**Modularity: exploring ways to abstract parts of individuals
**This allows for creating "building blocks"
**ARL is Adaptive Representation through learning
**Previously, searched the population for combinations of parent and children nodes
**Ongoing project: increase complexity of ARLs. 
**Edited multiple methods to deal with increased depth of ARLS
**Experiment setup: Titanic dataset had feature data, MNIST had stream
**Potential objectives: Precision, Recall, F1, Accuracy, Cohen Kappa Score
**30 Seeded individuals in MNIST had an F1 and accuracy score < 0.1.
**Would like to change seeding file individuals for MNIST

====Lessons from Presentation====
*The presentation was stopped, and questions were asked about my slide comparing improvements in accuracy when seeded individuals had mediocre accuracy (70%) versus accuracy near our baseline model (91%).
*The key takeaway was that marginal improvements in accuracy were due to seeds already having very high accuracy (low error to minimize) yet high number of parameters (which left a lot to minimize), as opposed to an error existing in EMADE or the NNLearner. This was not conveyed well.
*Ideas to improve this could have been showing the accuracies directly on the Pareto Front. Having a key takeaway note on the slide as opposed to using space to show one of the individuals may have better conveyed this as well.

=== Action Items ===
{| class="wikitable"
!Task
!Current Status
!Date Assigned
!Date Resolved
!Date Due
|-
|Pull Cameron's update
|Complete
|04/26/2021
|04/27/2021
|04/30/2021
|-
|Get a False Positive False Negative Run in w/ updated branch
|Complete
|04/26/2021
|04/28/2021
|04/30/2021
|-
|Analyze Runs and Make Visualizations
|Complete
|04/28/2021
|04/29/2021
|04/29/2021
|}

===Emade Runs===
*We wanted to get a decent sample of runs using false positive rate and false negative rate as the objectives in order to get some confidence intervals on the runs so that we could conclude that our final individuals improved from our seeded individuals. Having PACE setup, I had to get one of these runs in.
*First, I pulled the changes Cameron made on GitHub. This required taking the files from Github and using scp to transfer them to PACE as the .git folder had to be removed due to disk quote issues.
*After running bash reinstall.sh, I then had to change the input.xml file to accommodate the new changes. The biggest difference was adding pace path.
*By the time I got to runs, however, PACE was overloaded, so MySQL was not working. I asked about it on the slack, and Cameron told me he had run into MySQL Error: 2003 as well before. It happens when too many people are on pace, so a different host is used for MySQL. Thus, I had to run stat -n to find the name of the host, and change it in the input.xml file. I added it to the MySQL common errors documentation, at the top of my notebook.
*I was then able to successfully run EMADE using FPR and FNR. I uploaded my runs to the "Runs Results" google doc. Across 3 8 hour submissions to PACE, it reached generation 40. 

===Presentation Work===
*We met on Wednesday to talk about our presentation, and did a rehearsal on Thursday.
*I was tasked with 2 slides.
**1) A slide explaining our difficulties with PACE, and how we future proofed ourselves to prevent having such difficulties again.
**2) A slide explaining our initial run results on PACE, and how we had only marginal improvements in accuracy due to the high accuracy of our seeds, which already almost reached the baseline model Sumit made using fastext.
*The first slide was easy, as we had already compiled much of our PACE issues for our midterm presentation. I shortened them, splitting them into two categories (EMADE and PACE errors). Harrison would discuss the MySQL issues we've documented.
*I created better visualizations of the Pareto Front for the individuals resulting from 90% accuracy achieving seeds versus individuals resulting from 70% accuracy achieving seeds. These visualizations are shown below.
*On a side note, these visualizations are much better than the ones I manually graphed for my notebook, and convey the details, such as the actual Pareto Front using the steps, very well. I will use python and matplotlib to graph from now on, as opposed to excel.
*I also looked into the master.out files and added area under the curve values.
*As an update, I should use accuracy next time, as that was the metric I was elaborating on, as opposed to area under the curve.

== Week 13: April 19, 2021 ==

=== General Meeting Notes ===
*NLP/NN
**We presented our progress; Most members are on PACE
**We're making progress w/ getting individuals with some complexity
*EZGCP
**Mating team finished last week's paper, working on benchmarks
**Dense layers had a best accuracy of 55%
**Visualization team added Parameters
*Modularity
**ARL's are properly being created and added to the database
**Everyone is tasked on the team of restructuring the database
**Getting things ready for the final presentation
*Stocks
**Considering adding more datasets
**Plan to do an Emade Run to use newly made functions

=== Sub Team Meeting ===
*We discussed results about my most recent run
*Hua's runs are stock on Generation 1
*Moving to using false positives and false negatives instead of accuracy/num parameters may be the best move.

=== Friday Meeting ===
*Discussed my mediocre individuals run and improvement
*Will begin shifting focus to how to present our findings this semester best in our presentation

=== Action Items ===
{| class="wikitable"
!Task
!Current Status
!Date Assigned
!Date Resolved
!Date Due
|-
|Complete and Analyze Mediocre Seeded Individuals Run
|Completed
|04/19/2021
|04/25/2021
|04/26/2021
|}

=== EMADE Run on PACE ===
*With my mediocre individuals seeded, I then ran EMADE again to see how close it could get to our benchmarked results that Sumit and Anshul produced (92%).
*The final Pareto individuals (blue) are shown against the seeded (red) individuals below. Note that some red, particularly on the number of parameters axis, can't be seen as they are directly below the blue.

*The best results were these two individuals
**NNLearner(ARG0, OutputLayer(EmbeddingLayer(5, ARG0, glorotUniformWeights, InputLayer())), myIntMult(falseBool, 3), NadamOptimizer)
**NNLearner(ARG0, OutputLayer(EmbeddingLayer(-3, ARG0, passWeightInitializer(passWeightInitializer(heWeights)), InputLayer())), 128, RMSpropOptimizer)
*This individual had a bit above 89% accuracy. It wasn't Pareto at the end, but it is rather complex:
**NNLearner(HighpassIrst(ContourMaskMinEquDiameter(ARG0, TriState.STREAM_TO_FEATURES, Axis.AXIS_1, 10), passTriState(TriState.STREAM_TO_STREAM), passAxis(Axis.AXIS_1)), OutputLayer(EmbeddingLayer(myIntMult(6, 6), ARG0, heWeights, InputLayer())), -4, RMSpropOptimizer)
*Both score between 89% and 90% accuracy, compared to the 66% accuracy that our seeded individuals offered us.
*The difference is quite significant. This leads me to believe that we aren't having a problem with obtaining complex individuals, especially as we've reached just 2% shy of the benchmarked best possible solution we could get. Else, we would have had trouble getting past 66%. Furthermore, the individuals EMADE produced are quite complex. This would lead me to believe that, last semester, the lack of complexity was due to num parameters being an objective. With a data set split 95 to 5, it's quite likely that a simple learner could still score extremely high on accuracy, while, due to its lack of complexity, score really well on num parameters as well. This, however, only has the good results of Amazon to evidence this conclusion; more analysis may be needed.

== Week 12: April 12, 2021 ==

=== General Meeting Notes ===
*NLP/NN
**Cameron has runs that we will look at in the general meeting
**We've laid out tasks for people to work on, and assigned them
*Stocks
**Still onboarding first semesters
**Random experiments found that they were doing much better, other than on hypervolume
*EZGCP
**New members are working on visualization and mating
**Added average pooling, hardcoded some dense layers
*Modularity
**Made progress on the depth problem (functions being found, lambdas created)
**Added more members to database restructuring task

=== Sub Team Meeting ===
*Cameron and I both finished rather large runs that we analyzed in the meeting.
*Our results (visualized in my work below) were not too different from the seeded individuals, which was quite disappointing.
*Dr. Rohling commented on our discussion.
*He noted that we had to visualize our seeded individuals on the Pareto Front as well. It would be one thing if these individuals evolved from 60%, but another from the 92% individuals.
*Most of our new members all have PACE setup now.
*Anshul and Sumit are essentially done as well.

=== Friday Meeting ===
*We wrapped up tasking people via the google doc.
*Harris, Nishant, and Karthik are exploring how to query the database and how individuals are stored by EMADE in MySQL.
*I will continue trying to find bad individuals.

=== Action Items ===
{| class="wikitable"
!Task
!Current Status
!Date Assigned
!Date Resolved
!Date Due
|-
|Seed Amazon w/ less accurate individuals
|Completed
|04/12/2021
|04/16/2021
|04/16/2021
|-
|Look into potential new tasks
|Completed
|04/12/2021
|04/16/2021
|04/16/2021
|}

=== EMADE Run on PACE ===
*To get around the 8 hour PACE wall time, I changed the location the database was hosted on to localhost:3307, so that I could run EMADE from the terminal. Thus, as long as my laptop didn't sleep, EMADE could keep going. I used the Mac command "caffeinated" in a different terminal to accomplish this.
*In total, this second run ran for 24 hours, seeded with 10 individuals, all similar but slightly changed from the single NNLearner previously seeded. The Hypervolume here was 70360579.5, an improvement from the last run. There were 6 Pareto individuals, shown on the Pareto Front below.


*These results are noticeably similar to the first Amazon Run. Disappointingly, the best individual regarding accuracy is only slightly different from the seeded one. This best individual is '''NNLearner(ARG0, OutputLayer(LSTMLayer(16, defaultActivation, 0, trueBool, trueBool, GRULayer(150, eluActivation, 50, trueBool, trueBool, EmbeddingLayer(-4, ARG0, heWeights, InputLayer())))), 89, RMSpropOptimizer)'''. The number of parameters is improved though, suggesting that it was slightly optimized along this end.
*The number of parameters Pareto Front is much more fleshed out in this run. While the ones with only around 300 parameters are trivial solutions (an accuracy of about 50%, essentially guessing all positive or all negative each time), they at least explore this end of the Pareto Front.
*One of the reasons why the seeded individual is not being surpassed in accuracy is that this may be as accurate as the individual can easily get. The benchmark Sumit and Anshul had was about 93% accuracy. Therefore, if we seed an individual that already gets extremely close to this number, it will be incredibly hard to improve upon it. Seeding less accurate individuals might be a better way to see if EMADE can produce complex individuals.
*Through the meeting, it was also discussed with Dr. Rohling that it would be nice to see the seeded Pareto Front on the same chart. We also discussed how it would drastically change things if seeded individuals were already performing as well as the results.
*Consequently, I decided to begin looking into new individuals to seed that would not be already performing at near maximum accuracy.
*To start, I began a non seeded run. This should weed out non NNLearners fairly quickly, and get some poor performing NNLearners at first before reaching maximum accuracy.
*Most of the individuals produced here were fairly trivial, only reaching about 50% accuracy in an evenly split binary classification problem.
*One individual, NNLearner(ARG0, OutputLayer(EmbeddingLayer(6, ARG0, heWeights, InputLayer())), myFloatToInt(100.0), AdagradOptimizer), received about 66% accuracy. This would do nicely for the problem of needing mediocre individuals. I began testing variants of this individual to seed.

=== Looking into New Tasks ===
*I checked the tasks list, and noted that the analysis and getting EMADE to work on PACE tasks that I've been doing fell under the "Evolutionary" category of new tasks. I marked myself down to work on this, so that I could continue working on the same problem I've been working on all semester.

== Week 11: April 5, 2021 ==

=== General Meeting Notes ===
*Stocks
**Continued with their onboarding process for first semester students
**finished all Talibans methods, but final optimizations are not done yet
*NLP/NN
**I am working on resolving final errors
**Cameron discovered the problems with PACE
**Anshul will give a presentation on Neural Networks
*Modularity
**Still benchmarking MNIST
**Still working on the depth problem
**Will do Analysis the rest of the semester
*EZGCP
**Acheived a run, obtained 68% accuracy
**Had just one individual with only small mutations at the end of a couple generations

=== Sub Team Meeting ===
*We worked on getting people setup on PACE
*The issue with transformers keeps popping up, despite it being in the YML file. This is rather strange; when extra time exists, we'll have to look into it.

=== Friday Meeting ===
*Anshul gave a presentation on neural networks
*Karthik made a script to facilitate ease of initial tasks when logging into PACE
*I wrote my script to divide the dataset, and will get a long run in over the weekend

===Action Items===
{| class="wikitable"
!Task
!Current Status
!Date Assigned
!Date Resolved
!Date Due
|-
|Resolve Large Dataset Issue
|In Progress
|04/05/2021
|04/09/2021
|04/09/2021
|-
|Resolve NNLearner Code Issue
|Complete
|04/05/2021
|04/09/2021
|04/09/2021
|-
|Discover why NNLearners all Fail
|Complete
|04/05/2021
| 04/09/2021
|04/09/2021
|}

=== EMADE Run on PACE ===

====Resolving Final Errors====
*Having discussed the errors with NNLearner and the Amazon dataset on Friday, I worked to resolve these issues.
*Cameron already resolved the bug in the NNLearner that he found. I pulled this updated file and ran "bash reinstall.sh" to fix this.
*Now, the final issue was the size of the Amazon dataset. As there were 3.6 million examples in the training file, it was 20 times the size of toxicity, and was consequently crashing PACE, which made evaluating NNLearners fail every time. This is why every NNLearner had a fitness of (infinity, infinity) during the runs, even known good seeded ones.
*My initial plan to resolve this was to use K-folds, however, instead of partitioning the same dataset into different train and test splits, I would divide the dataset up into 20 evenly split groups. Then, I could enter the datasets into the XML file the same way as K-folds are used in the input_titanic.xml file.
*First, I loaded in arrays from the .npz test and train files, as this was the only way to work with non-utf8 characters. I ran into an issue with the default settings of bumpy, however, shown below
*I resolved this by saving numpy's default load settings to change back to at the end, and then inserting this line of code before loading the .npz files: "np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)"
*This code changed the value allow_pickle to true, fixing the error. Below, I've shown the code that fixes this error, and successfully loads in the data.
*I don't have permission to push to Github, so I gave the script and results to Cameron. He took one of the splits and pushed it to Github, which can be found here: https://github.gatech.edu/emade/emade/tree/nn-vip/datasets/amazon
*Having resolved this error, I proceeded to writing a script performed as intended, dividing up the dataset and saving it to 20 test and train csv files.
*You can download this script here https://drive.google.com/file/d/1Wdv3BQwyBsQDLfyjGOovei1xjDaOoF2u/view?usp=sharing
*I discovered we couldn't use folds for the dataset in this branch
*I resorted to only use a dataset that was 1/20th of the size
*The training data was now a 49.3/50.6 split, which should be good enough still to get non-trivial results.

====Running EMADE, results analysis====
*With the NNLearner fixed and data set divided properly, I could now do runs that yielded meaningful results.
*Thus, I started my first run. I seeded one NNLearner, set wall time on PACE to the max (8 hours), and let it run.
*It hit 8 hours without any errors. In total, it ran for 15 generations, had 3 Pareto individuals, and a hyper volume of 71355488.6. The results are shown in the graph below.
*The results are rather disappointing. There's a large amount of unexplored territory on the Pareto Front, especially on the number of parameters axis. Furthermore, the individual with the most accuracy is identical to the seeded individual. This individual, with an accuracy of 92.9%, was NNLearner(ARG0, OutputLayer(DenseLayer(10, defaultActivation, 10, LSTMLayer(16, defaultActivation, 0, trueBool, trueBool, EmbeddingLayer(100, ARG0, randomUniformWeights, InputLayer())))), 100, AdamOptimizer).
*Other results that appeared did work well, but none surpassed this seeded individual in accuracy. This was of concern, as EMADE was unable to improve upon a seeded individual in regards to accuracy. Other individuals had close to as high accuracy and less parameters, but none scored as high on accuracy.

== Week 10: March 29, 2021 ==

=== General Meeting Notes ===
*Modularity
**Still testing against the depth problem
**Multiclass False positives and false negatives may be outdated
*NN/NLP
**We've decided that an all out attempt to get everyone setup on PACE was best for us
**Sumit is also moving from benchmarking to getting set up on PACE
**Based off our presentation, we'll be all out focusing on getting non-trivial results
*Stocks
**Doing General Onboarding with members
*EZGCP
**Working on adding dropping and max pool layers
**Also working on onboarding new members

=== Sub Team Meeting ===
*Most of this meeting was spent acquainting new members with our branch and goal for the semester
*I compiled several links needed to get PACE up and running.
*I pasted them into the slack and BlueJeans call.
*I also gave an update on my progress w/ Amazon Dataset.
*All members were to start trying to get PACE set up as soon as the wiki was available again.

=== Friday Meeting ===
*I gave an update on my progress. Having run into logical errors in the code, I was doing a deep dive into EMADE to understand what was going wrong.
*Cameron actually already found two issues.
**The first was a bug in NNLearner, which he pushed a fix to Github for
**The second was that the dataset was too big
*He talked about a couple solutions for the size of the dataset, such as changing the padding size for text, which was well larger than needed.
*I suggested splitting up the data and using it the same way K-folds were used.
*After that, he gave a brief presentation on EMADE to the first semesters.

===Action Items===
{| class="wikitable"
!Task
!Current Status
!Date Assigned
!Date Resolved
!Date Due
|-
|Do a Seeded Run on Amazon
|Complete
|03/29/2021
| 03/29/2021
|03/29/2021
|-
|Familiarize with EMADE code
|Complete
|03/29/2021
|04/01/2021
|04/03/2021
|-
|Discover why NNLearners all Fail
|Complete
|03/29/2021
| 04/01/2021
|04/03/2021
|}

=== EMADE Run on PACE ===
*With the code working, I was able to get a seeded run in.
*After 6 hours, the run hit wall time on PACE without any errors.
*However, the NNLearners were all still failing.
*I began to use standalone_tree_evaluator.py to run an NNLearner individually, and see results.
*Just like in the full runs, the NNLearners were returning infinity on both objectives, accuracy and number of parameters. This was confusing; even if accuracy had the worst score, there was definitely a finite number of parameters, well below the maximum defined by the XML file.
*As a result, I started to do a deep-dive into the code base for EMADE, to see what was going on.

== Week 9: March 22, 2021 ==

=== General Meeting/Presentation Notes ===
* Stocks
** Overview: How can we use technical indicators to predict specific stock trends and buy/sell signals
** Previous semester had an inconsistent research paper, so they found a new one
** Writing more Technical Indicators
** Using the same stocks, except for EPISTAR as they can't find it
** ML Model is used to predict trading signals
** Technical issues include deciphering implementation details (such as whether or not to use a Genetic algorithm) 
** Just did an EMADE run, ran for roughly 6 hours
** Some transactions resulted in a loss. This is likely due to lag.
* Modularity
** Exploring ways to abstract parts of individuals
** allows for "building blocks" to help the process"
** ARL: adaptive representation through learning (thus, when a good part of a tree is found, it can become a new primitive).
** Using Sphinx this semester to generate documentation
** New method: searching an individual (takes in an individual ARL candidate, traverses node in order).
** Making progress on all subtrees
* NLP
** We presented our work for the semester
* EZGCP
** "easy cartesian genetic programming": graph based instead of tree based
** Using a block structure, so block only has data of a particular type
** Each block has its own set of mating and mutating strategies
** Primary goal for last semester was using the Block Structure on CIFAR-10 dataset
** They were able to successfully evolve with existing pre-trained model followed by transfer learning
** Exploring Transformers and RNN
** Pretrained and seeded models were likely redundant with Transfer learning, which they are trying to move away from
* Bootcamp 1
** Used EMADE to predict who survived the Titanic
** Used univariate selection, feature importance, and correlation matrix with heatmap to pre-process data
** Used 7 different ML Models
** Minimized squared sum of FN and FP and maximized squared sum of tp^2 and tf^2
** Had to turn off firewall to allow connections (this doesn't seem right from my POV view though... were they not using a stateful firewall? Why wasn't an entry added to their Access Control List? Not relevant exactly to the topic, just caught my interest).
* Bootcamp 2
** Used one hot encoding
** dropped cabin column
** used "regex" to extract names
** dropped embarked column
** Used 6 models, including xboost, SVM, RF, nn, and logistic regression
** Best individual primitive tree performed best toward FPR
** Ran EMADE for 20 generations
* Bootcamp 3
** Genetic Programming performed the best, followed by EMADE then ML
* Bootcamp 4
** Again, EMADE was outperformed by Genetic Programming
* Bootcamp 5
** Preprocessed
** Used KNN, Neural Networks, and decision tree
** Only difference in their pre-processing is that they normalized individuals before putting them in (normalized entire dataset)
** Had a very good AUC on Genetic Programming Runs, while ML only had an AUC on .2379
** On EMADE, they ran for only 4 hours, with 37 generations, and had an AUC of .2374
=== Friday Meeting ===
* Having presented our work, we decided on our best path forward for the rest of the semester
* We decided that everything we do should focus on our overall goal of fixing any error in the NNLearner
* I gave my update on the run I did, running into the empty sequence error. I was informed that this issue had actually been resolved via code changes that I had to update, as detailed below in this week's work.
* We decided that, to make any meaningful progress, we would have to get everyone set up on PACE.
{| class="wikitable"
!Task
!Current Status
!Date Assigned
!Date Resolved
!Date Due
|-
|Do a Seeded Run on Amazon
|In Progress
|03/12/2021
|
|03/29/2021
|-
|SCP new NNLearner Code from Git
|Complete
|03/26/2021
|03/27/2021
|03/29/2021
|}

=== EMADE Run on PACE ===
* Having presented our work, I moved forward with trying to debug the error I was getting (the one in remove_layer, detailed in the previous week). 
* The first thing I tried was creating a new environment from scratch; this lead to nowhere.
* I tried re-seeding, after deleting all individuals from my database. I did this using the "drop table individuals" command in MySQL.
* This seems to have worked; I got up to Generation 6 without error. Therefore, I think this was definitely the result of a primitive or some part of an individual, which didn't always pop up in runs. Below I've documented a learner that I was getting this issue with. In the near future I'll have to further investigate this to see exactly why it's giving this issue. For now, the main focus is on trying to successfully get complex individuals on the Amazon Dataset with EMADE.
** Individual that crashed the run: '''''NNLearner(ContourMaskMinEnclosingCircle(ARG0, TriState.STREAM_TO_STREAM, Axis.AXIS_2, 0.01), [], greaterThan(10.0, -1.98222821704125), passOptimizer(FtrlOptimizer))'''''
* Instead, I got the error shown below during Generation 6.
* I remembered that Cameron and Anish had run into this issue. From reading the slack, I though it was an issue with the Pre-trained Embedding layer.
* However, on Friday, I learned that it was actually a result of errors in the code. Changes had been pushed to Github. Changes were made in emade_operators.py and eval_methods.py. I copied these over into PACE using scp and ran them.

== Week 8: March 15, 2021 ==

=== General Meeting Notes ===
* Stocks
** Ran EMADE using new primitives and new pipeline
** results were interesting; not good
** a lot of individuals didn't evaluate properly
** a lot of errors; only 2 or 3 valid individuals over the course of the entire run (even after seeding of 20-30 individuals)
** There are likely problems with the code
* Modularity
** Wanna schedule a practice presentation this week
** nearly done testing regarding the depth problem
** some trees are taking a little longer to generate
* NLP
** Cameron and I have an issue w/ seeding
** Sumit has a benchmark from Fasttext
* EZGCP
** Got Benchmark results, working on improving those results
** added a few more primitives
** sometimes getting a shape mismatch
*We all need to prepare for presentations, midterm grades should be out by tomorrow night.
=== Sub Team Meeting ===
* Sumit went and ran his code on a Google Collab and encountered the same error as before
* We're unsure of how else to proceed regarding getting a baseline that doesn't use Fasttext.
* Cameron and I are encountering a MYSQL error when attempting to seed
* As it turns out, the seeding file requires us to run "qsub mysql.pbs" prior to seeding. This is because the socket can't be used in the terminal environment, it has to be submitted as a job to do.
* We will all begin going through the powerpoint this week and filling it in.
=== Friday Meeting ===
* I gave an update on my progress; I seeded individuals successfully, but, unfortunately, I was also getting errors when these individuals were being mutated
* Anish and Cameron weren't getting this issue
* We spent the majority of the meeting then detailing our presentation and determine how to present our work this semester, as detailed at the bottom of this week's work.
{| class="wikitable"
!Task
!Current Status
!Date Assigned
!Date Resolved
!Date Due
|-
|Do a Seeded Run on Amazon
|In Progress
|03/12/2021
|
|03/29/2021
|-
|Resolve Database Errors with Seeding
|Complete
|03/12/2021
|03/19/2021
|03/19/2021
|-
|Resolve New Code Errors with Seeding
|Complete
|03/12/2021
|03/19/2021
|03/19/2021
|-
|Presentation Prep
|Complete
|03/15/2021
|03/20/2021
|03/20/2021
|}

=== EMADE Run on PACE ===

==== Seeding File/Database Problems ====
* I started off by trying to simply seed the individuals, running the command " python3 src/GPFramework/seeding_from_file.py templates/input_amzn.xml seeding_test_toxicity". However, this kept giving an error, saying that MYSQL socket couldn't be reached.
* Thanks to instructions from Cameron, I learned that the environment qsubbed processes run in is different from the ones run from the terminal. Therefore, while the socket could be reached from the EMADE program when running from the terminal, it was a completely different story regarding any qsubbed processes. Thus, the seeding program would need a different way to connect.
* I had to move the database to run on "atl1-1-02-012-5-l:3307" (atl1-1-02-012-5-l using port 3307) in order for it to be reachable during seeding. 
* Furthermore, this database had to be run as a submitted job using qsub. However, doing so meant that mysql could no longer be accessed from the terminal.
** This required two different ways of running the server: one to access from the terminal, and one that EMADE programs could access.
** To run one from the terminal, still do the old way, running "cd /usr" and then "mysqld_safe --datadir='/storage/home/hpaceice1/sleone6/scratch/db'" inside of this directory (Note you must change sleone6 if a different user)
*** This is important for granting privileges to users and viewing individuals, paretofront, and statistics tables
** To run one that EMADE can access, put that same command in a .pbs file called "pbsmysql.pbs" and run "qsub pbsmysql.pbs" (the name doesn't have to be "pbsmysql"). This way, there is a database server running that your seeding file can access.
* As a result of changing how the database is run, I had to also run EMADE inside of a submitted job via qsub. This requires the command "qsub launchEMADE.pbs". You can view the output using "vim emade-amzn.xml"
* Seeding also required the deletion of all other tables in the MYSQL database.

==== Seeded Individual Problems ====
* Initially, I was using the file "seeding_test_toxicity" to seed. However, this had several errors, such as malformed strings. Even after fixing some typos (like RG0 into ARG0), it still didn't quite work. Consequently, I switched to "seeding_test_toxicity_empty", seeding it with individuals such as "NNLearner(ARG0, OutputLayer(ARG0, EmbeddingLayer(100, ARG0, randomUniformWeights, InputLayer(ARG0))), 87, AdamOptimizer)".
* Seeding with "seeding_test_toxicity_empty" proved successful; the run was seeded, and could proceed as normal, with the aforementioned changes from the previous section.

==== DEAP searchSubTree Error ====
* Previously, in the non-seeded run, I encountered this error. However, upon my first seeded-run, I encountered this error in Gen 2, considerably earlier on in the process.
* The fact that it occurred much earlier this time around makes me think this is a problem in more developed mutation functions. As the run is seeded with more developed individuals earlier on, it's causing this error to pop up.
* I looked up the source code of DEAP to try to resolve this issue. SearchSubTree is shown below
* As shown in notebook work the week after this, I temporarily resolved this issue by deleting all of the individuals from the database. This definitely means it was a certain part of an individual that was causing this error. 

==== Presentation Work ====
* We created presentation slides to present our work to first year students.
* First, we explained our team, and how the NNLearner works
* We went into detail about how our problem this semester, and how using a balanced dataset will help show if we have a problem making trivial individuals.
* We documented our errors on PACE and how we've resolved them.
* We also added slides detailing our benchmarking, to see if EMADE could at least match these results.
* We then divided up the slides which we would present on.

== Week 7: March 08, 2021 ==

=== General Meeting Notes ===
*Modularity
**Got Feedback on Sphinx documentation
**Designing architecture for database storage
*EZGCP
**Resolved their pipeline bugs
**Got an 8 hour run with 7 generations in
**Planning on visualizing their results with a Pareto Front 
*Stocks
**The team is confused as to the purpose of genetic algorithms in the paper they're reading
**Trying to wrap up TI primitives so they can get a run in soon
*NLP/NN
**Considering looking into setting up Pytorch
**Collab Issues are persisting for Sumit
**Cameron and I both have PACE Instances set up

=== Sub Team Meeting ===
*It was decided that a universal .yml file was necessary
*Collab is disconnecting; asking stocks team for help as they've apparently had experience with this
*JG is finishing up his documentation

=== Friday Meeting ===
* Sumit successfully ran a Fastpace control run w/ Amazon dataset
* I ran EMADE for 6 hours and 68 generations before it crashed. Anish suggested doing a seeded run as opposed to a non seeded run
* The error I'm running into seems to be an issue with DEAP, not EMADE
* Cameron is having an issue getting past Gen 0; Anish walked him through fixing it by seeding w/ an NNlearner to avoid the issue with the regular learner
* John has been looking at pytorch vs keras models
* Link for walking through seeding https://vip.gatech.edu/wiki/index.php/Guide_to_Debugging_Individual_Algorithms 
* Use Seeding from toxicity (dense layer, lstm)
{| class="wikitable"
!Task
!Current Status
!Date Assigned
!Date Resolved
!Date Due
|-
|Finish Amazon Run and Share Results
|Complete
|03/08/2021
|03/06/2021
|03/12/2021
|}

=== EMADE Run on PACE ===
* Having resolved last week's issues by pulling emade_operators.py from the nn branch, I was able to run EMADE successfully without crashing
* It ran for a total of 67 generations, crashing on generation 68. I analyzed my results, as shown below.
* Via the XML file, there were two objectives in this run. One was to minimize the number of parameters in the NNLearner, while the other was to achieve optimal accuracy (minimizing with the accuracy_score function, with a 1 being the worst).
* This is the Pareto Front of the results. The x-axis is the accuracy score, and the y-axis is the number of parameters.
* These results were pretty bad, with a Hypervolume of 1 * 10^9. 
* I investigated further and found out why. The best individual was a Gaussian, not an NNLearner. By design, these individuals don't have a number of parameters, as they aren't a neural network tree. Therefore, the value wasn't actually 1 * 10^9, but actually infinity. Therefore, all of these learners weren't NNLearners, and they were failing. This design choice was made so that NNLearners would be evaluated sooner.
* At Generation 68, this error happened. I was informed on Friday that it was most likely a DEAP issue.
* In the meeting on Friday, I was informed that the best path forward would be to seed individuals that were NNLearners, which wouldn't fail.

== Week 6: March 01, 2021 ==

=== General Meeting Notes ===
* Stocks
** Implemented new technical indicators, like BIAS and DeltaSMA
** Looking into why trading signals have flat parts between segments
** Some price data is inconsistent with the paper, currently looking into that
* Modularity
** Looking at literature to vision improvements to current infrastructure
** Stil using Sphinx
* EZGCP
** Benchmarking and fine tuning individual training times
** Performed 8 hours of runs and 7 generations
** Working on visualizing Pareto Front
* NLP
** I had a tourney selection (individuals must be divisible by 4) which we resolved by running pip install deap==1.2.2
** Sumit has an issue with Google Colab that is being debugged to get the Keras model working.

=== Sub Team Meeting ===
* We worked through Sumit's error in Google Colab
* Cameron is also running Amazon on PACE successfully now
* The tournament selection works now with the fix, waiting for the rest of the run to continue

=== Friday Meeting ===
* Decided to pursue a new control set for Amazon as Google Collab wasn't working
* Will problem on Stack Overflow
* Anish gave me the updated emade_operators.py to fix the concat_learner issue

=== Action Items: ===
{| class="wikitable"
!Task
!Current Status
!Date Assigned
!Date Resolved
!Due Date
|-
|Attempt EMADE run on PACE
|In Progress
|02/08/2021
|03/05/2021
|03/05/2021
|-
|Resolve Remaining Gen 0 Runtime Errors on EMADE
|In Progress
|02/26/2021
|03/04/2021
|03/05/2021
|}

=== EMADE Run on Pace ===
* To fix tourney selection issue, run "pip install deap==1.2.2"
* Gen 0 continued running successfully after this, until I obtained a RecursionValue Error
* This error happens when your program recurses past the max allowed by the system
* You can resolve this with "u-limit -s", as demoed below, which shows the initial max value as 8192 and changes it to 12000

* On Friday, it turns out this recursion was actually an error with the concat learner that Anish fixed in the nn branch; the emade_operators.py file was changed to resolve this.

== Week 5: February 22, 2021  ==

=== General Meeting Notes ===
* Stocks:
** There were bugs on the PLR, labels should be working, indicators should be working
** Most Indicators mentioned in the paper are already developed
** Going to Report scores on aggregate of 6 folds
** Were only able to get 2 of Taiwanese Stock
* Modularity:
** Pretty much done with the documentation side of things
** Sphinx was easy to use
** Team is more familiar with the code base now
* NLP:
** Making sure Amazon is ready to go on PACE-ICE
** Would like to refactor to Pytorch
** I am working on fixing .gz issues w/ the dataset
* EZGCP:
** Got PACE-ICE setup for their accounts
** made a shared .conda configuration file
** getting errors before individuals are saved
** only made it through 1 generation
** could technically save individuals earlier, but think bugs can be fixed
** Goal for next week: Get a full run
=== Sub-Team Meeting ===
* I've been working through bugs in setting up Amazon
* Our branch doesn't have a set environment .yml file; Cameron will work on that
* I had to use pip install to get keras pickle wrapper, nltk, and transformers from tensorflow
* I set up Git LSF wrong, files were not in proper .gz format
* Group working on a Keras Model is still encountering errors as well

=== Friday Meeting ===
* I was able to successfully get Amazon running on Emade on Pace
* There are some bugs in the NN-VIP branch that printed out 0 individuals
* I worked with Anish to add lines of code to particular files to resolve these errors.
* There are some issues with seeding that still need to be resolved
* The following line of code was added to line 213 in gp_framework_helper.py in the nn-vip branch to resolve the issue

* After adding this line of code, 'bash reinstall.sh' has to be run.

=== Action Items: ===
{| class="wikitable"
!Task
!Current Status
!Date Assigned
!Date Resolved
!Due Date
|-
|Attempt EMADE run on PACE
|In Progress
|02/08/2021
|
|03/05/2021
|-
|Resolve Runtime Errors w/ Data formatted incorrectly
|Complete
|02/22/2021
|02/24/2021
|02/26/2021
|-
|Resolve Runtime Errors w/ Seeding
|Complete
|02/26/2021
|03/01/2021
|03/05/2021
|}

=== EMADE Run on Pace ===
* Environment issues were resolved last week and before Monday Meeting
* I resolved MySQL issues; I had to give my user permission for all tables, all databases, on both password and no password settings
* I encountered an error with .gzip.
* I setup Git LSF wrong; I only had a pointer to the data, not the data itself. This was resolved by downloading the data raw and transferring it over with scp.
* After Friday, EMADE successfully runs and completes NSGA II. There is an issue with length of individuals not being divisible by 4 with TournamentDCD.

== Week 4: February 15, 2021 ==

=== General Meeting Notes ===
* Stocks: have to write a new fitness function and have new data, still looking for a new paper, got all the datasets used in one paper currently being used.
** Data is from 2008, which could be a problem due to 2008 recession.
* EZGCP:
** worked to run EZGCP without transfer learning
** Added multi-channel support to equalize
* Modularity:
** Adding more complexity into ARL's
** Still conducting a literature review
* NLP:
** Cameron and I are still working on running EMADE on PACE
** Submit, Alex, Anshul still working on Keras Model
=== Sub-Team Meeting ===
* launchEmade.py isn't working, there are MYSQL issues
* Goal: resolve issues by Friday

=== Friday Meeting ===
* I was tasked w/ performing an EMADE run using Amazon
* I also gave an update on the run on PACE thus far
* Anish would give us the data and XML for Amazon on Sunday

=== Action Items: ===
{| class="wikitable"
!Task
!Current Status
!Date Assigned
!Date Resolved
!Due Date
|-
|Attempt EMADE run on PACE
|In Progress
|02/08/2021
|
|03/05/2021
|-
|Have MySQL and Environment for EMADE PACE setup
|Complete
|02/15/021
|02/20/2021
|02/20/2021
|}

=== EMADE Run on Pace ===
* launchEmade.py is not working
* There was something running on port 3306, so I had to set it to 3307
* MySQL connects now
* I created a user, and gave him permissions
* LaunchEmade.py now runs without crashing, I just have some new library issues; it seems some primitives use libraries like nltk
* I installed nltk, transformers, and keras pickle wrapper using pip

== Week 3: February 8, 2021 ==

=== General Meeting Notes ===
* Stocks: have a little better understanding of emade now. Trying to find a new paper to base their work off of. Discussing ways to explore new primitives and change theirs. Also discussing types of stocks that should be targeted.
* EZGCP: met with team to discuss semester goals; are very consistent with last semester, using minGPT to make a new primitive
* Modularity: still doing a literature review, everyone brought a paper and people are reading now
* NLP: Refining goals for the semester. Focusing purely on NLP, addressing a neural net only finding trivial solutions, one group focusing on control w/ Kerns on Amazon dataset, another group focusing on Emade on Amazon dataset.
* Self Evaluation is Due next Week, Rubric is on the Wiki.

=== Sub-Team Meeting ===
* Sumit, Anshul, Alex are looking for papers to use as a baseline for Amazon dataset. Should be ready for Friday meeting.
* Amazon should be ready on EMADE by the week's end.
* As it only takes one person to really prep Amazon, I will be attempting to setup runs on PACE to help for the end of semester rush, in case we have to have a bunch of runs again.

=== Friday Meeting ===
* I updated the sub team on my progress setting up PACE
* I had issues with bash reinstall.sh
* I resolved it by setting up a different environment to download GPFramework from
* Sumit, Anshul, and Alex are progressing, and found a paper



=== Action Items: ===
{| class="wikitable"
!Task
!Current Status
!Date Assigned
!Date Resolved
!Due Date
|-
|Attempt Emade Run on Pace
|In Progress
|02/08/2021
|
|03/05/2021
|-
|SCP transfer EMADE to pace
|Complete
|02/08/2021
|02/15/2021
|2/15/2021
|-
|Setup EMADE environment on PACE
|Complete
|02/08/2021
|02/15/2021
|02/15/2021
|}

=== Emade Run on Pace ===
* First, due to a resetting of my laptop, I had to redownload and install emade, using my notebook from last semester as a guide.
* New conda environment is named "emade34"
* First, to avoid not knowing the source of errors, I began trying to run the toxicity dataset on PACE, and decided to use Amazon later
* ssh'ed onto the pace server successfully
* scp at first denied permission. Creating a folder called "vip" and specifying it in the destination path resolved this issue.
* Had some conda environment issues at first. The pace guide helped resolve this
* Had to remove GPFramework from .yml file, name it "emade35", and then "bash reinstall.sh" and "pip install gpramework==1.0" afterwards to add this.
* PACE emade environment is named EMADE35

=== Notebook Self Evaluation ===

== Week 2: February 1, 2021 ==

=== General Meeting Notes ===
* Stocks: trying to come up with a larger goal.
** Had problems with the dataset. 
* NLP: team leadership undecided on
** Working out goals for this semester
* EZGCP:
** Two projects, one person each
* Modularity
** want to try more complex construction of ARL's
** Didn't get too many experiments in previously

=== Sub-Team Meeting ===
* We considered the different problems we could pursue in the domain of NLP/Neural Architecture Search with EMADE
* Examples include finding a potential bug in the NNLearner that prevents complex solutions from happening
* Possibly going back to a more pure NLP task
* Goal of staying more united this Semester as opposed to working on several different projects

=== Friday Meeting ===
* Focusing on Pure NLP is an ideal task
* If there is a bug, then doing anything else is essentially useless
* First, attempting to see how Emade runs on a more distributed binary data set (Amazon) vs the results of a simple model from Keras on the same dataset
* If all is good there, use a multi-class dataset and have the same EMADE vs control model
* My task: continue work on getting Amazon up and ready, w/ Cameron and Anish



=== Action Items: ===
{| class="wikitable"
!Task
!Current Status
!Date Assigned
!Date Resolved
!Due Date
|-
|Meet with NLP Group
|Complete
|01/25/2021
|2/01/2021
|02/01/2021
|}

=== Preparing Amazon for Emade ===
* I posted the files I worked on last semester in the slack channel.
* This includes the pre-processed data and XML file
* Discussed brief modifications over the pre-processing
* UTF-8 issue was resolved

== Week 1: January 25, 2021 ==

=== General Meeting Notes ===
* Meeting Goals: set priorities and goals for the semester
* Notebooks remain important due to online format
* Returning Members are open to jumping ship.
* Teams continuing on: No new ones, Stocks, EZGCP, NN/NLP team, Modularity
* Will decide teams w/ a spreadsheet
* Main problem last semester on NLP team: Trees weren't building complexity very well (but we could build neural networks with trees)
* Possible goal: switch from Keras to Pytorch
* Using LSTM's on tree structure to learn existing grammar. (look at different fields in GP to increase complexity)<br>
=== Action Items: ===
{| class="wikitable"
!Task
!Current Status
!Date Assigned
!Date Resolved
!Due Date
|-
|Meet with NLP Group
|In Progress
|01/25/2021
|
|02/01/2021
|}
