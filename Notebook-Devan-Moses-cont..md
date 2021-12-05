== Devan Moses ==
Team Members: [https://github.gatech.edu/emade/emade/wiki/Notebook-Karthik-Subramanian Karthik Subramanian], [https://github.gatech.edu/emade/emade/wiki/Notebook-Kevin-Zheng Kevin Zheng], [https://github.gatech.edu/emade/emade/wiki/Notebook-Shiyi-Wang Shiyi Wang], [https://github.gatech.edu/emade/emade/wiki/Notebook-Steven-Anthony-Leone Steven Leone]

Email: dmoses@gatech.edu

Cell Phone: 404-509-3758

Interests:
*Academic: Artificial Intelligence, Machine Learning, NLP, Cognition, Bio-Inspired Learning
*Recreational: Video games, Reading (Mostly fiction & academic), Outdoor activities (Hiking, Kayaking, etc.)

== Nov 29, 2021 ==
'''Team Meeting Notes:'''
* img processing:
** plan for week is to finalize where theyre at to run experiments
** nearing code freeze
* mod:
** running more experiments
* nlp:
** make sure to use seed to take advantage of autoML
* nas:
** nearing code freeze
** testing weight sharing, seems to work so far, have branch that doesnt rely on weight sharing as backup

'''Sub-team Meeting Notes:'''
* 

'''Individual notes:'''
* 

'''Action Items:'''
{| class="wikitable"
!Task
!Current Status
!Date Assigned
!Suspense Date
!Date Resolved
|-
|Present to new members
|Completed
|Nov 01, 2021
|Oct 03, 2021
|Oct 03, 2021
|-
|Test layer i/o dimensions
|Completed
|Nov 01, 2021
|Nov 06, 2021
|Nov 06, 2021
|}

== Nov 22, 2021 ==
'''Team Meeting Notes:'''
* img processing:
** new members on pace
** did baseline run but it had poor results
* mod:
** documentation and refactoring
** standardized var names
** update google cloud script
** cachev2 integration nearly ready for deployment
* nlp:
** for hypothesis testing we need to measure in 2d (consider points that have pareto dominance on both metrics)
** draw image of this limited AUC
* nas:
** need to us train, test, AND val datasets - gives a third pareto front to validate model scores
** want to work on implementing NN style weight sharing, eventually coevolution weight sharing as well

'''Sub-team Meeting Notes:'''
* 

'''Individual notes:'''
* 

'''Action Items:'''
{| class="wikitable"
!Task
!Current Status
!Date Assigned
!Suspense Date
!Date Resolved
|-
|Present to new members
|Completed
|Nov 01, 2021
|Oct 03, 2021
|Oct 03, 2021
|-
|Test layer i/o dimensions
|Completed
|Nov 01, 2021
|Nov 06, 2021
|Nov 06, 2021
|}

== Nov 15, 2021 ==
'''Team Meeting Notes:'''
* jason:
** adfs in all branches of emade, adf1,2,3
** each subsequent tree appears in prev tree's pset
** they get printed, care if string doesnt change -> hash doesnt change

* img processing
** dataset and hyper param issues solved
** new guys integrated
** sucecss would be improvement over baseline performance
* mod
** bug with shell script which will continuously download dataset on initial setup
** improve arl selection
* nlp
** jason: for memory error specifically from emade, can set the memoryLimit param in the xml
*nas
** weight ssharing
** adfs not consistent, generate at beginning and use at needed
** invalid layer combos allowed, 
** track inds, options: adding names to inds or adding table in sql to keep track of parent hashes
*** jason likes the table
** hoping to come up with different name for adfs specifically for emade

'''Sub-team Meeting Notes:'''
* This week was the hackathon in which we got NNLearner2 working on classification but were having issues switching to regression.
* Steven has been testing integration of primitives into EMADE, and building Keras Models in Google Collab to ensure integration works. All primitives we need to get now work. We have a Keras Model that builds, and our layers are able to feed into Keras's LSTM layer.
Karthik and George fixed batch size issues with Bidirectional Attention.
* Rishit and Shiyi are still designing our experiment.
* Getting any pace issues worked out is a subtask as well

'''Individual notes:'''
* Turns out that the old issue that steven and I found about single datapairs being treated as lists was unintentional like we thought. It was the same reason that we were getting the ARG1 issue. The following changes to always treat it as a list and unpack it solved the issue:
** <img src="https://github.gatech.edu/storage/user/46196/files/2100cb5c-811c-4304-8b8b-dc559b078e0d" width="30%">

'''Action Items:'''
{| class="wikitable"
!Task
!Current Status
!Date Assigned
!Suspense Date
!Date Resolved
|-
|Present to new members
|Completed
|Nov 01, 2021
|Oct 03, 2021
|Oct 03, 2021
|-
|Test layer i/o dimensions
|Completed
|Nov 01, 2021
|Nov 06, 2021
|Nov 06, 2021
|}

== Nov 08, 2021 ==
'''Team Meeting Notes:'''
* img processing
** looking into selection methods
** getting some dependency conflicts, jason suggested cachev2 install
** look into behavioral GP
* mod
** onboarding for noobies
** major bug with match_arl, wasn't properly checking children
** beginning merge
* nlp
** get some info from jason ab datapairs
* nas
** want to make some infrastructure changes but ADFs having weird behavior (old implementation from mod team)
** jason advice: ind = 4 trees in create_representation, then mate/mut done individually on each tree, eval done on compiled ind
*** can access ind before eval and replace trees to control ADF placement in a pop

'''Sub-team Meeting Notes:'''
* Resolved issues of Big Merge
* We divided into subteams in order to tackle the problems we've ran into, with my team being NNlearner2: me, david, and geoffrey 
* confirmed keras model works outside of emade using our primitives

'''Individual notes:'''
* I tasked my team with initially just getting familiar with emade.py, nnlearner, and using standalone tree evaluator as this would be the first time either of them would be working directly on emade's code base.
* I made a feature branch for nnlearner2
* I helped Steven and co. troubleshoot the memory issue as I ran into the same thing testing NNLearner.
* After some initial changes to add NNLearner2 I ran into the following issue testing "NNLearner2(ARG0,ARG1,OutputLayer(DenseLayer(10, defaultActivation, 10, LSTMLayer(16, defaultActivation, 0, trueBool, trueBool,EmbeddingLayer(100, ARG0, randomUniformWeights, InputLayer())))), 100, AdamOptimizer)" with standalone tree evaluator:
* <img src="https://github.gatech.edu/storage/user/46196/files/0ed0038e-7280-4476-9ac1-6c206953af15" width="30%">
* <img src="https://github.gatech.edu/storage/user/46196/files/345450ac-0006-4f35-a076-9607921701b8" width="30%">
* <img src="https://github.gatech.edu/storage/user/46196/files/2f9a54b4-3d88-42df-94f8-3319632d22e7" width="30%">
** This individual is just a test individual to troubleshoot
** At this point I wasn't sure if it was something with standalone tree evaluator or the primitive because I was expecting an error from the code rather than the individual itself.
* After finding the issue (it was a typo...) I got a separate error
** <img src="https://github.gatech.edu/storage/user/46196/files/a0774af2-b5be-44bf-892b-d62ff88628d1" width="30%">
** This turned out to be because the datapair was being treated as a list in our new setup but some of the code didn't reflect that.
* After fixing that I got to the error that haunted me for the next week or so:
** <img src="https://github.gatech.edu/storage/user/46196/files/56b8b616-0e5f-4cad-b4e6-7628d5495112" width="30%">

'''Action Items:'''
{| class="wikitable"
!Task
!Current Status
!Date Assigned
!Suspense Date
!Date Resolved
|-
|Get NNLearner2 working with QA
|In Process
|Nov 08, 2021
|Nov 15, 2021
|Nov 21, 2021
|}
