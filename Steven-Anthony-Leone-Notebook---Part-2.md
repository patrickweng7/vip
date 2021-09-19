# Team Member
<b>Team Member:</b> Steven Leone <br>
<b> Major: </b> Computer Science <br>
<b>Email:  </b>sleone6@gatech.edu <br>
<b>Cell Phone:</b> (412)-378-7253 <br>
<b>Interests:</b> Machine Learning, Natural Language Processing, Software Engineering, Algorithms <br>
<b>Sub Team:</b> NLP

# Fall 2021
## Week 4

### Self Evaluation Rubric
<img width="962" alt="Screen Shot 2021-09-13 at 7 11 24 PM" src="https://github.gatech.edu/storage/user/27405/files/6e678180-14c6-11ec-811b-bb9acf9a8258">

### General Meeting
* We updated the team on our progress.
*3/5 of our members have gotten EMADE on PACE setup.

### Sub Team Meeting
* We gave our updates on the assigned tasks so far
* Everyone has been able to get EMADE to run the Amazon dataset on PACE.
* We then went over the objectives we needed to figure out to get EMADE to work.
* We decided our objectives would be F1 and number of parameters, to strike a balance between a match/accuracy and complexity. My task was to look into F1 for QA systems and find out how they worked (we ruled out EM as it was similar to F1, but harder to train with).
* Karthik created a Google Collab notebook to begin looking at the dataset.
* We would familiarize ourselves with the different layers of a QA system before planning what primitives to make on Monday.

### F1 Research
* As opposed to other models, I discovered that F1 is used to score each answer provided individually. Then, the F1 score of each data point is averaged.
* There are two answers, which are the predicted answer and the ground truth answer. Words in the predicted answers are true positives if they appear in the ground truth answer, false positives if they don't. We have false negatives if words from ground truth don't appear in the data as well.
* I began to write code to write this as an objective function for EMADE, basing it off of the current accuracy objective.



## Week 3
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


## Week 2
### General Meeting Notes
* During the general meeting, I informed the whole team of ideas discussed in our brainstorm meeting. Devan also suggested added more primitives for more than embeddings.
* The sub team for NLP was officially formed with members decided.

### Action Items
Task | Current Status | Date Assigned | Date Resolved | Date Due |
--- | --- | --- | --- |--- |
Setup Weekly Sub Team Meetings | Complete | 08/30/2021 | 09/07/2021 | 09/13/2021 |
Finalize Semester Goals | Complete | 08/30/2021 | 09/07/2021  | 09/13/2021 |
Start Progress Towards Semester Goals | Complete | 08/30/2021 | 09/12/2021 | 09/13/2021 |


## Week 1
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