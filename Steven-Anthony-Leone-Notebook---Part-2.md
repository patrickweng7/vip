# Team Member
<b>Team Member:</b> Steven Leone <br>
<b> Major: </b> Computer Science <br>
<b>Email:  </b>sleone6@gatech.edu <br>
<b>Cell Phone:</b> (412)-378-7253 <br>
<b>Interests:</b> Machine Learning, Natural Language Processing, Software Engineering, Algorithms <br>
<b>Sub Team:</b> NLP <br>
<b>Sub Team Teammates:</b> Karthik Subramanian, Devan Moses, Kevin Zheng, Shiyi Wang


# Fall 2021
## Week 5
### General Meeting Notes
* We informed the team of our progress, including selecting objectives. We decided that previous models using num params vs an accuracy objective weren't quite diverse enough. Adding other objectives will allow a better tradeoff space that will lead to diverse models.
* Having decided on our objectives and analyzed the dataset, we then had to tackle the building blocks of a QA system. The biggest challenge with them is how there are two inputs, the context and the query.
* I would research the current state of the nn-vip EMADE branch and present a best strategy for implementing a QA system.

### Sub Team Meeting
* I presented to the sub team the information I compiled on NNLearners, shown in the section underneath Action Items, and how we could use them for QA systems.
* We discussed several strategies for the layers we needed. 

### Action Items
Task | Current Status | Date Assigned | Date Resolved | Date Due |
--- | --- | --- | --- |--- |
Research NNLearners and how to implement QA | Complete | 09/20/2021 | 09/22/2021 | 09/22/2021 |
Research creating a new type of data pair | In Progress | 09/20/2021 | ...  | 10/03/2021 |


### Researching NNLearners and how to implement QA
* My assignment was to research the current state of NNLearners and see how we could add primitives to make the problem solvable
* Having never actually coded for the branch before, I started by looking at NNLearners in neural_network_methods.py.
* The NNLearner was setup so that each primitive was a layer from a Keras Neural Network. A lot of primitives we would need were already there. The output layer, attention layer, and embedding layers existed.
* The initial input for an NNLearner was a data_pair, referred to in the tree structure as ARG0. I figured out that the type is an EMADEDataPair by looking at the inputs and comparing methods called on it.
* From this information, I formed strategies for our two obstacles:
** To deal with the requirement of a different way of obtaining outputs from an outputted probability matrix, we could make a new data pair type, similar to the types 'textdata' and 'imagedata' that have already been implemented in EMADE. Then, at the end of the NNLearner code, we could implement an if statement to obtain the proper start and endpoints. 
** To deal with the requirement of two inputs per data point, the context and the query, we could make primitives that specifically return an embedding of exactly one of the inputs. For example, one could be named "ContextEmbeddingLayer" and another could be "QueryEmbeddingLayer".
* Analyzing these solutions from the mindset of attempting to increase and maintain diversity in individuals, I think these solutions will work well. As opposed to the original thought process of constraining NNLearners to always have one type of input or output, this will maintain the normal constraints of mating individuals, which should still allow for any individuals to be mated, increasing our diversity.

## Week 4

### Self Evaluation Rubric
<img width="962" alt="Screen Shot 2021-09-13 at 7 11 24 PM" src="https://github.gatech.edu/storage/user/27405/files/6e678180-14c6-11ec-811b-bb9acf9a8258">

### General Meeting
* We updated the team on our progress.
* 3/5 of our members have gotten EMADE on PACE setup.

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
* I finished writing the modified F1, based on the official SQUAD 2.0 evaluation dataset:
<img width="839" alt="Screen Shot 2021-09-19 at 1 31 24 PM" src="https://github.gatech.edu/storage/user/27405/files/01aa3500-1a3a-11ec-80fa-7324ee23d348">

### Question Answering Systems Research / Refresher
* A tasked we assigned to all members of the team was to get refreshed on Question Answering Systems, or learn about them if no previous knowledge existed. 
* 



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
* As the linked paper describes, gated-self matching networks work as follows: the query and the context are embedded separately, then processed by a bidirectional RNN. Then, the outputs are combined through gated attention layers. Another attention layer is then used, which applies this output to itself in this attention layer. The paper diagrams this structure with the figure below:
* TODO: Insert Image
* I don't believe this paper is useful in terms of finding inspiration for new primitives. As opposed to a new mechanism that could be used, the primary innovation in this paper is the structure of the QA system. The individual units, like GRU and Attention layers, are already implemented. The structure, or how these building blocks will fit together, is for EMADE to decide by evaluating generation after generation of individuals and looking at the ones on the Pareto Front for inspiration.
* However, I do believe this paper is helpful for our task in that it is a state of the art model for the SQUAD dataset. We can use it as inspiration to make a seeded individual, by writing this out in a tree structure. 


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