# Team Member Information
Name: Vincent Huang\
Major: CS, Math\
Contact: vhuang31@gatech.edu

## Previous Semesters' Notebook
https://wiki.vip.gatech.edu/mediawiki/index.php/Notebook_Vincent_H_Huang

# Fall 2021

### Week 6: Sep 27
Lots of different places where the entire individual is accessed
mating, mutating, inserting modify learner, generating children dict in adfs, etc

Example problem individual
INDV PRINT CRASH TEST node_idx: 2 len: 2 ind: learnerType('BOOSTING', {'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 3}, 'SINGLE', None)
indvlist [('arl2', 2), ("learnerType('LIGHTGBM', {'max_depth': -1, 'learning_rate': 0.1, 'boosting_type': 0, 'num_leaves': 31}, 'BAGGED', None)", 0)]
MUTANT len 6, indv learnerType('DEPTH_ESTIMATE', {'sampling_rate': 1, 'off_nadir_angle': 20.0}, 'SINGLE', None)

indvlist [('Learner', 2), ('ARG0', 0), ('ModifyLearnerList', 2), ('ModifyLearnerInt', 3), ('ModifyLearnerFloat', 2), ("learnerType('SVM', {'C': 1.0, 'kernel': 0}, 'BAGGED', None)", 0)]

INSERT MODIFY LEARNER index 0, indv len 4, indv learnerType('RAND_FOREST', {'n_estimators': 100, 'criterion': 0, 'max_depth': 3, 'class_weight': 0}, 'SINGLE', None)
indvlist [('Learner', 2), ('arl4', 3), ('-6', 0), ("learnerType('BOOSTING', {'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 3}, 'BAGGED', None)", 0)]

INVALID INDIVIDUAL IN MATING:
INDIVIDUAL: 4
INDIVIDUAL LIST: [('arl8', 3), ('ARG0', 0), ('4', 0)]

INVALID INDIVIDUAL IN MATING:
INDIVIDUAL: learnerType('RAND_FOREST', {'n_estimators': 100, 'criterion': 0, 'max_depth': 3, 'class_weight': 0}, 'SINGLE', None)
INDIVIDUAL LIST: [('Learner', 2), ('arl6', 3), ('-10', 0), ("learnerType('BOOSTING', {'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 3}, 'BAGGED', None)", 0)]

INVALID INDIVIDUAL IN MATING:
INDIVIDUAL: arl11(ARG0, ARG0, 3, 2, 3, learnerType('BAYES', None, 'SINGLE', None))
INDIVIDUAL LIST: [('Learner', 2), ('arl11', 6), ('ARG0', 0), ('ARG0', 0), ('3', 0), ('2', 0), ('3', 0), ("learnerType('BAYES', None, 'SINGLE', None)", 0)]

INVALID INDIVIDUAL IN MATING:
INDIVIDUAL: learnerType('DEPTH_ESTIMATE', {'sampling_rate': 1, 'off_nadir_angle': 20.0}, 'SINGLE', None)
INDIVIDUAL LIST: [('Learner', 2), ('ARG0', 0), ('ModifyLearnerList', 2), ('ModifyLearnerInt', 3), ('ModifyLearnerFloat', 2), ("learnerType('SVM', {'C': 1.0, 'kernel': 0}, 'BAGGED', None)", 0)]

More bugs:
Sometimes very large individuals generate (Eg, length ~80, depth 5 or so)
Takes very long to find all subtrees, causes python to run out of memory?

unpickling data results in
AttributeError: Can't get attribute 'Individual' on <module 'deap.creator' from '/home/vincent/anaconda3/lib/python3.6/site-packages/deap/creator.py'>

|Task|Status|Assigned Date|Due Date|Date Completed|
|----|------|-------------|--------|--------------|
|Identify what's causing crashes during runs|
|Check primitive set for registered primitives|
|Gather and check pickled individuals for unregistered primitives|

### Week 5: Sep 20
  Using TensorFlow backend.
  /home/vincent/anaconda3/lib/python3.6/site-packages/GPFramework-1.0-py3.6.egg/GPFramework/adfs.py:135: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray
  /home/vincent/anaconda3/lib/python3.6/site-packages/deap/tools/emo.py:735: RuntimeWarning: invalid value encountered in double_scalars
  individuals[j].fitness.values[l]
  Traceback (most recent call last):
  File "src/GPFramework/didLaunch.py", line 126, in <module> main(evolutionParametersDict, objectivesDict, datasetDict, stats_dict, misc_dict, reuse, database_str, num_workers, debug=True)
  File "src/GPFramework/didLaunch.py", line 116, in main
    database_str=database_str, reuse=reuse, debug=True)
  File "/home/vincent/anaconda3/lib/python3.6/site-packages/GPFramework-1.0-py3.6.egg/GPFramework/EMADE.py", line 821, in master_algorithm
    count = mutate(offspring, _inst.toolbox.mutateUniform, MUTUPB, needs_pset=True, needs_expr=True)
  File "/home/vincent/anaconda3/lib/python3.6/site-packages/GPFramework-1.0-py3.6.egg/GPFramework/EMADE.py", line 599, in mutate
    mutate_function(mutant, getattr(_inst.toolbox, _inst.expr), _inst.pset)
  File "/home/vincent/anaconda3/lib/python3.6/site-packages/deap/gp.py", line 749, in mutUniform
    slice_ = individual.searchSubtree(index)
  File "/home/vincent/anaconda3/lib/python3.6/site-packages/deap/gp.py", line 180, in searchSubtree
    total += self[end].arity - 1
  IndexError: list index out of range


Looks like the contract_arls method is in a try except block and if it encounters an error it just ignores it?

|Task|Status|Assigned Date|Due Date|Date Completed|
|----|------|-------------|--------|--------------|
|Identify what's causing crashes during runs|
|Check primitive set for registered primitives|
|Gather and check pickled individuals for unregistered primitives|

### Week 4: Sep 13

- Began doing extended ARL runs
    - Starting off with max depth 10 trees
    - Everything seems to be working, there exist ARLs with depth > 2
    - Example ARL

 Learner(arl_arg_0,ModifyLearnerBool(learnerType('BOOSTING', {'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 3}, 'ADABOOST', {'n_estimators': 50, 'learning_rate': 1.0}),arl_arg_1))

- New goal: Test the significance of the depth of ARLs on the performance of individuals
    - Problem 1: It takes a while for individuals with significant depth to appear, and therefore it takes ARLs with significant depth even longer to appear
        - Working on a seeding file which has more complex individuals so larger ARLs can generate more quickly
        - Manually randomly select individuals from runs which look different from the original seeds
        - Potential problem with limiting diversity?
        - Another solution would be potentially search invalid individuals for ARLs, something we brainstormed last semester
Old Seeds
```
Learner(ARG0, learnerType('RAND_FOREST', {'n_estimators': 100, 'criterion':0, 'max_depth': 3, 'class_weight':0}, 'SINGLE', None))
Learner(ARG0, learnerType('KNN', {'K': 3, 'weights':0}, 'BAGGED', None))
Learner(ARG0, learnerType('SVM', {'C':1.0, 'kernel':0}, 'SINGLE', None))
Learner(ARG0, learnerType('DECISION_TREE', {'criterion':0, 'splitter':0}, 'SINGLE', None))
```

New Seeds (Used in addition to Old Seeds)

```
Learner(ARG0, learnerType('BOOSTING', {'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 3}, 'SINGLE', None))
Learner(EqualizeHist(ARG0, 2, 3), learnerType('RAND_FOREST', {'n_estimators': 100, 'criterion': 0, 'max_depth': 3, 'class_weight': 0}, 'SINGLE', None))
Learner(ARG0, learnerType('LIGHTGBM', {'max_depth': -1, 'learning_rate': 0.1, 'boosting_type': 0, 'num_leaves': 31}, 'ADABOOST', {'n_estimators': 50, 'learning_rate': 1.0}))
Learner(ARG0, learnerType('ARGMAX', {'sampling_rate': 1}, 'BAGGED', None))
Learner(ARG0, ModifyLearnerFloat(learnerType('ARGMIN', {'sampling_rate': 1}, 'SINGLE', None), 0.01))
Learner(ARG0, learnerType('ARGMAX', {'sampling_rate': 1}, 'ADABOOST', {'n_estimators': 50, 'learning_rate': 1.0}))
Learner(ARG0, ModifyLearnerList(ModifyLearnerInt(ModifyLearnerFloat(learnerType('DEPTH_ESTIMATE', {'sampling_rate': 1, 'off_nadir_angle': 20.0}, 'SINGLE', None), 1.0), notEqual(-2.6349412187954435, 0.1), myIntSub(255, -6)), passList(myListAppend([1, 6], [-2, 14]))))
```

-    - Problem 2: There are several uncommon bugs which are ending runs prematurely
        - Both have to do with invalid value encountered in double_scalars individuals[j].fitness.values[l]
        - https://stackoverflow.com/questions/27784528/numpy-division-with-runtimewarning-invalid-value-encountered-in-double-scalars
        - Might have to do with a floating point error causing divide by zero errors
        - Dr. Zutty mentioned that it could be caused by an unregistered primitive, check the primitives.


```
/home/vincent/anaconda3/lib/python3.6/site-packages/GPFramework-1.0-py3.6.egg/GPFramework/adfs.py:134: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray
/home/vincent/anaconda3/lib/python3.6/site-packages/deap/tools/emo.py:735: RuntimeWarning: invalid value encountered in double_scalars individuals[j].fitness.values[l]
Traceback (most recent call last):
File "src/GPFramework/didLaunch.py", line 126, in main(evolutionParametersDict, objectivesDict, datasetDict, stats_dict, misc_dict, reuse, database_str, num_workers, debug=True)
File "src/GPFramework/didLaunch.py", line 116, in main database_str=database_str, reuse=reuse, debug=True)
 File "/home/vincent/anaconda3/lib/python3.6/site-packages/GPFramework-1.0-py3.6.egg/GPFramework/EMADE.py", line 802, in master_algorithm count = mutate(offspring, _inst.toolbox.mutateLearner, MUTLPB, needs_pset=True)
 File "/home/vincent/anaconda3/lib/python3.6/site-packages/GPFramework-1.0-py3.6.egg/GPFramework/EMADE.py", line 600, in mutate mutate_function(mutant, inst.pset)
 File "/home/vincent/anaconda3/lib/python3.6/site-packages/GPFramework-1.0-py3.6.egg/GPFramework/emade_operators.py", line 26, in insert_modifyLearner slice = individual.searchSubtree(index)
 File "/home/vincent/anaconda3/lib/python3.6/site-packages/deap/gp.py", line 180, in searchSubtree total += self[end].arity - 1
 IndexError: list index out of range
```

```
/home/vincent/anaconda3/lib/python3.6/site-packages/deap/tools/emo.py:735: RuntimeWarning: invalid value encountered in double_scalars individuals[j].fitness.values[l]
Traceback (most recent call last):
File "src/GPFramework/didLaunch.py", line 126, in main(evolutionParametersDict, objectivesDict, datasetDict, stats_dict, misc_dict, reuse, database_str, num_workers, debug=True)
File "src/GPFramework/didLaunch.py", line 116, in main database_str=database_str, reuse=reuse, debug=True)
File "/home/vincent/anaconda3/lib/python3.6/site-packages/GPFramework-1.0-py3.6.egg/GPFramework/EMADE.py", line 1097, in master_algorithm new_adfs, updated_individual_indices = _inst.adf_controller.update_representation(parents) # only modifies parent representation
File "/home/vincent/anaconda3/lib/python3.6/site-packages/GPFramework-1.0-py3.6.egg/GPFramework/adfs.py", line 1160, in update_representation population_info = self._find_arls(population)
File "/home/vincent/anaconda3/lib/python3.6/site-packages/GPFramework-1.0-py3.6.egg/GPFramework/adfs.py", line 568, in _find_arls self.search_individual(population[individual_num], individual_num, dictionary, self.max_adf_size)
File "/home/vincent/anaconda3/lib/python3.6/site-packages/GPFramework-1.0-py3.6.egg/GPFramework/adfs.py", line 1219, in search_individual self.generate_child_dict(individual, child_dict, next_dict, 0)
File "/home/vincent/anaconda3/lib/python3.6/site-packages/GPFramework-1.0-py3.6.egg/GPFramework/adfs.py", line 623, in generate_child_dict child_idx = self.generate_child_dict(individual, child_dict, next_dict, child_idx)
File "/home/vincent/anaconda3/lib/python3.6/site-packages/GPFramework-1.0-py3.6.egg/GPFramework/adfs.py", line 610, in generate_child_dict num_children_left = individual[node_idx].arity
IndexError: list index out of range
```

- Fixed temporary commits from last semester that were causing issues
- [Notebook Self Evaluation](https://gtvault-my.sharepoint.com/:w:/g/personal/vhuang31_gatech_edu/EbvqH1NxuhRNkHojhS7cAwMBRufrsPn2O3e4NHXhJc3aWA?e=9RPwiv)
- Mnist team is looking into working with PACE-ICE

|Task|Status|Assigned Date|Due Date|Date Completed|
|----|------|-------------|--------|--------------|
|Start doing runs on arl_update|

### Week 3: Sep 6
- Decided on finishing last semester's work
    - I decided to work on extending the depth of ARLs past depth 2, Xufei joining me
    - Angela, Bernadette, Tian are working on investigating issues from last semester's MNIST runs with super individuals
        - Could be caused by correlated objectives
        - Other possible causes mentioned during last semester's final presentation
- Loss of knowledge from Gabe's departure, need to learn how to do several things
    - Last semester used Gabe's AWS instance to do runs- figure out how to create one
    - Data visualizations
        - Statistical significance over generations
        - Visualization of individuals
    - I did a little bit of Pareto Front/AUC visualization work last semester
    - Fork new EMADE repo or continue using Gabe's fork
        - Ask Dr Zutty how this work should persist into the future

|Task|Status|Assigned Date|Due Date|Date Completed|
|----|------|-------------|--------|--------------|
|Finalize topics for semester|Completed|
|Migrate old Notebook|Completed|

### Week 2: Aug 30
- Joined modularity subteam
- Split leadership roles
    - Since team is basically split in half between regular and time conflict meetings, split leadership role
    - Bernadette leads discussion for time conflict, I'll lead Monday discussions
    - Look into combining entire team into the time conflict section
- More brainstorming ideas
    - Do baseline run on stock team’s data
        - https://github.gatech.edu/rbhatnager3/emade/tree/stocks-pr
    - New Models
        - Deep Ensembles with a diversity term
        - A CNN architecture with decaying learning rate
    - Selection Method
        - Modifying the evolutionary selection method to help encourage the spread of ARLs throughout the population and complexity. 
    - New Dataset Training
        - Look at other datasets to expand ARL training to see which ARLs stored in the database are the most used and why.
        - Practice on more image datasets and multi-class classification datasets.
    - Diversity Measures
        - Create some quantifiable way to measure diversity, generalizable for EMADE. May use a diversity measure as a heuristic when finding ARLs.
    - ARL Database Storage
        - Improve the way ARLs are stored in the database to keep any information from being lost
    - EMADE Integration
        - Integrate ARLs with EMADE’s original architecture and other modularity techniques


|Task|Status|Assigned Date|Due Date|Date Completed|
|----|------|-------------|--------|--------------|
|Rank desired subteams for semester|Completed|
|Continue brainstorming tasks|Completed|Aug 30|Aug 31|Sep 6|
|Ask Dr Zutty|Completed|Aug 30|Aug 32|Sep 6|

### Week 1: Aug 23
- Brainstorming topics, deciding subteams
    - Modularity
        - Finishing work from last semester, primarily extending depth beyond depth 2.
        - Finishing up mnist runs from last semester.
        - Fixing "super individuals" problem, could be caused by correlated objectives.
    - Infrastructure
        - Modify EMADE to use a distributed system instead of a single master and multiple workers
        - Add additional printing output so that users can know when errors occur without checking master file
        - Automatically close process during fatal error
        - Explore different data storage options besides mysql
    - EZCGP
    - Covid Data
        - Run emade to predict covid trends within communities

|Task|Status|Assigned Date|Due Date|Date Completed|
|----|------|-------------|--------|--------------|
|Brainstorm Topics for new semester|Completed|Aug 23|Aug 30|Aug 25|



