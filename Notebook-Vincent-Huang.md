# Team Member Information
Name: Vincent Huang\
Major: CS, Math\
Contact: vhuang31@gatech.edu

## Previous Semesters' Notebook
https://wiki.vip.gatech.edu/mediawiki/index.php/Notebook_Vincent_H_Huang

# Fall 2021

### Week 6: Sep 27
|Task|Status|Assigned Date|Due Date|Date Completed|
|----|------|-------------|--------|--------------|
|Fix contract ARLs method|
|Investigate add_all_subtrees problem|
|Do runs of extended ARLs|

Currently at index 2
Generated ARL:
 arl5 : lambda arl_arg_0,arl_arg_1,arl_arg_2: (Learner(EqualizeHist(arl_arg_0,-6,arl_arg_1),arl_arg_2))
examining subtree (('Learner', 1, 2, -1), ('ARG0', 0, 0, 0))
Currently at index 0
Currently at index 1
Ignoring inter-generation duplicate lambda arl_arg_0,arl_arg_1: (Learner(arl_arg_0,arl_arg_1))
Updateing PSET Representation with 2 arls
	arl4:
lambda arl_arg_0,arl_arg_1,arl_arg_2: (EqualizeHist(arl_arg_0,arl_arg_1,arl_arg_2))
Args in:  (<class 'GPFramework.data.EmadeDataPair'>, <class 'GPFramework.constants.TriState'>, <class 'GPFramework.constants.QuadState'>)
	arl5:
lambda arl_arg_0,arl_arg_1,arl_arg_2: (Learner(EqualizeHist(arl_arg_0,-6,arl_arg_1),arl_arg_2))
Args in:  (<class 'GPFramework.data.EmadeDataPair'>, <class 'GPFramework.constants.QuadState'>, <class 'GPFramework.gp_framework_helper.LearnerType'>)
Indiv copy:  Learner(EqualizeHist(ARG0, 2, 3), learnerType('RAND_FOREST', {'n_estimators': 100, 'criterion': 0, 'max_depth': 3, 'class_weight': 0}, 'SINGLE', None))
occurrence!  68 ((('EqualizeHist', 2, 3, -1), ('ARG0', 0, 0, 0), ('-6', 0, 0, 1)), 1)
Learner(EqualizeHist(ARG0, 2, 3), learnerType('RAND_FOREST', {'n_estimators': 100, 'criterion': 0, 'max_depth': 3, 'class_weight': 0}, 'SINGLE', None))
Contracting ARL
len individual before removal 6
individual before removal [('Learner', 2, 0), ('EqualizeHist', 3, 1), ('ARG0', 0, 2), ('-6', 0, 3), ('0', 0, 4), ("learnerType('BOOSTING', {'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 3}, 'BAGGED', None)", 0, 5)]
Nodes to remove:  [3, 2, 1]
len individual after removal 3
individual after removal [('Learner', 2, 0), ('0', 0, 1), ("learnerType('BOOSTING', {'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 3}, 'BAGGED', None)", 0, 2)]
arl to insert <deap.gp.Primitive object at 0x7f205264e6d8> arity 3 newarity 1
len individual after arl insert 4
individual after arl insert [('Learner', 2, 0), ('arl4', 1, 1), ('0', 0, 2), ("learnerType('BOOSTING', {'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 3}, 'BAGGED', None)", 0, 3)]
occurrence!  68 ((('Learner', 1, 2, -1), ('EqualizeHist', 1, 3, 0), ('-6', 0, 0, 1)), 0)

### Week 5: Sep 20

unpickling data results in
AttributeError: Can't get attribute 'Individual' on <module 'deap.creator' from '/home/vincent/anaconda3/lib/python3.6/site-packages/deap/creator.py'>




Looks like the contract_arls method is in a try except block and if it encounters an error it just ignores it?


- Bug causing the crashes has been identified
    - First tried getting the pickled individuals
        - Got from MySQL Workbench by right clicking and saving as file, then opening python terminal and loading it
        - Results in `AttributeError: Can't get attribute 'Individual' on <module 'deap.creator' from '/home/vincent/anaconda3/lib/python3.6/site-packages/deap/creator.py'>` Error
        - Did not work since the crashes usually happened before the problematic individuals were inserted into the database
        - Ended up manually printing problematic individuals' information
        -
        ```python
        def _check_valid_indv(indv, indv_idx, location_str):
                    if not indv:
                        print("CHECKVALID: NULL INDV")
                        return -1
                    if indv_idx >= len(indv):
                        # with open(f'invalid_indv.pickle', 'wb') as handle:
                        #     pickle.dump(indv, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        return -1
                    num_children_left = indv[indv_idx].arity
                    curr_idx = indv_idx + 1
                    for i in range(num_children_left):
                        curr_idx = _check_valid_indv(indv, curr_idx, location_str)
                        if curr_idx == -1:
                            break
                    if indv_idx == 0 and curr_idx == -1:
                        print(f"INVALID INDIVIDUAL IN {location_str}:\nINDIVIDUAL: {indv}\nINDIVIDUAL LIST: {[(indv[i].name, indv[i].arity) for i in range(len(indv))]}")
                    return curr_idx
        ```

    - Contract ARLs method wasn't properly updating arities of the node(s) surrounding the contracted ARL
        - Example
        -

        ```
             (node 0, arity 2)
                   /  \  
          (node 1, arity 0) (node 2, arity 0)
        ```

        -

        ```
             (ARL, arity 2)
                    \
                   (node 2, arity 0)
        ```
        - Causes a list index out of bounds error whenever an individual containing such an arl was iterated through in mating, mutating, inserting modify learner, finding all subtrees, etc
        - Large chunk of code just wrapped in a try except block
- Problem with add_all_subtrees method
    - The current ARL creation code stores all possible subtrees in memory and randomly chooses a number of them, weighted based on its "goodness" (fitness of individual the ARL was created from)
    - This causes problems with decently sized individuals (eg, length of 82 and depth of 6)
    - Python really doesn't like this, grinds to a halt. Could be running out of memory or just taking a really long time to find all subtrees.
    - Workaround: Don't consider individuals above a certain size for ARLs
    - Future solution: Refactor code to generate ARLs as the subtrees are found
- Mnist team working through getting everyone on PACE-ICE to do runs
    - There was some ambiguity in the instructions which caused some confusion


|Task|Status|Assigned Date|Due Date|Date Completed|
|----|------|-------------|--------|--------------|
|Identify what's causing crashes during runs|Completed|Sep 20|Sep 26|Sep 25|
|Check primitive set for registered primitives|Completed|Sep 20|Sep 26|Sep 24|
|Gather and check pickled individuals for unregistered primitives|Completed|Sep 20|Sep 26|Sep 25|

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
|Start doing runs on arl_update|Completed|Sep 13|Sep 20|Sep 19|

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
|Finalize topics for semester|Completed|Sep 6|Sep 13|Sep 12|
|Migrate old Notebook|Completed|Sep 6|Sep 13|Sep 8|

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
|Rank desired subteams for semester|Completed|Aug 30|Aug 31|Sep 4|
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



