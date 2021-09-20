# Team Member Information
Name: Vincent Huang\
Major: CS, Math\
Contact: vhuang31@gatech.edu

## Previous Semesters' Notebook
https://wiki.vip.gatech.edu/mediawiki/index.php/Notebook_Vincent_H_Huang

# Fall 2021
### Week 4: Sep 13
- fix bug from last semester
- self eval
-pace ice
- [Notebook Self Evaluation](https://gtvault-my.sharepoint.com/:w:/g/personal/vhuang31_gatech_edu/EbvqH1NxuhRNkHojhS7cAwMBRufrsPn2O3e4NHXhJc3aWA?e=9RPwiv)

Problem: hard to test if greater than depth 2 is working since individuals aren't getting very large fast
solution 1: Get more seeding individuals
solution 2: merge with mnist to get seeding individuals
solution 3: Search invalid individuals for ARLs

/home/vincent/anaconda3/lib/python3.6/site-packages/deap/tools/emo.py:735: RuntimeWarning: invalid value encountered in double_scalars
  individuals[j].fitness.values[l]
Traceback (most recent call last):
  File "src/GPFramework/didLaunch.py", line 126, in <module>
    main(evolutionParametersDict, objectivesDict, datasetDict, stats_dict, misc_dict, reuse, database_str, num_workers, debug=True)
  File "src/GPFramework/didLaunch.py", line 116, in main
    database_str=database_str, reuse=reuse, debug=True)
  File "/home/vincent/anaconda3/lib/python3.6/site-packages/GPFramework-1.0-py3.6.egg/GPFramework/EMADE.py", line 802, in master_algorithm
    count = mutate(offspring, _inst.toolbox.mutateLearner, MUTLPB, needs_pset=True)
  File "/home/vincent/anaconda3/lib/python3.6/site-packages/GPFramework-1.0-py3.6.egg/GPFramework/EMADE.py", line 600, in mutate
    mutate_function(mutant, _inst.pset)
  File "/home/vincent/anaconda3/lib/python3.6/site-packages/GPFramework-1.0-py3.6.egg/GPFramework/emade_operators.py", line 26, in insert_modifyLearner
    slice_ = individual.searchSubtree(index)
  File "/home/vincent/anaconda3/lib/python3.6/site-packages/deap/gp.py", line 180, in searchSubtree
    total += self[end].arity - 1
IndexError: list index out of range

https://stackoverflow.com/questions/27784528/numpy-division-with-runtimewarning-invalid-value-encountered-in-double-scalars

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



