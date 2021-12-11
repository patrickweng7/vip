# Team Member Information
Name: Vincent Huang\
Major: CS, Math\
Contact: vhuang31@gatech.edu

## Previous Semesters' Notebook
https://wiki.vip.gatech.edu/mediawiki/index.php/Notebook_Vincent_H_Huang

# Fall 2021
### Week 16: Dec 6
- [Final Presentation Slides](https://docs.google.com/presentation/d/1crLSG4QjQPni3eeq-UIoN-_2AxYlqRWalCTuuCEUSbA/edit?usp=sharing)
    - Helped Sai implement changes to visualization to aggregate data for ARL size experiment visualizations
    - Met with first year students to discuss what to say for slides
- Implemented visualizations for ARL clustering experiment
```
results_tree_idx = 0
results_false_positives_idx = 1
results_false_negatives_idx = 2
results_gen_idx = 3

dbresults = dict()
arlresults = dict()
for dbnum in range(4, 12):
    db = MySQLdb.connect(host='database-2.ch6igrzmr2yu.us-east-2.rds.amazonaws.com', user='admin', passwd='mypassword', database=f'extendedarl{dbnum}')
    mycursor = db.cursor()
    mycursor.execute("SELECT tree, `FullDataSet False Positives`, `FullDataSet False Negatives`, evaluation_gen from individuals where `FullDataSet False Positives` is not null")
    results = mycursor.fetchall()
    dbresults[dbnum] = results

    mycursor.execute("SELECT name from adf")
    results = mycursor.fetchall()
    arlresults[dbnum] = results

for dbnum in dbresults.keys():
    result1 = dbresults[dbnum]
    arl_names = [res[0] for res in arlresults[dbnum]]

    arl_to_pareto_occ_map = dict()

    for arl_name in arl_names:
        has_indv = False
        for result in result1:
            if f"{arl_name}(" in result[results_tree_idx]:
                if arl_name not in arl_to_pareto_occ_map:
                    arl_to_pareto_occ_map[arl_name] = set()
                arl_to_pareto_occ_map[arl_name].add((result[results_false_positives_idx], result[results_false_negatives_idx], result[results_gen_idx]))

    fig, ax = plt.subplots()

    for arl_name, arl_occ_set in arl_to_pareto_occ_map.items():
        if len(arl_occ_set) > 10:
            x = []
            y = []
            color = []

            gens_to_graph = set([occ[2] for occ in arl_occ_set])
            reduced_to_gens = set([tuple(occ[1:]) for occ in dbresults[dbnum] if occ[results_gen_idx] in gens_to_graph and tuple(occ[1:]) not in arl_occ_set])

            for occ in reduced_to_gens:
                x.append(occ[0])
                y.append(occ[1])
                color.append([0, 0, 0, .3])

            for occ in arl_occ_set:
                x.append(occ[0])
                y.append(occ[1])
                color.append([1, 0, 0, .5])

            plt.scatter(x, y, c=color)
            t = plt.title(f'Objective Score of Aggregated Individuals Containing {arl_name}')
            plt.xlabel('False Positives')
            plt.ylabel('False Negatives')
            t.set_color("black")
            plt.xlim([0, 120])
            plt.ylim([0, 80])
            print(dbnum)
            plt.show()
            plt.clf()

to_graph = [(23, 4)] # arl,dbnum 

for arl_num, dbnum in to_graph:
    result1 = dbresults[dbnum]
    arl_names = [res[0] for res in arlresults[dbnum]]

    arl_name = f"arl{arl_num}"

    arl_to_pareto_occ_map = dict()

    for result in result1:
        if f"{arl_name}(" in result[results_tree_idx]:
            if arl_name not in arl_to_pareto_occ_map:
                arl_to_pareto_occ_map[arl_name] = set()
            arl_to_pareto_occ_map[arl_name].add((result[results_false_positives_idx], result[results_false_negatives_idx], result[results_gen_idx]))

    fig, ax = plt.subplots()

    for arl_name, arl_occ_set in arl_to_pareto_occ_map.items():
        if len(arl_occ_set) > 10:
            gens_to_graph = set([occ[2] for occ in arl_occ_set])
            reduced_to_gens = set([tuple(occ[1:]) for occ in dbresults[dbnum] if occ[results_gen_idx] in gens_to_graph and tuple(occ[1:]) not in arl_occ_set])

            gens_to_graph_list = sorted(list(gens_to_graph))
            print(dbnum)
            for gen in gens_to_graph_list:
                x = []
                y = []
                color = []
                for occ in reduced_to_gens:
                    if occ[2] == gen:
                        x.append(occ[0])
                        y.append(occ[1])
                        color.append([0, 0, 0, .3])

                for occ in arl_occ_set:
                    if occ[2] == gen:
                        x.append(occ[0])
                        y.append(occ[1])
                        color.append([1, 0, 0, .5])

                plt.scatter(x, y, c=color)
                t = plt.title(f'Objective Score of Aggregated Individuals Containing {arl_name}\nGeneration {gen}')
                plt.xlabel('False Positives')
                plt.ylabel('False Negatives')
                t.set_color("black")
                plt.xlim([0, 120])
                plt.ylim([0, 80])
                plt.savefig(f"{arl_name}_db{dbnum}_gen{gen}")
                plt.clf()
```
![](https://i.imgur.com/BQV8uTA.png)
![](https://i.imgur.com/cP7rTYY.png)
![](https://i.imgur.com/gcJXcEs.png)
![](https://i.imgur.com/S9O2xbk.png)

### Code Commits
- [Changes](https://github.gatech.edu/vhuang31/emade-viz/commit/c25b59577a108e1f1e80b03ea45216cc2632e777)
    - Implemented ARL clustering visualizations
|Task|Status|Assigned Date|Due Date|Date Completed|
|----|------|-------------|--------|--------------|
|extended ARL runs|Complete|Nov 21|Nov 24|Nov 29|
|Take a look at CacheV2's code|Complete|Nov 21|Nov 28|Nov 29|

### Week 15: Nov 29
worked on visualizations

- Since the CacheV2 Integrations team seemed to be struggling, I offered to take a look at their code
    - Very problematic code commits
    - Primary problem was that in `sql_connection_orm_master.get_seeded_pareto()`, it references `self.optimization`.
    - In the old ARL_Update branch, this was instantiated in the parent class `sql_connection_orm_base.__init__()`
    - In the new CacheV2 branch, the optimization attribute is instantiated via `ConnectionSetup()` in `EMADE.py`
    - ARL_Update makes no changes to `sql_connection_orm_master` and `sql_connection_orm_base`, so simply overriding all changes in those two files with CacheV2's version is perfectly fine
    - However, both ARL_Update and CacheV2 make changes to `EMADE.py`, so simply choosing one branch's changes and overriding the other's will not work.
    - The CacheV2 Integrations team simply chose one version of each file from a branch and overrode the other branch.
    - This is very problematic and not how a merge should work.
    - Given the time remaining before final presentations, I was not able to fix the problems in time.
    - Essentially no useful work was done, so the merge would need to be done over again from scratch.

|Task|Status|Assigned Date|Due Date|Date Completed|
|----|------|-------------|--------|--------------|
|extended ARL runs|Complete|Nov 21|Nov 24|Nov 29|
|Take a look at CacheV2's code|Complete|Nov 21|Nov 28|Nov 29|

### Week 14: Nov 22
- Thanksgiving break!
- Continued doing runs and gathering data for extended arl experiments

|Task|Status|Assigned Date|Due Date|Date Completed|
|----|------|-------------|--------|--------------|
|Run master process for extended ARL experiments|Complete|Nov 15|Nov 21|Nov 22|

### Week 13: Nov 15
- Refactored and documented new implementation of code
    - Standardized variable names
        - Example:
        - The same data was called population_info in update_representation, myDict in search_individual, and dictionary in add_all_subtrees
        - Same with arl_lambda, lambda, arl_function_string, arl_str
    - Added additional classes for information that was previously stored within tuples
```
class ARLPoolInfo:
    def __init__(self, arl_population_info: ARLPopulationInfo, arl_name: str, arl_lambda: str, arl_function, arl_input_types):
        self.population_info = arl_population_info
        self.arl_name = arl_name
        self.arl_lambda = arl_lambda
        self.arl_function = arl_function
        self.arl_input_types = arl_input_types
```
```
class ARLNode:
    def __init__(self, name, arl_arity, individual_arity, child_idx_in_parent):
        self.name = name
        self.arl_arity = arl_arity
        self.individual_arity = individual_arity
        self.child_idx_in_parent = child_idx_in_parent

    def __eq__(self, other):
        return other and self.name == other.name and self.arl_arity == other.arl_arity \
            and self.individual_arity == other.individual_arity \
            and self.child_idx_in_parent == other.child_idx_in_parent

    def __neq__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.name, self.arl_arity, self.individual_arity, self.child_idx_in_parent))

    def __repr__(self):
        return f"{self.name} {{arl_arity: {self.arl_arity}, indv_arity: {self.individual_arity}, idx_in_par: {self.child_idx_in_parent}}}"
```

### Code Commits
- [Changes](https://github.gatech.edu/vhuang31/emade/commit/5baa10d1c44a63ec65c893edaeac60e258c78afc)
    - Added ARLNode class
    - standardized variable names
    - Added documentation to gen_child_next_dicts, _contract_arls, _contract_arls, _find_arls, lambda_helper, update_representation_with_arl_pset, 
- [Changes](https://github.gatech.edu/vhuang31/emade/commit/7e1f2a539265e22a50c521e580cb37f4f4213aa1)
    - Added documentation to arl_info_to_lambda, update_representation, search_individual
    - Added typing to methods 
- [Changes](https://github.gatech.edu/vhuang31/emade/commit/952fc5e85978bf08ed142e4dbdfb9da90cc46e70)
    - Added ARLPoolnfo class

|Task|Status|Assigned Date|Due Date|Date Completed|
|----|------|-------------|--------|--------------|
|Write documentation for ARL code|Complete|Nov 9|Nov 12|Nov 15|
|Refactor existing code|Complete|Nov 9|Nov 14|Nov 15|
|Continue doing runs for extended ARL|Complete|Nov 9|Nov 13|Nov 15|

### Week 12: Nov 8
- Refactored and documented new implementation of code
    - Added typing and classes to implementation
        - Lots of information was previously stored in tuples, making it difficult to determine the meaning behind accesses
        - Example
Before:
```
for arl_instance_root_idx, individual_idx in new_arl_pool[arl][0][1:]:
```
After:
```
for arl_instance_root_idx, individual_idx in new_arl_pool[arl].population_info.occurrences:

class ARLPopulationInfo:
    def __init__(self, arl_fitness: int, occurrences: Set[Tuple[int, int]]):
        # An ARL's "fitness". See ADFController._evaluate_ARL
        self.arl_fitness = arl_fitness

        # Set of Tuples (ARL_idx, indv_idx)
        # ARL_idx: the index of the root node of the arl occurrence within the individual
        # indv_idx: the index of the individual within the population
        self.occurrences = occurrences
```
-
    -
        - Creating classes also allowed for making a repr function, which helped with printing information for debugging
        - Also added method input parameter types and return types where possible
    - Changed object instantiations to be more precise
        - Eg, `x = {}` to `x = dict()` and `y = {}` to `y = set()`
        - Note that the second instantiation is incorrect and is also a bug fix.

#### Code Commits
- [Changes](https://github.gatech.edu/vhuang31/emade/commit/5baa10d1c44a63ec65c893edaeac60e258c78afc)
    - Removed ARL arg index dictionary from population info
    - Commented out sanity check code for sake of performance
    - Added ARLPopulationInfo class
    - Added typing to the init, _get_best_arls, lambda_helper, etc methods
    - Changed object instantiations to be more precise
    - Added documentation to _get_best_arls,  _pick_arls, _match_arl_in_individual, add_all_subtrees, _evaluate_ARL, search_individual

|Task|Status|Assigned Date|Due Date|Date Completed|
|----|------|-------------|--------|--------------|
|Write documentation for ARL code|Complete|Nov 1|Nov 8|Nov 8|
|Write unit tests for new and old methods|Complete|Nov 1|Nov 8|Nov 8|
|Refactor existing code|Complete|Nov 1|Nov 7|Nov 8|
|Master process for extended arl runs|Complete|Nov 1|Nov 6|Nov 8|

### Week 11: Nov 1
- Reassessed and readjusted team's goals for the semester
    - Part of the Stocks group will focus on migrating modularity into Cache V2, as part of the problems with integrating stocks data with modularity was that stocks used CacheV2 functionality
    - Extended ARL group will continue doing runs and running experiments
- I onboarded first semester students regarding our team's tools
    - Briefly reviewed EMADE again and went over Google Collab
    - Wrote up the following guide

- How to get on google collab:
1. Clone the repo https://github.gatech.edu/vhuang31/emade/tree/master

2. If the CloudCopy.sh does not exist in the main directory, go ahead and make it by copying this file here 
https://github.gatech.edu/vhuang31/emade/blob/ARL_Update/CloudCopy.sh

3. If you cannot run CloudCopy.sh (it says something about permission denied), then run chmod +x CloudCopy.sh
This should create a emade-cloud directory. Upload that directory onto google drive and call it whatever the branch you're working on is (eg, emade-extended-arl) Note that you can call it whatever you want, just make sure to call it something so that you can tell which version of EMADE it is.
In google drive (or alternatively, you can do this before uploading the files to google drive), open up emade-cloud (or whatever you renamed the folder to) and navigate to templates

4. Open input_titanicADFON.xml in a text editor

5. Near the top of the file, there should be a dbConfig similar to this one. Edit it to match the following details, with the database renamed to the schema name the run you're trying to join is
```
<dbConfig>
        <server>database-2.ch6igrzmr2yu.us-east-2.rds.amazonaws.com</server>
        <username>admin</username>
        <password>mypassword</password>
        <database>INSERT_SCHEMA_NAME_HERE</database>
        <reuse>1</reuse>
</dbConfig>
```

You'll also want to change the following line to have a max arl size of 10
```
<maxAdfSize>10</maxAdfSize>
```

8.  In google drive, make a copy of this Google Collab Notebook https://colab.research.google.com/drive/1tUqnDzLHNg7RoYc4sarB3e2k3BvR_7D7?usp=sharing

9. In the notebook, edit the second step %cd /content/gdrive/MyDrive/INSERT-DIRECTORY-NAME-HERE/ to be whatever you renamed your directory in google drive to

10. Run all of the commands in the notebook sequentially except for the !python src/GPFramework/seeding_from_file.py [input xml template] [seeding file] command. This seeds the run with individuals, which only needs to be done once by the master process

11. Make sure that the !python src/GPFramework/launchEMADE.py -w templates/INSERT-TEMPLATE-FILE-NAME command has the -w flag. Otherwise, you will join as a master process which could cause problems.

12. Once the final command has been run, wait ~10 minutes and check the directory in google collab. You should see a new worker####.err and worker#####.out file. Check the worker#####.out file and note its progress. Wait another ~10 minutes and open the worker####.out file again. If nothing new has been written to the file, EMADE is probably not working and something has gone wrong. Otherwise you should be good to go! Alternatively, you could use MySQLWorkbench to check the status of the run. 

|Task|Status|Assigned Date|Due Date|Date Completed|
|----|------|-------------|--------|--------------|
|Reassess goals for semester|Complete|Oct 26|Oct 28|Nov 1|
|Onboard new students|Complete|Oct 26|Oct 29|Nov 1|
|Continue running experiments|Complete|Oct 26|Oct 30|Nov 1|

### Week 10: Oct 25
- [Midterm Presentation Slides Link](https://docs.google.com/presentation/d/1Lus6qHH9vwdfaLxcBg50PBBOl56qF_A7wFGT4F-1hlI/edit?usp=sharing)
- Visualization tools
    - Follow instructions on emade-viz repo README for environment setup and installation
    - emade-viz app takes in the input schema xml file
    - Unfortunately could not get it to work with remote aws instance database, so had to download the database to local in order to run the tool
    - Use command
```
mysqldump --column-statistics=0 -h [remote database host name] -u [remote database username] -p --hex-blob [remote database schema name] &> [local dump file].sql
```
-
    - On the ADFonlyAUC branch, this will output the AUC for each generation, which I then plugged into some new notebooks I made regarding AUC over time analysis and ARL size analysis

#### Code Commits
- [Changes](https://github.gatech.edu/vhuang31/emade-viz/commit/73aa08adc7ff013b7118c3bee361349753ab60af)
    - Added MNIST new objectives support to AUC visualization tool (commented out to preserve functionality)
- [Changes](https://github.gatech.edu/vhuang31/emade-viz/commit/18bfb97c973d9bbd3a51f6d06e00e774e1fa395a)
    - Added AUC over time analysis graph notebook
    - Added ARL size analysis visualization notebook

|Task|Status|Assigned Date|Due Date|Date Completed|
|----|------|-------------|--------|--------------|
|Figure out how to use emade visualization tools|Complete|Oct 24|Oct 25|Oct 25|
|Work on midterm presentation|Complete|Oct 18|Oct 25|Oct 25|
|Midterm presentation|Complete|Oct 18|Oct 25|Oct 25|

### Week 9: Oct 18
- Implemented ARL match helper method
    - Variation on dfs
    - We expect there to be (arity of parent node in original subtree) - (arity of parent node in ARL) arl arguments
    - We iterate over each child in the individual's tree, and if the current node does not match a child in the ARL, it must be an argument, so mark it as an argument.
    - This way we can identify which nodes in the individual should be deleted during contraction, and which nodes should be set as arguments to the new ARl primitive
- Refactored contract ARLs method
    - Iterate through individuals which contain ARL occurrences
    - For each occurrence, use the ARL match method to identify which nodes to contract
    - If there is no overlap, add the nodes to a removal set
    - Reverse sort the nodes to remove so that we don't change the index of future nodes to remove
    - Insert the arl with the arguments identified during match ARL
- Refactored ARL Lambda function
    - Similar to match ARL function in that it also runs a modified dfs where the arity of the original parent node is used to calculate missing children
    - Instead of marking "missing" children as arguments, instead inserts a placeholder arl_arg_# for the argument.

#### Code Commits
- [Changes](https://github.gatech.edu/vhuang31/emade/commit/1e8e1a0a255de2f7198f2419a978477fb6a578ce)
    - Refactored contract arls method
    - Added utility sanity check functions
    - Removed try catch blocks causing error suppression
    - Added helper function for matching ARL instance within individual
    - Fixed parent node arities not being updated
    - Fixed ARG0 begin removed during contraction
    - Fixed only a single ARL of each type being allowed for an individual
    - Added restriction for large individuals for ARL canidate formation
    - Added arl_argument index position information to population information
    - Added restriction for arl lambda to have at least 2 non arl argument nodes
- [Changes](https://github.gatech.edu/vhuang31/emade/commit/08bbedc7ab033f7cd075c30a0c4a17b680aa49ec) ([Hotfix](https://github.gatech.edu/vhuang31/emade/commit/ebc3364f9f37c5f6ce4d375d357f3b8c8dba162c)) ([Hotfix](https://github.gatech.edu/vhuang31/emade/commit/0eda0acf98452d10f9b1bb18f8eab0273d946055))
    - Fixed ARL Lambda method replacing fixed ARL arguments with variable ARL argument placeholders

|Task|Status|Assigned Date|Due Date|Date Completed|
|----|------|-------------|--------|--------------|
|Fix incorrect arities problem|Completed|Oct 11|Oct 18|Oct 17|
|Fix ARL fixed arguments being deleted|Completed|Oct 11|Oct 18|Oct 18|

### Week 7-8: Oct 4 and Oct 11
- Fixed implementation detail regarding ARLs were being created with only a single non-arg node
    - Now has a check to make sure that there exists > 1 non-arg node before creating the ARL.
- Fixed bug regarding each individual being restricted to a single ARL in order to prevent conflicts, but this code was non-functional
    - This was originally implemented because our framework contains the root node of ARL occurrences, and therefore contracting ARLs may cause several problems.
        - If we contract an ARL and therefore the root index of another ARL is changed
            - Solution: If we sort in reverse index DFS order, I believe we should be able to safely contract without changing the root indices of other ARLs.
            - Just in case this is not true, implemented sanity check to ensure that we completely remove a given ARL before removing nodes from another one.
        - If two ARLs overlap, then after the first one has been contracted, the overlapping nodes are gone and the second ARL either attempts to contract nodes which have been deleted (Index out of bound crash) or it contracts nodes which it isn't supposed to based on their indexes (Arity problems)
**** Solution: Use a set to keep track of which indices have already been marked for contraction, and don't attempt to contract an ARL who has nodes which have already been marked for contraction.
- Major bug regarding contraction which has been causing nearly all other bugs mentioned during previous weeks:
    - Our framework currently uses two major systems for identifying where ARLs occur in individuals
        - Firstly, the population_info stores the root indices for each occurrence of an ARL within the population
        - Example:
```
                 Learner
            /               \
     MorphDilateCross    learnerType
       /   |  \
    ARG0   3   5
```
-
    -
        - We also stored encoded ARL primitives to tell where additional args were
        - Example
```
                                            Learner
                                /                              \
                      MorphDilateCross                      learnerType
         /       /       |        |        \       \
    arl_arg0 arl_arg1 arl_arg2 arl_arg3 arl_arg4 arl_arg5
```
-
    -
        - Originally, the code deleted all the nodes it found within the ARL occurrence, and this caused us to potentially have more args than we expected
        - In the above example, we had already fixed arl_arg0 = ARG0, arl_arg2 = '3', and arl_arg3 = '5', but it still expects 5 arguments
        - Attempted solution: Don't contract args
        - This again caused problems because we weren't contracting the nodes which were fixed args, and therefore had more nodes than expected
        - Final solution: Update arity of ARL upon contracting, with special edge case for treating ARG0 as an arl_arg since (to my understanding) we don't want to contract it.
- Still have a bug
```
      File "/home/vincent/anaconda3/lib/python3.6/site-packages/GPFramework-1.0-py3.6.egg/GPFramework/adfs.py", line 517, in _contract_arls
       print(f"{new_individual}")
     File "/home/vincent/anaconda3/lib/python3.6/site-packages/deap/gp.py", line 97, in __str__
       string = prim.format(*args)
     File "/home/vincent/anaconda3/lib/python3.6/site-packages/deap/gp.py", line 204, in format
       return self.seq.format(*args)
     IndexError: tuple index out of range

    print(f"{[(node.name,node.arity) for node in new_individual]}")
    [('arl6', 4), ('EqualizeAdaptHist', 4), ('ARG0', 0), ('0', 0), ('0', 0), ('1.0', 0), ('2', 0), ('passQuadState', 1), ('3', 0), ('4.262076198386659', 0)]
```

#### Code Commits
- [Changes](https://github.gatech.edu/vhuang31/emade/commit/2d5d323d3d8a7edb162d3f90fd82b2633de6ecc9)
    - Fixed incorrect data type for ARL selection, removed unused and depreciated code

|Task|Status|Assigned Date|Due Date|Date Completed|
|----|------|-------------|--------|--------------|
|Fix population_info bug|Complete|Oct 4|Oct 11|Oct 10|
|Investigate Incorrect arities problem|Complete|Oct 4|Oct 11|Oct 8|

### Week 6: Sep 27
- Implemented workaround for add_all_subtrees large individuals bug
    - Gabe suggested instead of completely ignoring large individuals for ARL consideration or refactoring current framework, to only consider subtrees which take in an EMADE Data pair
    - This should be a lot easier to implement than refactoring current architecture; added to the to-do list.
- Fixed bug regarding incorrect arities in contract_arls
    - Example output

```
arl4: lambda arl_arg_0,arl_arg_1,arl_arg_2: (EqualizeHist(arl_arg_0,arl_arg_1,arl_arg_2))
Indiv copy:  Learner(EqualizeHist(ARG0, 2, 3), learnerType('RAND_FOREST', {'n_estimators': 100, 'criterion': 0, 'max_depth': 3, 'class_weight': 0}, 'SINGLE', None))
occurrence!  68 ((('EqualizeHist', 2, 3, -1), ('ARG0', 0, 0, 0), ('-6', 0, 0, 1)), 1)
Learner(EqualizeHist(ARG0, 2, 3), learnerType('RAND_FOREST', {'n_estimators': 100, 'criterion': 0, 'max_depth': 3, 'class_weight': 0}, 'SINGLE', None))
individual before removal [('Learner', 2, 0), ('EqualizeHist', 3, 1), ('ARG0', 0, 2), ('-6', 0, 3), ('0', 0, 4), ("learnerType('BOOSTING', {'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 3}, 'BAGGED', None)", 0, 5)]
Nodes to remove:  [3, 2, 1]
individual after removal [('Learner', 2, 0), ('0', 0, 1), ("learnerType('BOOSTING', {'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 3}, 'BAGGED', None)", 0, 2)]
arl to insert <deap.gp.Primitive object at 0x7f205264e6d8> original arity 3 new arity 1
len individual after arl insert 4
individual after arl insert [('Learner', 2, 0), ('arl4', 1, 1), ('0', 0, 2), ("learnerType('BOOSTING', {'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 3}, 'BAGGED', None)", 0, 3)]
```
-    - Individuals with incorrect arities still appear in the population
     - Problem with occurrences code, not properly including the entire ARL
- PACE-ICE still causing issues
    - What we thought switching to PACE-ICE could help us with over Google Collab:
        - Faster Runs (ARLs code doesn't benefit from GPUs on PACE-ICE as much as NN/CV subteams do)
        - Longer Runs (Guide notes a 8 hour limit for PACE-ICE, Google Collab has a 12 hour limit)
        - No inactivity clicking script
        - Not terribly difficult to switch to with the new guide (We have now spent 3 weeks trying to get it to work)
    - Switching back to Google Collab after giving PACE-ICE one more try
    - Stocks data has been migrated into our repo and we're now ready to do those runs

|Task|Status|Assigned Date|Due Date|Date Completed|
|----|------|-------------|--------|--------------|
|Fix contract ARLs method|Complete|Sep 27|Oct 3|Oct 2|
|Investigate add_all_subtrees problem|Complete|Sep 27|Oct 3|Oct 1|
|Do runs of extended ARLs|Complete|Sep 27|Oct 3|Oct 3|

### Week 5: Sep 20
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

#### Code Commits
- [Changes](https://github.gatech.edu/vhuang31/emade/commit/fd1794ba0ccabe5c3edbf8205653ba6cd9adb6c2)
    - Revert changes to titanic input XML file causing crashes

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



