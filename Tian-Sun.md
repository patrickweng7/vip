
== Tian Sun ==

Team Member: Tian Sun

Email: tsun90@gatech.edu

Cell Phone: 765-409-7220

Interests: Machine Learning, Python, Artificial Intelligence

== Self Evaluation ==
[[files/VIP AAD notebook evaluation.pdf|none|thumb]]

== Jan 20, 2021 ==

=== Team meeting notes: ===
* Course introduction, syllabus, introduction to notebooks.
* Evolutionary and bio-inspired algorism, a massive amount of solutions will evolve with small mutations to achieve a new generation of algorism toward our objection.
* process
** individual: One candidate in a large collection of solutions
** group: a group of individual solutions
** Objective : values that need to be maximized
** Fitness: comparision between different individual
** Evaluation: process to give a score of an individual to compute fitness
** Selection: chose individuals with highest score to produce next generation
** Crossover(Mating): produce next generation
** Mutate: result new generation will have certain differences(changes) based on their parents.
* introduction to One Max Problem

=== '''Subteam/'''Lab note: ===
'''One Max Problem'''
* Objective:  find a bit string containing all 1s with a set length
* Tools: DEAP Python library
* define the fitness objective from DEAP's base
** we define tuple for single maximize objective as (1.0)
** we define tuple for single minimize objective as (-1.0)
** we define tuple for the multi-objective problem as (1.0, 1.0)
* then use a random generator which produces either 0 or 1 to generate an individual and add into entire population which include 300 individuals
* next, we let the entire population go through genetic algorithms.
** evaluation process: map each individual into population list and assign fitness value to each of them
** then we set up evolution loops that go through 40 generations through the selection process to select an individual for crossover and cloning them to work on new instances.
** using mate and mutate functions we will mate two individuals with a 50% probability and mutates an individual with a 20% probability.
** finally printout min, max, avg, Std of each generation.
* Algorism ran for 39 generations  to achieve objection which is a max of 100
'''The N Queens Problem'''
* Objective: determine a configuration of n queens on a nxn chessboard such that no queen can be taken by one another where all queens are at the same colum

* we follow generic algorism as one max problem does however this time we define tuple to minimize
* The algorism ran for 100 generations achieved a result with a minimum at 1.0
[[files/Lab1 Graph.png|thumb|none]]

=== Action items: ===
{| class="wikitable"
!Task
!Current Status
!Date Assigned
!Suspense Date
!Date Resolved
|-
|Setup Notebook and review Syllabus
|Finished
|Jan 20, 2021
|Jan 27, 2021
|Jan 24, 2021
|-
|Setup GT GitHub and Anaconda on Laptop
|Finished
|Jan 20, 2021
|Jan 27, 2021
|Jan 24, 2021
|-
|Join our Slack group
|Finished
|Jan 20, 2021
|Jan 27, 2021
|Jan 24, 2021
|-
|Perform Lab 1 (Genetic Algorithms/DEAP)
|Finished
|Jan 20, 2021
|Jan 27, 2021
|Jan 26, 2021
|}

== Jan 27, 2021 ==

=== Team Meeting Note ===
Genetic Programing
* In this time individual is the evaluator itself
** the individual will take data and return results of evaluation
** we can represent this algorism with trees
** during crossover subtrees are exchanged to produce a children
** mutation involves following behavior
*** change a node
*** exchange a node
*** delete/add a node
** Symbolic Regression
*** as example f(x) = sin(x)
**** primitives: +,-,*,/,variables
**** use Taylor series of sin(x) to solve this one
***** we use primitives to represent Taylor series
**** EDAME: we want the computer to do itself as much as possible
***** as an example get more possible primitives that can help achieve the objective

=== Lab Note ===

===== Symbolic Regression =====
* This lab generates a function that models out objective function the best way it can, we will utilize Genetic Programing technique to achieve this object.
** Use a binary tree to represent algorism, each node is primitive data.
** nodes will be altered at each generation Multi-Objective Genetic Programming
** the code structure is similar to previous algorism which consist evaluation, selection, mutate process
** the fitness of the function will be determined based on its output with expectation, we will calculate fitness score as mean squares error, the objective of this algorism is to minimize mean squares error.
** two primitive types added are:
*** add
*** divide
** mutation method added is
*** mutInsert
**** This funtion will insert a new primitive node make original subtree become one of the children.
** Result of implementation: [[files/Lab2 result.png|none|thumb]][[files/Lab 2 result 2.png|none|thumb]]

==== Action Item: ====
{| class="wikitable"
!Task
!Current Status
!Date Assigned
!Suspense Date
!Date Resolved
|-
|Perform Lab 2 (Symbolic Regression)
|Finished
|Jan 27, 2021
|Feb 03, 2021
|Feb 02, 2021
|}

== Feb 03, 2021 ==

=== Team Meeting Note ===
Multiple Objective 
* Objective space
** space containing set of objectives to achieve
* Classify Algorism
** some item will be classified correctly some would not
** we want an algorism to eliminate such error and achieve all corrrect result
** There are 4 objectives: all positive will classified as positive and non of them classified as negative, all negative will classified as negative and non of them classified as positive.
** Objective space: minimize false-positive and false-negative rates.
** Accuracy: (TP+TN)/Total Number
** Precision = TP/(TP+NP)
** False Discovery Rate = 1 - Precision
* Pareto Optimality
** a individual which no other individual inside the population outperforms the individual on all objectives.
* Nondominated Sorting Generic Algorism II
** a set of Pareto optimal individual would be a good solution.
* Strength Pareto Evolutionary Algorism II
** Strength of Pareto is how many other points in the population it dominates
** Rank: sum of sterngre of individuals that dominates it.

=== Multi-Objective Genetic Programming ===
[[files/Muti Objective 1.png|none|thumb]]
[[files/Muti Objective 2.png|none|thumb]]
[[files/Muti Objective 3.png|none|thumb]]
AUC: 2.3841416372199005
{| class="wikitable"
!Task
!Current Status
!Date Assigned
!Suspense Date
!Date Resolved
|-
|Self Evaluation
|Finished
|Feb 03, 2021
|Feb 10, 2021
|Feb 8, 2021
|-
|LAB 2 muti Objective
|Finished
|Feb 03, 2021
|Feb 10, 2021
|Feb 9, 2021
|-
|NoteBook Update
|Finished
|Feb 03, 2021
|Feb 10, 2021
|Feb 10, 2021
|}

== Feb. 10, 2021 ==

==== Main Meeting ====
Discussed Titanic dataset and assigned subteams.

==== SubTeam Meeting ====
* We decided to use the same preprocessing technique as example notebook presented during main meeting with minor changes made.
* I am optimizing false positives

==== Individual Work ====
* After reading sklearn documents I decided to use knn model,
* I setn_neighbors = 550, weights = 'distance'
* I achieved result of Accuracy of 0.7039228, False Positive of 6, and False Negative of 90, confusion matrix given following
[[files/Confusion Matrix Titanic.png|none|thumb]]
{| class="wikitable"
!Task
!Current Status
!Date Assigned
!Suspense Date
!Date Resolved
|-
|Preprocessing
|Finished
|Feb 10, 2021
|Feb 17, 2021
|Feb 12, 2021
|-
|Model construction
|Finished
|Feb 10, 2021
|Feb 17, 2021
|Feb 17, 2021
|-
|NoteBook Update
|Finished
|Feb 10, 2021
|Feb 17, 2021
|Feb 17, 2021
|}

== Feb 17, 2021 ==

==== Team meeting ====
assigned titanic problems using generic algorithm 

==== subteam meeting Feb 19 ====
we started working on the generic algorism
* start with visual studio’s share code
* coded pre-processing processes
* we decided using one hot encoder to process sex and emitted catalog
* finished generic programming algorithms with a successful result at low efficiency
* final result: FP 0, fn 180, ACC 0.79
* we will work on this individually and resume group work at Sunday 

==== subteam meeting Feb 21 ====
* decided use label encoding for sex and emitted catalog as this will provided higher efficiency and higher accuracy rate at end
* We tweak our mutation model and fitness calculation try to optimize objective space.
* we have achieved a great perato front and ACC at 0.83
* assign task to take about several interesting finding during group presentation.
{| class="wikitable"
!Task
!Current Status
!Date Assigned
!Suspense Date
!Date Resolved
|-
|Preprocessing
|Finished
|Feb 17, 2021
|Feb 25, 2021
|Feb 19, 2021
|-
|Model construction
|Finished
|Feb 17, 2021
|Feb 25, 2021
|Feb 21, 2021
|-
|Presentation
|Finished
|Feb 17, 2021
|Feb 25, 2021
|Feb 25, 2021
|}

== Feb 25, 2021 ==

==== Main Meeting ====
Team presentation
{| class="wikitable"
!Task
!Current Status
!Date Assigned
!Suspense Date
!Date Resolved
|-
|EMADE installation
|Finished
|Feb 25, 2021
|March 5, 2021
|March 4, 2021
|-
|Installation Of MySqL
|Finished
|Feb 25, 2021
|March 5, 2021
|March 5, 2021
|-
|Installation of Anoconda
|Finished
|Feb 25, 2021
|March 5, 2021
|March 5, 2021
|-
|peer evaluation
|Finished
|Feb 25, 2021
|March 5, 2021
|March 5, 2021
|}

== March 3, 2021 ==
'''Main Meeting'''

Group 4, 5 are present their Titanic dataset assignment.

Start with EMADE instructions, the final presentation set for March 22, 2021, we will work on EMADE from this week until March 22 presentation.

=== Sub Team Meeting at March 5, 2021 ===
* Our team meeting today to figure out what would we do for this assignment.
* We started with setting up MySql Server, We have encountered several problems,
** First, we started to ensure each member have finished the installation of EMADE and MySql.
** We then attempt to set up a MySql server and let each member connect to it, however, the first attempt hse failed.
** Through I am able to create a connection with Workbench but the system keep saying I'm not on Windows while I am, other members are unable to set up a connection at all.
** As result, after few minutes of research Justin able to set up a new server, while the majority of the team are able to connect, I'm receiving ERROR 2003(60) when an attempt to connect.
* I will continue to search for a solution while my teammates attempt to run EMADE.
* GROUP WORKS
** After connection successful, we created a new schema(database) called Titanic and several tables for EMADE
** EMADE runs and outputs 6 pareto-optimal frontier datasets and returned an error to us.
** without any luck to resolve the error after several attempts we decided to wait for Wednesday's class.
* INDIVIDUAL WORKS
** Try to find a solution to ERROR 2003
*** Find it's necessary to let network firewall allow port 3306's network ingressive.
*** FInd there is a file "my.cnf(my.bin on Windows) may have IP-binding disallow the external connection
*** fixed all those prementioned problems and switched from Workbench to Sql pro for UI interface.
*** Will tested when the remote server is started at Wednesday

==== Action Table ====
{| class="wikitable"
!Task
!Current Status
!Date Assigned
!Suspense Date
!Date Resolved
|-
|MySql Connection
|Finished
|March 3, 2021
|March 22, 2021
|March 10, 2021
|-
|debug of Mysql
|Finished
|March 3, 2021
|March 22, 2021
|March 10, 2021
|-
|EMADE running(first stage)
|Finished
|March 3, 2021
|March 22, 2021
|March 10, 2021
|-
|EMADE debug
|Finished
|March 3, 2021
|March 22, 2021
|March 10, 2021
|}

== March 10, 2021 ==

==== Main Meeting ====
* a working meeting today.
* A bug for deap on newest version, it's necessary to change to version 1.2.2.
* INDIVIDUAL
** I am able to establish a connection and run EMADE as a worker process after connecting to school VPN, the reason behind this is unclear.
* GROUP
** the error encountered is not a major problem we will proceed with our paredo-optimal result.

==== Sub Team Meeting March 12, 2021 ====
* We just meet today to see if each of us are able to establish a connection and ensure after every member joined our paredo result are still good.
* We weren't able to generate a visualization of paredo fronts, we decided to wait for next Wednesday for questions.

==== Action Table ====
{| class="wikitable"
!Task
!Current Status
!Date Assigned
!Suspense Date
!Date Resolved
|-
|Visulization Result
|In progress
|March 14, 2021
|March 22, 2021
|Pending
|-
|deap 1.2.2 installation
|Finished
|March 10, 2021
|March 22, 2021
|March 14, 2021
|-
|EMADE running(After debug)
|Finished
|March 5, 2021
|March 22, 2021
|March 14, 2021
|-
|NoteBook Update
|Finished
|March 3, 2021
|March 22, 2021
|March 17, 2021
|}

== March 17, 2021 ==

==== Main Meeting ====
* Another work session
* Demonstrated how to visualize our results.
* Meet again this Friday to finish up visualization and create the presentation

==== Team Meeting ====
* Team meeting to finalize our result
* Take the output, use Jupiter code to generate a graph of Pareto front to visualize our data.
* Instead to take the raw result from EMADE, GP, and individual algorisms we decided to make a False-negative rate and false positive rate graph.
* Discussed the distribution of presentation and content we wish to address.
* I will be responsible to compare three methods we have used on the Titanic problem.
* Discussed why that EMADE although spend too much time and consume enormous computational resource the result is still not best.
* Individual work:
** looks into detail of each individual output.
** check results from previous semesters to see if we are a special case, it appears that EMADE generates worse results in a common situation.
** Ideas for poor EMADE result:
*** EMADE runs for only so many generations, we may be able to improve the result should the number of generations increases.
*** majority of result from EMADE is attempting to minimize false positives and each optimal individuals are better than general programing result. The poor overall result comes from the individuals on the false-negative side, EMADE may centralize to minimize false positives in early generations cause the poor general results.
** practice for midterm presentation.

==== Action Table ====
{| class="wikitable"
!Task
!Current Status
!Date Assigned
!Suspense Date
!Date Resolved
|-
|Visulization Result
|Finished
|March 14, 2021
|March 22, 2021
|March 19, 2021
|-
|Presentation Set up
|Finished
|March 17, 2021
|March 22, 2021
|March 21, 2021
|}

== March 22, 2021 ==

==== Main Meeting ====
* Today is the midterm presentation of all existing VIP groups include Bootcamp students.
* Presentation was given from stock group, Modularity group, NLP group, ezCGP group, and five Bootcamp groups.
* First semesters are expected to choose a group to join for the rest of the semester.
* I will join Monday meetings from now on.

==== Action Table ====
{| class="wikitable"
!Task
!Current Status
!Date Assigned
!Suspense Date
!Date Resolved
|-
|Choice team to Join
|Finished
|March 22, 2021
|March 29, 2021
|March 25, 2021
|}

== March 29, 2021 ==

==== Main Meeting ====
* I have been assigned to the Modularity group.

==== Team Meeting ====
* First team meeting as a modularity team member.
* Mainly an introduction section.
* Introduced with a modified EMADE GitHub repo to implement ARL, and collab which would be used for any future EMADE runs.
* Future work for the rest of the semester is to focus on MNIST Experiments.

* Another Team meeting on Sunday
** Addressed some error encountered by the first semester during collab run
** I attempt to run EMADE for 10 generations using colab locally.

==== Individual ====
* Set up new Github repo on my machine.
* Use the given google collab to run MINST locally.
* Start read some recommended Literature, following is a list of recommendation for ARLs:
** Discovery of Subroutines in Genetic Programming
** Towards Automatic Discovery of Building Blocks in Genetic Programming
** An Analysis of Automatic Subroutine Discovery in Genetic Programming.

==== Action Table ====
{| class="wikitable"
!Task
!Current Status
!Date Assigned
!Suspense Date
!Date Resolved
|-
|Set up Github repo
|Finished
|March 29, 2021
|April 5, 2021
|March 31, 2021
|-
|Set up colab
|Finished
|March 29, 2021
|April 5, 2021
|April 1, 2021
|-
|Read Literature
|In Progress
|March 29, 2021
|April 27, 2021
|
|}

== April 5, 2021 ==

==== Main Meeting ====
* Introduced with Statistical analysis
** A/B testing: make a control run, then change some variables we want to investigate and perform an experimental run to see the effect.
** Introduced standard deviation
* Hypothesis testing
** t-testing is a great way to test how far off we are from the mean value.
** p-value may be calculated from t-testing results with char.
** if p< 0.05 we have a good result.
** we can use Welch's t-test under the condition that we don't have a great deal of information.
*** This is a good test to test a hypothesis where two groups of data resulted in equal mean values.
** Note should two distributions are not equal we want to use a 2-tailed test with a higher t-statistic.
==== Team Meeting ====
* For first semesters:
** Read more literature for more understanding of the principles of ARL.
** run some baseline MNIST dataset.
** run as worker process and contribute processing power.

*NOTE: notes of work and team meeting are taken locally as the access to the wiki is not stable at this point, Edit functionality returns error 500.
==== Action Table ====
{| class="wikitable"
!Task
!Current Status
!Date Assigned
!Suspense Date
!Date Resolved
|-
|Run MNIST locally and on remote
|Finished
|April 5, 2021
|April 12, 2021
|April 10, 2021
|-
|Read Literature
|In Progress
|March 29, 2021
|April 27, 2021
|
|}

== April 12, 2021 ==
==== Team Meeting ====
* We are unable to have many successful individuals for seeding and analysis.
** Our current hypothesis of the problem is:
*** the failure of individuals comes from poor objectives and poor seeds.
*** Team will work to improve results based on this hypothesis.
* First Semester student task:
** creating new objectives
** Finding better-succeded individuals
==== Action Table ====
{| class="wikitable"
!Task
!Current Status
!Date Assigned
!Suspense Date
!Date Resolved
|-
|Run MNIST locally and on remote for more individuals
|Finished
|April 12, 2021
|April 19, 2021
|April 18, 2021
|-
|Read Literature(final)
|Finished
|March 29, 2021
|April 27, 2021
|April 22, 2021
|}
== April 19, 2021 - Final Week ==
* Now approaching the final week.
==== Team Meeting ====
* Team will now concentrate on the analysis of current results with baseline run and 
* Team will take meetings in final week for practice presentation and any last-minute questions.
* first semester will contribute as worker process while returned student makes changes to databases and analyze results.
* Preparation for final presentation will start with next week.
* I will talk about pareto fronts of our results from tatanic and MNIST dataset.

==== Action Table ====
{| class="wikitable"
!Task
!Current Status
!Date Assigned
!Suspense Date
!Date Resolved
|-
|Preperation of final presentation
|Finished
|April 19, 2021
|April 30, 2021
|April 30, 2021
|-
|Final peer evaluation
|Finished
|April 19, 2021
|April 27, 2021
|April 26, 2021
|-
|Move all local Notebook entry to wiki
|Finished
|April 19, 2021
|April 30, 2021
|April 29, 2021
|}